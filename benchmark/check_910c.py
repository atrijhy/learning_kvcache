"""Check the existing BFCL/tau2 pipeline and Qwen3 prefix on CPU only."""
from __future__ import annotations
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from prefix_tuning.virtual_prefix.export_qwen3_residual_kv_prefix_checkpoint import checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--model", type=Path, default=Path("/home/liyu.468/models/Qwen3-4B"))
    args = parser.parse_args()
    official = Path(os.getenv("BFCL_OFFICIAL_ROOT", HERE / "bfcl_v3/official"))
    sys.path.insert(0, str(official))
    from bfcl_eval.__main__ import cli  # noqa: F401
    from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING
    counts = {}
    for category in TEST_COLLECTION_MAPPING["multi_turn"]:
        data = HERE / "bfcl_v3/data/envscaler_v3" / f"BFCL_v3_{category}.json"
        rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
        assert len(rows) == 200, (category, len(rows))
        counts[category] = len(rows)
        assert (official / "bfcl_eval/data/possible_answer" / data.name).exists()
    tau_root = HERE / "tau2"
    tau_env = Path(os.getenv("TAU2_VENV", tau_root / ".venv-910c"))
    env = dict(os.environ, LITELLM_LOCAL_MODEL_COST_MAP="True")
    code = '''
import json
from tau2.domains.retail.environment import get_tasks as retail_tasks, get_environment as retail_env
from tau2.domains.airline.environment import get_tasks as airline_tasks, get_environment as airline_env
result = {}
for name, tasks_fn, env_fn, expected in [("retail", retail_tasks, retail_env, 114), ("airline", airline_tasks, airline_env, 50)]:
    tasks = tasks_fn()
    assert len(tasks) == expected, (name, len(tasks))
    environment = env_fn()
    assert environment.get_tools()
    result[name] = {"tasks": len(tasks), "tools": len(environment.get_tools())}
print(json.dumps(result))
'''
    output = subprocess.check_output([str(tau_env / "bin/python"), "-c", code], cwd=tau_root / "official", env=env, text=True)
    tau = json.loads(output.splitlines()[-1])
    checkpoint = checkpoint_path(args.checkpoint)
    config = json.loads((checkpoint / "prefix_config.json").read_text())
    base = json.loads((args.model / "config.json").read_text())
    expected = {f"model.layers.{i}.self_attn.prefix_{kind}_tokens" for i in range(base["num_hidden_layers"]) for kind in ["key", "value"]}
    from safetensors import safe_open
    import numpy as np
    with safe_open(str(checkpoint / "model.safetensors"), framework="np") as state:
        assert set(state.keys()) == expected
        for key in state.keys():
            tensor = state.get_tensor(key)
            assert tensor.shape == (config["num_virtual_tokens"], base["hidden_size"])
            assert np.isfinite(tensor).all()
    from dotenv import dotenv_values
    has_ark_key = bool(os.getenv("ARK_API_KEY") or dotenv_values(ROOT / ".env").get("ARK_API_KEY"))
    assert has_ark_key, "ARK_API_KEY is missing"
    report = {"ready_cpu": True, "architecture": platform.machine(), "python": sys.executable,
              "versions": {name: metadata.version(name) for name in ["torch", "torch-npu", "transformers", "openai"]},
              "entrypoint": "benchmark/run.sh", "bfcl_v3_multi_turn": counts, "tau2": tau,
              "tau2_commit": (tau_root / "official/.source_commit").read_text().strip(),
              "ark_key_present": has_ark_key, "checkpoint_validated": str(checkpoint),
              "prefix_tensors": len(expected), "npu_inference_tested": False}
    (HERE / "runtime").mkdir(exist_ok=True)
    (HERE / "runtime/910c_readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
