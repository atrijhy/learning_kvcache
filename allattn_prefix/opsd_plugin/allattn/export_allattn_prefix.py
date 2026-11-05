"""Export a trainable-only verl actor checkpoint of the all-layer residual attention prefix model into a copy of the
prepared model dir (base weights hardlinked; residual_attention_prefix.safetensors replaced). Same gathering as
workspace/prefix_training/export_hybrid_prefix.py, single prefix file.
Usage: export_allattn_prefix.py --checkpoint .../global_step_N/actor --prepared DIR --output DIR"""
import argparse, os, shutil, sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
sys.path.insert(0, os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[3])) + "/prefix_tuning/virtual_prefix")
from merge_prefix_checkpoint import _gather_tensor  # noqa: E402
PF = "residual_attention_prefix.safetensors"
ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", type=Path, required=True)
ap.add_argument("--prepared", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
a = ap.parse_args()
if a.output.exists(): raise SystemExit(f"refusing to overwrite {a.output}")
shards = sorted(a.checkpoint.glob("model_world_size_*_rank_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
states = [torch.load(p, map_location="cpu", weights_only=False) for p in shards]
with safe_open(a.prepared / PF, framework="pt") as h:
    init = {k: h.get_tensor(k) for k in h.keys()}
trained = {}
for name in states[0]:
    key = next((k for k in init if name.endswith(k.split("model.", 1)[-1])), None)
    if key is None: raise SystemExit(f"checkpoint tensor {name} has no prepared counterpart")
    t = _gather_tensor([s[name] for s in states], name)
    if tuple(t.shape) != tuple(init[key].shape): raise SystemExit(f"shape mismatch {name}")
    trained[key] = t.to(init[key].dtype)
if set(trained) != set(init): raise SystemExit(f"missing tensors: {sorted(set(init) - set(trained))[:5]}")
shutil.copytree(a.prepared, a.output, ignore=shutil.ignore_patterns(PF),
                copy_function=lambda s, d: os.link(s, d) if os.path.basename(s).startswith("model.safetensors-") else shutil.copy2(s, d))
save_file({k: v.contiguous() for k, v in trained.items()}, a.output / PF, metadata={"format": "pt"})
for grp in ("linear_attn.prefix_key", "linear_attn.prefix_value", "self_attn.prefix_key", "self_attn.prefix_value"):
    ks = [k for k in trained if grp in k]
    d = torch.cat([(trained[k].float() - init[k].float()).flatten() for k in ks])
    print(f"{grp:<26}{len(ks):>4} max|delta| {d.abs().max():.5f} mean|delta| {d.abs().mean():.2e}")
print(f"wrote {a.output}")
