"""Build an initial all-layer residual attention prefix model dir from base Qwen3.5-4B (base weights hard-linked).
GDN layers get N_GDN soft tokens, full-attention layers N_ATTN; keys ~ N(0, 0.02), values = 0 (identity at init).
Usage: build_allattn_model.py --base /root/kvmodels/Qwen3.5-4B --out DIR --gdn 512 --attn 64
(the author's sizes: GDN 512 / attention 64 -> Qwen3.5-4B-AllAttn-G512A64)"""
import argparse, json, os, shutil, torch
from safetensors.torch import save_file
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--gdn", type=int, default=512); ap.add_argument("--attn", type=int, default=64)
ap.add_argument("--seed", type=int, default=20261104); a = ap.parse_args()
ARCH = "Qwen3_5AllAttentionPrefixForConditionalGeneration"; PF = "residual_attention_prefix.safetensors"
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(a.out, exist_ok=False)
for f in os.listdir(a.base):
    s = os.path.join(a.base, f)
    if os.path.isfile(s):
        (os.link if f.startswith("model.safetensors-") else shutil.copy2)(s, os.path.join(a.out, f))
shutil.copy2(os.path.join(here, "../opsd_plugin/allattn/modeling_qwen3_5_all_attention_prefix.py"), a.out)
c = json.load(open(os.path.join(a.base, "config.json"))); H = c["text_config"]["hidden_size"]
g = torch.Generator().manual_seed(a.seed); st = {}
for i, t in enumerate(c["text_config"]["layer_types"]):
    mod, M = ("linear_attn", a.gdn) if t == "linear_attention" else ("self_attn", a.attn)
    st[f"model.language_model.layers.{i}.{mod}.prefix_key_tokens"] = (torch.randn(M, H, generator=g) * 0.02).to(torch.bfloat16)
    st[f"model.language_model.layers.{i}.{mod}.prefix_value_tokens"] = torch.zeros(M, H, dtype=torch.bfloat16)
save_file(st, os.path.join(a.out, PF), metadata={"format": "pt"})
c["architectures"] = [ARCH]; c["auto_map"] = {"AutoModelForImageTextToText": "modeling_qwen3_5_all_attention_prefix." + ARCH}
c["text_config"].update(residual_attention_prefix_num_virtual_tokens=a.attn, residual_gdn_prefix_num_virtual_tokens=a.gdn,
                        residual_attention_prefix_key_init_std=0.02)
c["hybrid_prefix"] = {"method": "all_layer_residual_attention_prefix", "gdn_num_virtual_tokens": a.gdn,
                      "attention_num_virtual_tokens": a.attn, "trainable_tensor_count": len(st),
                      "trainable_numel": sum(v.numel() for v in st.values())}
json.dump(c, open(os.path.join(a.out, "config.json"), "w"), indent=2)
idx = json.load(open(os.path.join(a.out, "model.safetensors.index.json"))); idx["weight_map"].update({k: PF for k in st})
json.dump(idx, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
print(a.out, len(st), c["hybrid_prefix"]["trainable_numel"])
