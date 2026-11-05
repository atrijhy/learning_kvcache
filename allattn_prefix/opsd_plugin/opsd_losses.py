"""Extra OPSD distillation losses registered into verl's registry (see sitecustomize.py).

forward_kl_topk_focus: forward_kl_topk restricted to each sequence's highest-KL tokens. On TACO rollouts the per-token
KL(teacher || student) is very concentrated (top 5% of tokens hold ~72% of it): the decisive solution-choice tokens are
diluted by thousands of agreeing tokens. Keeps per sequence only the tokens whose KL is in the top OPSD_FOCUS_TOP_FRAC
(default 0.05) fraction of that sequence's valid tokens; the selection is detached (Adam makes the scale irrelevant).
"""
import os
from typing import Any

import torch


def register():
    from verl.trainer.distillation import losses as L

    if "forward_kl_topk_focus" in L.DISTILLATION_SETTINGS_REGISTRY:
        return

    @L.register_distillation_loss(L.DistillationLossSettings(names=["forward_kl_topk_focus"], use_topk=True))
    def compute_forward_kl_topk_focus(config, distillation_config, model_output: dict, data) -> tuple[torch.Tensor, dict[str, Any]]:
        losses, metrics = L.compute_forward_kl_topk(config, distillation_config, model_output, data)
        frac = float(os.environ.get("OPSD_FOCUS_TOP_FRAC", "0.05"))
        rm = data["response_mask"]
        mask = rm.bool().to_padded_tensor(False) if rm.is_nested else rm.bool()
        with torch.no_grad():
            keep = torch.zeros_like(mask)
            for i in range(losses.shape[0]):
                vals = losses[i][mask[i]]
                if vals.numel() == 0:
                    continue
                k = max(1, int(round(frac * vals.numel())))
                thr = vals.float().topk(k).values[-1]
                keep[i] = mask[i] & (losses[i] >= thr)
            metrics["distillation/focus_kept_frac"] = (keep.sum() / mask.sum().clamp_min(1)).item()
            metrics["distillation/focus_kl_share"] = (
                (losses * keep).sum() / (losses * mask).sum().clamp_min(1e-12)).item()
        return losses * keep.to(losses.dtype), metrics
