"""Route B (Continual Learning Clustering in Frozen Base x_t Gate-Input Space):
- Extracts exact self_attn input hidden states x_t (input_layernorm output) on supervised tokens.
- Autonomously selects K via cosine silhouette (no hardcoded cluster count).
- Routes new incoming experiences by cosine similarity to existing centroids:
    * max_cos >= tau -> route to existing cluster C_k (for single-branch O(1) refresh)
    * max_cos < tau  -> route to OTHER buffer (for recursive new-branch incubation)
- Optionally uses Qwen3-4B to generate human-readable names & definitions for each discovered cluster.
"""
import argparse
import gc
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def kmeans_cosine(X: np.ndarray, K: int, n_init: int = 20, max_iter: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    N = len(X)
    best_inertia, best_labels, best_cents = -1e18, None, None
    for _ in range(n_init):
        idxs = [int(rng.integers(0, N))]
        for _ in range(1, K):
            sims = X @ X[idxs].T
            dists = np.clip(1.0 - sims.max(axis=1), 0.0, None)
            probs = dists / dists.sum() if dists.sum() > 0 else np.ones(N) / N
            idxs.append(int(rng.choice(N, p=probs)))
        cents = X[idxs].copy()
        for _ in range(max_iter):
            labels = (X @ cents.T).argmax(axis=1)
            new_cents = np.zeros_like(cents)
            for k in range(K):
                mask = labels == k
                if mask.any():
                    c = X[mask].mean(axis=0)
                    new_cents[k] = c / max(1e-12, np.linalg.norm(c))
                else:
                    new_cents[k] = X[int(rng.integers(0, N))]
            if np.allclose(cents, new_cents, atol=1e-6):
                cents = new_cents
                break
            cents = new_cents
        inertia = float((X * cents[labels]).sum())
        if inertia > best_inertia:
            best_inertia, best_labels, best_cents = inertia, labels.copy(), cents.copy()
    return best_labels, best_cents


def silhouette_cosine(X: np.ndarray, labels: np.ndarray) -> float:
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    Dmat = 1.0 - np.clip(X @ X.T, -1.0, 1.0)
    sils = []
    for i in range(len(X)):
        li = labels[i]
        same = (labels == li)
        same[i] = False
        a = Dmat[i, same].mean() if same.any() else 0.0
        b = min(Dmat[i, labels == lj].mean() for lj in uniq if lj != li)
        sils.append((b - a) / max(a, b, 1e-12))
    return round(float(np.mean(sils)), 4)


def auto_select_k_cosine(X: np.ndarray, k_min: int = 2, k_max: int = 5, min_cluster_frac: float = 0.08, seed: int = 42):
    """Autonomously choose K in [k_min..k_max] by cosine silhouette (filtering tiny degenerate buckets)."""
    mean_c = X.mean(axis=0, keepdims=True)
    mean_c = mean_c / max(1e-12, np.linalg.norm(mean_c))
    best_k, best_sil, best_lb, best_cents = 1, -1.0, np.zeros(len(X), dtype=int), mean_c
    k_scores = {}
    for K in range(k_min, min(k_max, len(X) - 1) + 1):
        lb, cents = kmeans_cosine(X, K, seed=seed)
        cnts = Counter(int(x) for x in lb)
        min_frac = min(cnts.values()) / len(X)
        sil = silhouette_cosine(X, lb)
        k_scores[K] = {"sil": sil, "min_frac": round(min_frac, 3), "sizes": dict(cnts)}
        if min_frac >= min_cluster_frac and sil > best_sil:
            best_k, best_sil, best_lb, best_cents = K, sil, lb, cents
    return best_k, best_lb, best_cents, k_scores


@torch.no_grad()
def extract_xt_features(base_model: Path, encoded_rows: list, target_layers=(17, 23, 29)) -> np.ndarray:
    """Single frozen forward pass extracting L2-normalized mean-pooled x_t (input_layernorm output) on supervised tokens."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    captured = {}
    hooks = []
    for li in target_layers:
        def make_hook(idx):
            def fn(_mod, _inp, out):
                captured[idx] = out.detach()
            return fn
        hooks.append(model.model.layers[li].input_layernorm.register_forward_hook(make_hook(li)))

    N, H = len(encoded_rows), model.config.hidden_size
    out = np.zeros((N, H), dtype=np.float32)
    with torch.inference_mode():
        for idx, row in enumerate(encoded_rows):
            ids, pos = row[1], row[3]
            p_idx = torch.tensor([p - 1 for p in pos if p > 0], device="cuda")
            model.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
            vecs = []
            for li in target_layers:
                v = captured[li][0, p_idx].float().mean(dim=0)
                vecs.append(v / v.norm().clamp_min(1e-12))
            vm = torch.stack(vecs, dim=0).mean(dim=0)
            out[idx] = (vm / vm.norm().clamp_min(1e-12)).cpu().numpy()

    for hk in hooks:
        hk.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def route_or_reject(X_new: np.ndarray, cents: np.ndarray, cos_threshold: float) -> tuple[list[int], np.ndarray]:
    """Route each new item to nearest centroid (0..K-1) or -1 (OTHER) if max cosine < cos_threshold."""
    sims = X_new @ cents.T
    best_k = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)
    assigned = [int(k) if float(s) >= cos_threshold else -1 for k, s in zip(best_k, best_sim, strict=True)]
    return assigned, best_sim


def main():
    ap = argparse.ArgumentParser(description="Route B: Frozen Qwen3-4B x_t hidden-state continual clustering")
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--encoded-pt", type=Path, required=True)
    ap.add_argument("--split", choices=("train", "validation"), default="train")
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--ood-quantile", type=float, default=0.05)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    enc = torch.load(args.encoded_pt, map_location="cpu", weights_only=False)[args.split]
    X = extract_xt_features(args.base_model, enc)
    K, labels, cents, k_scores = auto_select_k_cosine(X, k_min=2, k_max=args.max_k)
    fit_sims = (X @ cents.T).max(axis=1)
    ood_thr = float(np.quantile(fit_sims, args.ood_quantile))

    np.savez_compressed(args.out_dir / "xt_clusters.npz", features=X, centroids=cents, labels=labels)
    summary = {
        "chosen_k": K,
        "ood_cosine_threshold": round(ood_thr, 5),
        "ood_quantile": args.ood_quantile,
        "cluster_sizes": {str(k): int(v) for k, v in sorted(Counter(int(x) for x in labels).items())},
        "k_search_scores": k_scores,
    }
    (args.out_dir / "xt_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
