"""
Sessão #55 — Pixel kNN cross-domain CUB-200 sanity floor.

Nearest-neighbor sobre pixels brutos (sem encoder, sem aprendizado).
Distância L2 entre support flatten (1, 28, 28) -> (784,) e query flatten.

Setup idêntico ao eval_crossdomain.py (5w1s5q, 1000 eps, multi-seed):
- Mesmo CUBDataset test split (cache 28×28 grayscale)
- Mesmo CUBEpisodeSampler (sampling determinístico via seed)
- IC95% bootstrap por seed, mean ± std inter-seed

Pixel kNN é o sanity floor: se C3 e ProtoNet baseline não batem
significativamente Pixel kNN, sinal cross-domain é trivial.

Uso:
    python eval_pixel_knn.py --device cuda --seeds 42 43 44 45 46 --episodes 1000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cub_data import CUBDataset
from episodes import CUBEpisodeSampler
from eval_crossdomain import bootstrap_ci


def pixel_episode_acc(episode, device: torch.device) -> float:
    """ProtoNet-like com pixels: prototype = mean per class no espaço pixel."""
    sup = episode.support.to(device)        # (N*K, 1, 28, 28)
    qry = episode.query.to(device)
    sup_lbl = episode.support_labels.to(device)
    qry_lbl = episode.query_labels.to(device)

    sup_flat = sup.flatten(1)               # (N*K, 784)
    qry_flat = qry.flatten(1)

    n_way = episode.n_way
    protos = torch.zeros(n_way, sup_flat.shape[1], device=device)
    for c in range(n_way):
        mask = sup_lbl == c
        protos[c] = sup_flat[mask].mean(dim=0)

    dists = torch.cdist(qry_flat, protos).pow(2)
    preds = dists.argmin(dim=-1)
    acc = (preds == qry_lbl).float().mean().item()
    return acc


def run_one_seed(dataset, device, seed: int, episodes: int,
                  n_way: int, k_shot: int, n_query: int) -> dict:
    sampler = CUBEpisodeSampler(
        dataset, n_way=n_way, k_shot=k_shot, n_query=n_query, seed=seed,
    )
    accs = []
    t0 = time.time()
    for _ in range(episodes):
        episode = sampler.sample()
        accs.append(pixel_episode_acc(episode, device))
    elapsed = time.time() - t0

    arr = np.array(accs)
    mean = float(arr.mean())
    std = float(arr.std())
    lo, hi = bootstrap_ci(arr)
    return {"seed": seed, "mean": mean, "std": std, "lo": lo, "hi": hi,
            "elapsed": elapsed, "n_eps": len(arr)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=1)
    p.add_argument("--n-query", type=int, default=5)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[pixel-knn #55] device={device} seeds={args.seeds} eps={args.episodes}")

    print(f"[pixel-knn #55] loading CUB-200 test split (28x28 grayscale)...")
    dataset = CUBDataset(split="test", verbose=True)

    chance = 1.0 / args.n_way
    runs = []
    for seed in args.seeds:
        r = run_one_seed(dataset, device, seed=seed, episodes=args.episodes,
                          n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query)
        z = (r["mean"] - chance) / (r["std"] / np.sqrt(r["n_eps"])) if r["std"] > 0 else float("inf")
        print(f"  [pixel-knn] seed={seed}: ACC={r['mean']*100:.2f}% IC95%[{r['lo']*100:.2f}, {r['hi']*100:.2f}] "
              f"z={z:.1f} std/ep={r['std']*100:.2f}% t={r['elapsed']:.1f}s")
        runs.append(r)

    means = np.array([r["mean"] for r in runs])
    print(f"\n[pixel-knn #55] === Summary across {len(runs)} seeds ===")
    if len(runs) >= 2:
        inter_mean = float(means.mean())
        inter_std = float(means.std(ddof=1))
        rng = np.random.default_rng(0)
        boots = rng.choice(means, size=(1000, len(means)), replace=True).mean(axis=1)
        inter_lo = float(np.quantile(boots, 0.025))
        inter_hi = float(np.quantile(boots, 0.975))
        print(f"  Pixel kNN cross-domain mean: {inter_mean*100:.2f}% +/- {inter_std*100:.2f}% "
              f"(inter-seed std), IC95% inter-seed [{inter_lo*100:.2f}, {inter_hi*100:.2f}]")
    else:
        print(f"  Pixel kNN cross-domain single seed: {means[0]*100:.2f}%")
    print(f"  chance={chance*100:.1f}%")


if __name__ == "__main__":
    main()
