"""
Sessão #55 — Random encoder (Kaiming init) + k-WTA cross-domain CUB-200.

Sanity crítico: distingue 3 hipóteses sobre o sinal residual de C3 (~21.8%):
  (a) sinal residual real do treino Omniglot
  (b) artefato encoder treinado vs random — qualquer CNN forwarding produz isso
  (c) pré-processamento 28×28 grayscale destrói discriminabilidade (todos = chance)

Encoder = ProtoEncoderSparse(k=16) com Kaiming init padrão (default do
PyTorch nn.Conv2d) — pesos puramente aleatórios, congelados, sem treino.
Aplica k-WTA k=16 (mesmo do C3). Diferentes seeds geram diferentes init.

Setup idêntico ao eval_crossdomain.py:
- Mesmo CUBDataset test split (cache 28×28 grayscale)
- Mesmo CUBEpisodeSampler (sampling determinístico)
- 5 seeds × 1000 episodes
- IC95% bootstrap por seed, mean ± std inter-seed

Interpretação:
- random ACC <= pixel kNN  → encoder random pior que pixel direto: hipótese (c)
- random < C3 ≈ ProtoNet  → treino Omniglot agrega: hipótese (a)
- random ≈ C3 ≈ ProtoNet  → encoder forward (treinado ou não) é o que importa: hipótese (b)

Uso:
    python eval_random_encoder.py --device cuda --seeds 42 43 44 45 46 --episodes 1000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment_01_oneshot"))

from cub_data import CUBDataset
from episodes import CUBEpisodeSampler, proto_episode_eval
from c3_protonet_sparse import ProtoEncoderSparse
from eval_crossdomain import bootstrap_ci


def make_random_encoder(k_wta: int, seed: int, device: torch.device) -> torch.nn.Module:
    """Random Kaiming init congelado. Diferentes seeds = diferentes init."""
    torch.manual_seed(seed)
    encoder = ProtoEncoderSparse(k=k_wta).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def run_one_seed(dataset, device, seed: int, episodes: int, k_wta: int,
                  n_way: int, k_shot: int, n_query: int) -> dict:
    encoder = make_random_encoder(k_wta=k_wta, seed=seed, device=device)
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    if n_trainable != 0:
        raise RuntimeError(f"Random encoder has {n_trainable} trainable params")

    sampler = CUBEpisodeSampler(
        dataset, n_way=n_way, k_shot=k_shot, n_query=n_query, seed=seed,
    )
    accs = []
    t0 = time.time()
    for _ in range(episodes):
        episode = sampler.sample()
        acc, _ = proto_episode_eval(encoder, episode, device)
        accs.append(acc)
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
    p.add_argument("--k-wta", type=int, default=16)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=1)
    p.add_argument("--n-query", type=int, default=5)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[random-enc #55] device={device} k_wta={args.k_wta} seeds={args.seeds} eps={args.episodes}")

    print(f"[random-enc #55] loading CUB-200 test split (28x28 grayscale)...")
    dataset = CUBDataset(split="test", verbose=True)

    chance = 1.0 / args.n_way
    runs = []
    for seed in args.seeds:
        r = run_one_seed(dataset, device, seed=seed, episodes=args.episodes, k_wta=args.k_wta,
                          n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query)
        z = (r["mean"] - chance) / (r["std"] / np.sqrt(r["n_eps"])) if r["std"] > 0 else float("inf")
        print(f"  [random-enc] seed={seed}: ACC={r['mean']*100:.2f}% IC95%[{r['lo']*100:.2f}, {r['hi']*100:.2f}] "
              f"z={z:.1f} std/ep={r['std']*100:.2f}% t={r['elapsed']:.1f}s")
        runs.append(r)

    means = np.array([r["mean"] for r in runs])
    print(f"\n[random-enc #55] === Summary across {len(runs)} seeds ===")
    if len(runs) >= 2:
        inter_mean = float(means.mean())
        inter_std = float(means.std(ddof=1))
        rng = np.random.default_rng(0)
        boots = rng.choice(means, size=(1000, len(means)), replace=True).mean(axis=1)
        inter_lo = float(np.quantile(boots, 0.025))
        inter_hi = float(np.quantile(boots, 0.975))
        print(f"  Random encoder + k-WTA k={args.k_wta} mean: {inter_mean*100:.2f}% +/- {inter_std*100:.2f}% "
              f"(inter-seed std), IC95% inter-seed [{inter_lo*100:.2f}, {inter_hi*100:.2f}]")
    else:
        print(f"  Random encoder single seed: {means[0]*100:.2f}%")
    print(f"  chance={chance*100:.1f}%")


if __name__ == "__main__":
    main()
