"""
Sessão #54 — Avaliação cross-domain CUB-200, 1000 episodes 1 seed.

Carrega encoder Omniglot frozen + roda N episódios 5w1s no CUB test split.
ACC mean + IC95% via bootstrap (1000 resamples).

Single seed nesta sessão (seed=42). Multi-seed fica pra #55.

Não modifica scripts existentes. Reusa CUBDataset, CUBEpisodeSampler,
proto_episode_eval do pipeline criado em #53.

Uso:
    python eval_crossdomain.py --device cuda --encoder c3 --seed 42 --episodes 1000
    python eval_crossdomain.py --device cuda --encoder protonet --seed 42 --episodes 1000
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
from c3_protonet_sparse import ProtoEncoder, ProtoEncoderSparse


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def load_encoder(encoder_kind: str, ckpt_path: Path, k_wta: int, device: torch.device,
                  resolution: int = 28):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    if encoder_kind == "c3":
        encoder = ProtoEncoderSparse(k=k_wta).to(device)
    elif encoder_kind == "protonet":
        encoder = ProtoEncoder().to(device)
    elif encoder_kind == "cub_retrained":
        if resolution == 28:
            encoder = ProtoEncoder().to(device)
        elif resolution == 84:
            from train_cub_protonet import ProtoEncoderRGB
            encoder = ProtoEncoderRGB().to(device)
        else:
            raise ValueError(f"unsupported resolution: {resolution}")
    else:
        raise ValueError(f"unknown encoder_kind: {encoder_kind}")

    encoder.load_state_dict(state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def run_one_seed(encoder_kind: str, ckpt_path: Path, k_wta: int, dataset, device,
                  seed: int, episodes: int, n_way: int, k_shot: int, n_query: int,
                  resolution: int = 28) -> dict:
    encoder = load_encoder(encoder_kind, ckpt_path, k_wta, device, resolution=resolution)
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    if n_trainable != 0:
        raise RuntimeError(f"Encoder has {n_trainable} trainable params; should be frozen")

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
    p.add_argument("--encoder", choices=("c3", "protonet", "cub_retrained"), default="c3")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--k-wta", type=int, default=16)
    p.add_argument("--resolution", type=int, choices=(28, 84), default=28,
                   help="Input resolution. 28=grayscale (compat C3); 84=RGB (literatura).")
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=1)
    p.add_argument("--n-query", type=int, default=5)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="One or more seeds.")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[eval] device={device} encoder={args.encoder} resolution={args.resolution} "
          f"seeds={args.seeds} eps={args.episodes}")

    repo_root = Path(__file__).resolve().parent.parent
    print(f"[eval] loading CUB-200 test split (res={args.resolution})...")
    dataset = CUBDataset(split="test", resolution=args.resolution, verbose=True)

    chance = 1.0 / args.n_way
    runs = []
    for seed in args.seeds:
        if args.ckpt is not None:
            ckpt_path = Path(args.ckpt)
        elif args.encoder == "c3":
            ckpt_path = repo_root / "experiment_01_oneshot" / "checkpoints" / f"c3_kwta_k{args.k_wta}_seed{seed}.pt"
        elif args.encoder == "protonet":
            ckpt_path = repo_root / "experiment_01_oneshot" / "checkpoints" / f"protonet_omniglot_seed{seed}.pt"
        else:  # cub_retrained
            ckpt_path = repo_root / "experiment_01_oneshot" / "checkpoints" / f"protonet_cub_{args.resolution}_seed{seed}.pt"

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for seed={seed}: {ckpt_path}."
            )
        print(f"[eval] seed={seed} ckpt={ckpt_path.name}")
        r = run_one_seed(args.encoder, ckpt_path, args.k_wta, dataset, device,
                          seed=seed, episodes=args.episodes, n_way=args.n_way,
                          k_shot=args.k_shot, n_query=args.n_query, resolution=args.resolution)
        z = (r["mean"] - chance) / (r["std"] / np.sqrt(r["n_eps"])) if r["std"] > 0 else float("inf")
        print(f"  seed={seed}: ACC={r['mean']*100:.2f}% IC95%[{r['lo']*100:.2f}, {r['hi']*100:.2f}] "
              f"z={z:.1f} std/ep={r['std']*100:.2f}% t={r['elapsed']:.1f}s")
        runs.append(r)

    print(f"\n[eval] === Summary across {len(runs)} seeds ===")
    means = np.array([r["mean"] for r in runs])
    if len(runs) >= 2:
        inter_mean = float(means.mean())
        inter_std = float(means.std(ddof=1))
        # IC95% inter-seed via bootstrap das means
        rng = np.random.default_rng(0)
        boots = rng.choice(means, size=(1000, len(means)), replace=True).mean(axis=1)
        inter_lo = float(np.quantile(boots, 0.025))
        inter_hi = float(np.quantile(boots, 0.975))
        print(f"  encoder={args.encoder} mean across seeds: {inter_mean*100:.2f}% +/- {inter_std*100:.2f}% "
              f"(inter-seed std), IC95% inter-seed [{inter_lo*100:.2f}, {inter_hi*100:.2f}]")
    else:
        print(f"  encoder={args.encoder} single seed: {means[0]*100:.2f}%")
    print(f"  chance={chance*100:.1f}%")


if __name__ == "__main__":
    main()
