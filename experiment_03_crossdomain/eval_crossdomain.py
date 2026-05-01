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


def load_encoder(encoder_kind: str, ckpt_path: Path, k_wta: int, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    if encoder_kind == "c3":
        encoder = ProtoEncoderSparse(k=k_wta).to(device)
    elif encoder_kind == "protonet":
        encoder = ProtoEncoder().to(device)
    else:
        raise ValueError(f"unknown encoder_kind: {encoder_kind}")

    encoder.load_state_dict(state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--encoder", choices=("c3", "protonet"), default="c3")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--k-wta", type=int, default=16)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=1)
    p.add_argument("--n-query", type=int, default=5)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[eval #54] device={device} encoder={args.encoder} seed={args.seed} eps={args.episodes}")

    repo_root = Path(__file__).resolve().parent.parent
    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
    elif args.encoder == "c3":
        ckpt_path = repo_root / "experiment_01_oneshot" / "checkpoints" / f"c3_kwta_k16_seed{args.seed}.pt"
    else:
        ckpt_path = repo_root / "experiment_01_oneshot" / "checkpoints" / f"protonet_omniglot_seed{args.seed}.pt"

    print(f"[eval #54] loading {ckpt_path.name} (frozen)")
    encoder = load_encoder(args.encoder, ckpt_path, args.k_wta, device)
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    if n_trainable != 0:
        raise RuntimeError(f"Encoder has {n_trainable} trainable params; should be frozen")

    print(f"[eval #54] loading CUB-200 test split (28x28 grayscale)...")
    dataset = CUBDataset(split="test", verbose=True)

    print(f"[eval #54] sampling {args.episodes} episodes {args.n_way}w{args.k_shot}s{args.n_query}q seed={args.seed}")
    sampler = CUBEpisodeSampler(
        dataset, n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query, seed=args.seed,
    )

    accs = []
    t0 = time.time()
    log_every = max(1, args.episodes // 10)
    for ep in range(args.episodes):
        episode = sampler.sample()
        acc, _ = proto_episode_eval(encoder, episode, device)
        accs.append(acc)
        if (ep + 1) % log_every == 0:
            elapsed = time.time() - t0
            mean_so_far = float(np.mean(accs))
            print(f"  [eval #54] {ep+1:5d}/{args.episodes}  running ACC={mean_so_far*100:.2f}%  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    arr = np.array(accs)
    mean = float(arr.mean())
    std = float(arr.std())
    lo, hi = bootstrap_ci(arr)
    chance = 1.0 / args.n_way
    z = (mean - chance) / (std / np.sqrt(len(arr))) if std > 0 else float("inf")

    print(f"\n[eval #54] CUB {args.n_way}w{args.k_shot}s {args.episodes} episodes seed={args.seed} encoder={args.encoder}: "
          f"ACC={mean*100:.2f}% IC95%[{lo*100:.2f}, {hi*100:.2f}] z={z:.1f}")
    print(f"[eval #54] chance={chance*100:.1f}%, std_per_ep={std*100:.2f}%, total_time={elapsed:.1f}s")


if __name__ == "__main__":
    main()
