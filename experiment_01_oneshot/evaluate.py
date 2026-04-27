"""
Avaliação few-shot N-way K-shot em Omniglot.

Uso:
    python evaluate.py --checkpoint checkpoints/stdp_model.pt --ways 5 --shots 1
    python evaluate.py --ways 20 --shots 1 --episodes 1000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from config import default_config
from data import load_evaluation, EpisodeSampler
from model import STDPHopfieldModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path pro checkpoint do pretreino STDP. Se None, usa pesos iniciais (random).")
    p.add_argument("--ways", type=int, default=5)
    p.add_argument("--shots", type=int, default=1)
    p.add_argument("--queries", type=int, default=5)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    """IC95% por bootstrap percentil."""
    rng = np.random.default_rng(0)
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def main() -> None:
    args = parse_args()
    cfg = default_config()
    if args.device is not None:
        cfg.device = args.device

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    print(f"Dispositivo: {device}")

    torch.manual_seed(args.seed)

    print("Carregando Omniglot evaluation...")
    dataset = load_evaluation(cfg)
    sampler = EpisodeSampler(
        dataset,
        n_way=args.ways,
        k_shot=args.shots,
        n_query=args.queries,
        seed=args.seed,
    )
    print(f"  {len(sampler.classes)} classes disponíveis pra few-shot")

    model = STDPHopfieldModel(cfg).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        # strict=False porque a projeção pode não estar no checkpoint (criada lazy)
        model.load_state_dict(state, strict=False)
        print(f"  Checkpoint carregado: {args.checkpoint}")
    else:
        print("  ⚠️  Sem checkpoint — usando pesos iniciais (random). É baseline-zero.")
    model.eval()

    print(f"\nAvaliação: {args.ways}-way {args.shots}-shot, {args.episodes} episódios")
    accs = []
    t0 = time.time()
    with torch.no_grad():
        for ep in range(args.episodes):
            episode = sampler.sample()
            support = episode.support.to(device)
            support_labels = episode.support_labels.to(device)
            query = episode.query.to(device)
            query_labels = episode.query_labels.to(device)

            logits = model(support, support_labels, query, n_classes=args.ways)
            preds = logits.argmax(dim=-1)
            acc = (preds == query_labels).float().mean().item()
            accs.append(acc)

            if (ep + 1) % 50 == 0:
                running = np.mean(accs)
                print(f"  episode {ep+1:4d}/{args.episodes}  acc rolling={running*100:.2f}%")

    accs_arr = np.array(accs)
    mean = accs_arr.mean()
    lo, hi = bootstrap_ci(accs_arr)
    print(f"\nResultado final ({args.episodes} episódios em {time.time()-t0:.1f}s):")
    print(f"  Acurácia média: {mean*100:.2f}%   IC95%: [{lo*100:.2f}%, {hi*100:.2f}%]")

    # Comparação com chance
    chance = 1.0 / args.ways
    z = (mean - chance) / accs_arr.std() if accs_arr.std() > 0 else float("inf")
    print(f"  Chance: {chance*100:.2f}%   Distância de chance: {(mean-chance)*100:.2f} p.p.  z≈{z:.1f}")


if __name__ == "__main__":
    main()
