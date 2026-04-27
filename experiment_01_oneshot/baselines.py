"""
Baselines pra comparação: Pixel kNN e Prototypical Networks.

Pixel kNN é o baseline-zero — funciona bem em Omniglot por ser simples
o suficiente. Se sua rede STDP não bate isso, algo está errado.

Prototypical Networks é o baseline forte (deep learning, supervisionado).
Implementação minimalista: encoder CNN-4 + classificação por distância
ao centróide.

Uso:
    python baselines.py --baseline pixel_knn --ways 5 --shots 1
    python baselines.py --baseline proto_net --ways 5 --shots 1 --train-episodes 5000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import default_config
from data import load_evaluation, load_background, EpisodeSampler


# ---------------------------------------------------------------------------
# Baseline 1: Pixel kNN
# ---------------------------------------------------------------------------
def pixel_knn(args: argparse.Namespace) -> None:
    cfg = default_config()
    dataset = load_evaluation(cfg)
    sampler = EpisodeSampler(dataset, args.ways, args.shots, args.queries, args.seed)
    accs = []
    t0 = time.time()
    for ep in range(args.episodes):
        e = sampler.sample()
        sup_flat = e.support.flatten(1)
        qry_flat = e.query.flatten(1)
        # Distância euclidiana
        dists = torch.cdist(qry_flat, sup_flat)  # (Q, N*K)
        # Média de distância por classe (faz sentido pra K>1)
        dists = dists.view(qry_flat.shape[0], args.ways, args.shots).mean(dim=-1)
        preds = dists.argmin(dim=-1)
        acc = (preds == e.query_labels).float().mean().item()
        accs.append(acc)
    print(f"Pixel kNN ({args.ways}w{args.shots}s, {args.episodes}eps, {time.time()-t0:.1f}s):  "
          f"{np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")


# ---------------------------------------------------------------------------
# Baseline 2: Prototypical Networks (Snell 2017)
# ---------------------------------------------------------------------------
class ProtoEncoder(nn.Module):
    """CNN-4 padrão de Snell et al."""

    def __init__(self):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
        self.net = nn.Sequential(
            block(1, 64), block(64, 64), block(64, 64), block(64, 64),
        )

    def forward(self, x):  # x: (B, 1, H, W)
        z = self.net(x)
        return z.flatten(1)


def proto_episode_loss(encoder: nn.Module, episode, n_classes: int, device: torch.device) -> tuple[torch.Tensor, float]:
    sup = episode.support.unsqueeze(1).to(device)
    qry = episode.query.unsqueeze(1).to(device)
    sup_lbl = episode.support_labels.to(device)
    qry_lbl = episode.query_labels.to(device)

    sup_emb = encoder(sup)
    qry_emb = encoder(qry)

    # Centróide por classe
    protos = []
    for c in range(n_classes):
        mask = sup_lbl == c
        protos.append(sup_emb[mask].mean(dim=0))
    protos = torch.stack(protos)

    # Logits = -dist²
    logits = -torch.cdist(qry_emb, protos).pow(2)
    loss = F.cross_entropy(logits, qry_lbl)
    acc = (logits.argmax(-1) == qry_lbl).float().mean().item()
    return loss, acc


def proto_net(args: argparse.Namespace) -> None:
    cfg = default_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    train_ds = load_background(cfg)
    eval_ds = load_evaluation(cfg)
    train_sampler = EpisodeSampler(train_ds, args.ways, args.shots, args.queries, args.seed)
    eval_sampler = EpisodeSampler(eval_ds, args.ways, args.shots, args.queries, args.seed + 1)

    encoder = ProtoEncoder().to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    print(f"Treinando ProtoNet por {args.train_episodes} episódios...")
    encoder.train()
    for step in range(args.train_episodes):
        ep = train_sampler.sample()
        loss, acc = proto_episode_loss(encoder, ep, args.ways, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            print(f"  step {step+1:5d}  loss={loss.item():.3f} acc={acc*100:.1f}%")

    print(f"\nAvaliando em {args.episodes} episódios...")
    encoder.eval()
    accs = []
    with torch.no_grad():
        for _ in range(args.episodes):
            ep = eval_sampler.sample()
            _, acc = proto_episode_loss(encoder, ep, args.ways, device)
            accs.append(acc)
    print(f"ProtoNet ({args.ways}w{args.shots}s, {args.episodes}eps):  "
          f"{np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")


# ---------------------------------------------------------------------------
# Entrada principal
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", choices=["pixel_knn", "proto_net"], required=True)
    p.add_argument("--ways", type=int, default=5)
    p.add_argument("--shots", type=int, default=1)
    p.add_argument("--queries", type=int, default=5)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--train-episodes", type=int, default=5000, help="só pro proto_net")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.baseline == "pixel_knn":
        pixel_knn(args)
    elif args.baseline == "proto_net":
        proto_net(args)


if __name__ == "__main__":
    main()
