"""
Sessão #56 — Treina ProtoNet RETREINADO em CUB-200 (baseline a bater pelo
critério literal Marco 2-A).

Duas resoluções:
- 28x28 grayscale (comparável com C3 cross-domain): usa ProtoEncoder original
  (input shape (B, 1, 28, 28))
- 84x84 RGB (literatura standard cross-domain few-shot): usa ProtoEncoderRGB
  (CNN-4 adaptado, primeira layer Conv2d(3, 64) + AdaptiveAvgPool2d no final
  pra colapsar (B, 64, 5, 5) -> (B, 64), preservando embed_dim=64
  consistente com C3)

Treino: episode-based 5w1s5q em CUB train split (~5994 imgs, 200 classes).
Mesmos hyperparams da sessão #20 (Adam lr=1e-3, 5000 episodes).

Não modifica c3_protonet_sparse.py — importa ProtoEncoder dele e define
ProtoEncoderRGB localmente.

Uso:
    python train_cub_protonet.py --device cuda --resolution 28 --seeds 42 43 44 45 46
    python train_cub_protonet.py --device cuda --resolution 84 --seeds 42 43 44 45 46
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment_01_oneshot"))

from cub_data import CUBDataset
from c3_protonet_sparse import ProtoEncoder  # 1-channel, 28x28


class ProtoEncoderRGB(nn.Module):
    """CNN-4 adaptado para 3 canais RGB + adaptive pool pra preservar 64D embed.

    Diferencas vs ProtoEncoder (1ch, 28x28):
    - Primeira layer Conv2d(3, 64) em vez de Conv2d(1, 64).
    - AdaptiveAvgPool2d((1,1)) no final pra colapsar feature map espacial
      em qualquer resolucao -> (B, 64, 1, 1) -> flatten -> (B, 64).
    - Em 84x84: 4 maxpools 2x2 dao 84->42->21->10->5 = (B, 64, 5, 5),
      adaptive pool reduz pra (B, 64, 1, 1).
    """

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
            block(3, 64), block(64, 64), block(64, 64), block(64, 64),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):  # x: (B, 3, H, W)
        z = self.net(x)
        z = self.pool(z)
        return z.flatten(1)  # (B, 64)


@dataclass
class CUBEpisode:
    support: torch.Tensor       # (N*K, C, H, W)
    support_labels: torch.Tensor  # (N*K,)
    query: torch.Tensor          # (N*Q, C, H, W)
    query_labels: torch.Tensor   # (N*Q,)
    n_way: int
    k_shot: int


class TrainEpisodeSampler:
    """Amostra episodios 5w1s5q do CUB train split (~5994 imgs, 200 classes)."""

    def __init__(self, dataset: CUBDataset, n_way: int, k_shot: int, n_query: int, seed: int):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.rng = random.Random(seed)
        self.classes = sorted(dataset.by_class.keys())

        # Filtra classes com instancias suficientes pra k_shot+n_query
        valid_classes = [c for c in self.classes
                          if len(dataset.by_class[c]) >= k_shot + n_query]
        if len(valid_classes) < n_way:
            raise ValueError(
                f"CUB train tem {len(valid_classes)} classes com >={k_shot+n_query} samples, "
                f"n_way={n_way}"
            )
        self.classes = valid_classes

    def sample(self) -> CUBEpisode:
        chosen = self.rng.sample(self.classes, self.n_way)
        sup_imgs, sup_lbls = [], []
        qry_imgs, qry_lbls = [], []
        for new_lbl, cls in enumerate(chosen):
            indices = self.rng.sample(self.dataset.by_class[cls], self.k_shot + self.n_query)
            for i, idx in enumerate(indices):
                img, _ = self.dataset[idx]
                if i < self.k_shot:
                    sup_imgs.append(img)
                    sup_lbls.append(new_lbl)
                else:
                    qry_imgs.append(img)
                    qry_lbls.append(new_lbl)
        return CUBEpisode(
            support=torch.stack(sup_imgs),
            support_labels=torch.tensor(sup_lbls),
            query=torch.stack(qry_imgs),
            query_labels=torch.tensor(qry_lbls),
            n_way=self.n_way,
            k_shot=self.k_shot,
        )


def proto_episode_loss(encoder, episode: CUBEpisode, device: torch.device):
    sup = episode.support.to(device)        # (N*K, C, H, W)
    qry = episode.query.to(device)
    sup_lbl = episode.support_labels.to(device)
    qry_lbl = episode.query_labels.to(device)

    sup_emb = encoder(sup)
    qry_emb = encoder(qry)

    n_way = episode.n_way
    protos = []
    for c in range(n_way):
        mask = sup_lbl == c
        protos.append(sup_emb[mask].mean(dim=0))
    protos = torch.stack(protos)

    logits = -torch.cdist(qry_emb, protos).pow(2)
    loss = F.cross_entropy(logits, qry_lbl)
    acc = (logits.argmax(-1) == qry_lbl).float().mean().item()
    return loss, acc


def train_one_seed(args, seed: int, device, dataset_train: CUBDataset, save_path: Path):
    print(f"\n{'='*72}\n[seed={seed}] Train ProtoNet on CUB res={args.resolution}\n{'='*72}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.resolution == 28:
        encoder = ProtoEncoder().to(device)
    else:
        encoder = ProtoEncoderRGB().to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  [seed={seed}] encoder={type(encoder).__name__} params={n_params}")

    sampler = TrainEpisodeSampler(
        dataset_train, n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query, seed=seed,
    )
    opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    encoder.train()

    t0 = time.time()
    log_every = max(1, args.train_episodes // 10)
    for step in range(args.train_episodes):
        ep = sampler.sample()
        loss, acc = proto_episode_loss(encoder, ep, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"  [seed={seed}] step {step+1:5d}/{args.train_episodes}  "
                  f"loss={loss.item():.3f}  acc={acc*100:.1f}%  ({elapsed:.1f}s)")

    train_time = time.time() - t0
    print(f"  [seed={seed}] training done in {train_time:.1f}s")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": encoder.state_dict(),
        "encoder_class": type(encoder).__name__,
        "resolution": args.resolution,
        "seed": seed,
        "train_episodes": args.train_episodes,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "n_query": args.n_query,
        "session": "#56 ProtoNet retreinado em CUB",
        "train_time_seconds": train_time,
    }, save_path)
    print(f"  [seed={seed}] saved to {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--resolution", type=int, choices=(28, 84), default=28)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--train-episodes", type=int, default=5000)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=1)
    p.add_argument("--n-query", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ckpt-dir", default="experiment_01_oneshot/checkpoints")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}, resolution={args.resolution}, seeds={args.seeds}")

    repo_root = Path(__file__).resolve().parent.parent
    ckpt_dir = repo_root / args.ckpt_dir

    print(f"[train-cub] Loading CUB train split (res={args.resolution})...")
    dataset_train = CUBDataset(split="train", resolution=args.resolution, verbose=True)

    for seed in args.seeds:
        save_path = ckpt_dir / f"protonet_cub_{args.resolution}_seed{seed}.pt"
        if args.skip_existing and save_path.exists():
            print(f"\n[seed={seed}] checkpoint exists, skipping (--skip-existing)")
            continue
        train_one_seed(args, seed, device, dataset_train, save_path)


if __name__ == "__main__":
    main()
