"""
N-way K-shot episode sampler for CUB-200 cross-domain few-shot.

Pattern adaptado de experiment_01_oneshot/data.py:EpisodeSampler e
experiment_02_continual/baseline_naive.py:TaskEpisodeSampler.

Saída consistente com FewShotEpisode do Omniglot pra reuso direto de
proto_episode_loss em c3_protonet_sparse.py.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from cub_data import CUBDataset


@dataclass
class CUBEpisode:
    support: torch.Tensor       # (N*K, 1, 28, 28) — keeps channel dim for direct CNN forward
    support_labels: torch.Tensor  # (N*K,) labels remapped 0..N-1
    query: torch.Tensor          # (N*Q, 1, 28, 28)
    query_labels: torch.Tensor   # (N*Q,)
    n_way: int
    k_shot: int


class CUBEpisodeSampler:
    """Sample N-way K-shot episodes from CUB. Output shape compatible com C3."""

    def __init__(self, dataset: CUBDataset, n_way: int, k_shot: int, n_query: int,
                  seed: int = 42):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.rng = random.Random(seed)
        self.classes = sorted(dataset.by_class.keys())

        if len(self.classes) < n_way:
            raise ValueError(f"Dataset tem {len(self.classes)} classes < n_way={n_way}")
        for cls in self.classes:
            if len(dataset.by_class[cls]) < k_shot + n_query:
                raise ValueError(
                    f"Classe {cls} tem {len(dataset.by_class[cls])} samples < "
                    f"k_shot+n_query={k_shot+n_query}"
                )

    def sample(self) -> CUBEpisode:
        chosen = self.rng.sample(self.classes, self.n_way)
        support_imgs, support_lbls = [], []
        query_imgs, query_lbls = [], []
        for new_lbl, cls in enumerate(chosen):
            indices = self.rng.sample(self.dataset.by_class[cls], self.k_shot + self.n_query)
            for i, idx in enumerate(indices):
                img, _ = self.dataset[idx]
                if i < self.k_shot:
                    support_imgs.append(img)
                    support_lbls.append(new_lbl)
                else:
                    query_imgs.append(img)
                    query_lbls.append(new_lbl)

        return CUBEpisode(
            support=torch.stack(support_imgs),       # (N*K, 1, 28, 28)
            support_labels=torch.tensor(support_lbls),
            query=torch.stack(query_imgs),           # (N*Q, 1, 28, 28)
            query_labels=torch.tensor(query_lbls),
            n_way=self.n_way,
            k_shot=self.k_shot,
        )


def proto_episode_eval(encoder, episode: CUBEpisode, device: torch.device) -> tuple[float, dict]:
    """Run forward pass + prototype-based classification. Returns (acc, diagnostics)."""
    sup = episode.support.to(device)        # (N*K, 1, 28, 28)
    qry = episode.query.to(device)
    sup_lbl = episode.support_labels.to(device)
    qry_lbl = episode.query_labels.to(device)

    with torch.no_grad():
        sup_emb = encoder(sup)
        qry_emb = encoder(qry)

        protos = []
        for c in range(episode.n_way):
            mask = sup_lbl == c
            protos.append(sup_emb[mask].mean(dim=0))
        protos = torch.stack(protos)        # (N_way, D)

        dists = torch.cdist(qry_emb, protos).pow(2)
        logits = -dists
        preds = logits.argmax(-1)
        acc = (preds == qry_lbl).float().mean().item()

    return acc, {
        "sup_emb_shape": tuple(sup_emb.shape),
        "qry_emb_shape": tuple(qry_emb.shape),
        "protos_shape": tuple(protos.shape),
        "dists_shape": tuple(dists.shape),
        "preds_min": int(preds.min().item()),
        "preds_max": int(preds.max().item()),
        "dists_mean": float(dists.mean().item()),
        "dists_std": float(dists.std().item()),
        "sup_emb_norm_mean": float(sup_emb.norm(dim=-1).mean().item()),
        "qry_emb_norm_mean": float(qry_emb.norm(dim=-1).mean().item()),
    }
