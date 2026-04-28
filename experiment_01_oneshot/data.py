"""
Carregamento do Omniglot e codificação imagem → trem de spikes.

Omniglot tem dois splits oficiais:
  - background: 30 alfabetos, ~1200 caracteres → pretreino STDP
  - evaluation: 20 alfabetos, ~420 caracteres → episódios few-shot

Cada caractere tem 20 amostras (escritas por 20 pessoas diferentes).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import Omniglot

from config import Config


# ---------------------------------------------------------------------------
# Codificação spike
# ---------------------------------------------------------------------------
def poisson_encode(image: torch.Tensor, T: int, max_rate: float, dt_ms: float) -> torch.Tensor:
    """
    Rate coding por Poisson. Pixel ∈ [0,1] vira probabilidade de spike por dt.

    image: (..., H, W) com valores em [0,1]
    Retorna: (T, ..., H, W) com 0/1 spikes.

    Para max_rate=100 Hz e dt=1ms, pixel=1.0 dispara em ~10% dos timesteps.
    """
    p_spike = image * max_rate * (dt_ms / 1000.0)
    p_spike = p_spike.clamp(0, 1)
    # broadcasting: cada timestep amostra independente
    spikes = (torch.rand((T,) + image.shape, device=image.device) < p_spike).float()
    return spikes


def temporal_encode(image: torch.Tensor, T: int) -> torch.Tensor:
    """
    Time-to-first-spike: pixels mais brilhantes disparam mais cedo.

    Mais parcimonioso que Poisson (cada neurônio dispara no máximo 1 vez)
    mas mais sensível a hiperparâmetros. Implementado pra fase 2.
    """
    spike_time = ((1 - image) * (T - 1)).long().clamp(0, T - 1)
    spikes = torch.zeros((T,) + image.shape, device=image.device)
    spikes.scatter_(0, spike_time.unsqueeze(0), 1.0)
    # zera pixels muito escuros (não devem disparar)
    spikes = spikes * (image > 0.05).float()
    return spikes


def encode(image: torch.Tensor, cfg: Config) -> torch.Tensor:
    if cfg.spike.encoding == "poisson":
        return poisson_encode(image, cfg.spike.timesteps, cfg.spike.max_rate_hz, cfg.spike.dt_ms)
    if cfg.spike.encoding == "temporal":
        return temporal_encode(image, cfg.spike.timesteps)
    raise ValueError(f"encoding desconhecido: {cfg.spike.encoding}")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def _invert_intensity(x: torch.Tensor) -> torch.Tensor:
    """Omniglot é fundo branco; invertemos pra fundo preto.
    Função module-level (não lambda) pra ser picklável em Windows multiprocessing (spawn)."""
    return 1.0 - x


def build_transforms(cfg: Config) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
        transforms.ToTensor(),  # já normaliza pra [0,1]
        transforms.Lambda(_invert_intensity),
    ])


def load_background(cfg: Config) -> Omniglot:
    """Split de pretreino STDP: 30 alfabetos."""
    return Omniglot(
        root=cfg.data.root,
        background=True,
        download=True,
        transform=build_transforms(cfg),
    )


def load_evaluation(cfg: Config) -> Omniglot:
    """Split de avaliação few-shot: 20 alfabetos novos."""
    return Omniglot(
        root=cfg.data.root,
        background=False,
        download=True,
        transform=build_transforms(cfg),
    )


# ---------------------------------------------------------------------------
# Episódio few-shot (N-way K-shot)
# ---------------------------------------------------------------------------
@dataclass
class FewShotEpisode:
    support: torch.Tensor       # (N*K, H, W)
    support_labels: torch.Tensor  # (N*K,) labels em 0..N-1
    query: torch.Tensor          # (N*Q, H, W)
    query_labels: torch.Tensor   # (N*Q,)
    n_way: int
    k_shot: int


class EpisodeSampler:
    """
    Samples N-way K-shot episódios do split de evaluation.

    Cada episódio sorteia N classes, K supports por classe e Q queries por classe.
    """

    def __init__(self, dataset, n_way: int, k_shot: int, n_query: int, seed: int = 42):
        """
        dataset: torchvision.datasets.Omniglot ou objeto compatível
                 com `_flat_character_images` (lista de (path, char_idx))
                 OU implementando __len__ e __getitem__ retornando (img, label).
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.rng = random.Random(seed)

        # Indexação eficiente: torchvision Omniglot expõe _flat_character_images
        # como lista de (path, char_idx). Usamos isso pra evitar carregar imagens
        # só pra coletar labels — economiza minutos no startup com 24k+ amostras.
        self.by_class: dict[int, list[int]] = defaultdict(list)
        flat = getattr(dataset, "_flat_character_images", None)
        if flat is not None:
            for idx, (_, label) in enumerate(flat):
                self.by_class[label].append(idx)
        else:
            # Fallback: dataset genérico. Tenta acessar .targets/.labels primeiro
            # antes de cair no caminho lento (carregar imagens).
            labels = getattr(dataset, "targets", None) or getattr(dataset, "labels", None)
            if labels is not None:
                for idx, label in enumerate(labels):
                    self.by_class[int(label)].append(idx)
            else:
                for idx in range(len(dataset)):
                    _, label = dataset[idx]
                    self.by_class[int(label)].append(idx)
        self.classes = sorted(self.by_class.keys())

        if len(self.classes) < n_way:
            raise ValueError(
                f"Dataset tem {len(self.classes)} classes mas n_way={n_way}."
            )
        for cls, indices in self.by_class.items():
            if len(indices) < k_shot + n_query:
                raise ValueError(
                    f"Classe {cls} tem {len(indices)} amostras < k_shot+n_query="
                    f"{k_shot + n_query}."
                )

    def sample(self) -> FewShotEpisode:
        chosen = self.rng.sample(self.classes, self.n_way)
        support_imgs, support_lbls = [], []
        query_imgs, query_lbls = [], []
        for new_label, cls in enumerate(chosen):
            indices = self.rng.sample(self.by_class[cls], self.k_shot + self.n_query)
            for i, idx in enumerate(indices):
                img, _ = self.dataset[idx]
                if i < self.k_shot:
                    support_imgs.append(img.squeeze(0))
                    support_lbls.append(new_label)
                else:
                    query_imgs.append(img.squeeze(0))
                    query_lbls.append(new_label)
        return FewShotEpisode(
            support=torch.stack(support_imgs),
            support_labels=torch.tensor(support_lbls),
            query=torch.stack(query_imgs),
            query_labels=torch.tensor(query_lbls),
            n_way=self.n_way,
            k_shot=self.k_shot,
        )
