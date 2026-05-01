"""
CUB-200-2011 dataloader cross-domain few-shot (Marco 2-A, sessão #53).

Pré-processamento da primeira passada (sessão #52 decisão):
- Resize 28×28 (mesma shape do C3 Omniglot)
- Convert to grayscale (1 canal — input shape compatible com encoder treinado em chars)
- Normalize tensor [0,1]
- Output shape: (1, 28, 28) igual Omniglot

Limitação reconhecida: resize 500×500→28×28 grayscale destrói detalhe visual
fino. Aceito nesta passada pra preservar comparabilidade arquitetural com C3
(input shape (B, 1, 28, 28)). Resolução maior fica pra Marco 2-A.2 (#61-#62
condicional).

Splits: usa train_test_split.txt oficial do CUB (não cria split custom).

Cache: imagens preprocessadas salvas em data/CUB_200_2011/cache_28x28_gray.pt
pra evitar re-processar 11.788 imagens em cada execução. #54+ usa cache direto.

Estrutura esperada do CUB extraído:
    data/CUB_200_2011/
        images/<class_name>/<image_name>.jpg
        images.txt                  # <image_id> <relative_path>
        image_class_labels.txt      # <image_id> <class_id>
        train_test_split.txt        # <image_id> <is_training_image>
        classes.txt                 # <class_id> <class_name>
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Default paths (resolved relative to repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
CUB_ROOT_DEFAULT = REPO_ROOT / "data" / "CUB_200_2011"
CACHE_PATH_DEFAULT = CUB_ROOT_DEFAULT / "cache_28x28_gray.pt"


def _build_transform() -> transforms.Compose:
    """Pipeline: resize 28×28 + grayscale + ToTensor [0,1].

    Output shape per image: (1, 28, 28).
    """
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # [0, 1]
    ])


def _parse_metadata(cub_root: Path) -> dict:
    """Parses CUB metadata files. Returns dict with id→path, id→class, id→is_train."""
    images_txt = cub_root / "images.txt"
    labels_txt = cub_root / "image_class_labels.txt"
    split_txt = cub_root / "train_test_split.txt"

    for f in (images_txt, labels_txt, split_txt):
        if not f.exists():
            raise FileNotFoundError(f"CUB metadata missing: {f}")

    id_to_path: dict[int, str] = {}
    with open(images_txt) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                id_to_path[int(parts[0])] = parts[1]

    id_to_class: dict[int, int] = {}
    with open(labels_txt) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                id_to_class[int(parts[0])] = int(parts[1])

    id_to_train: dict[int, bool] = {}
    with open(split_txt) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                id_to_train[int(parts[0])] = parts[1] == "1"

    return {"id_to_path": id_to_path, "id_to_class": id_to_class, "id_to_train": id_to_train}


def build_cache(cub_root: Path = CUB_ROOT_DEFAULT, cache_path: Path = CACHE_PATH_DEFAULT,
                 verbose: bool = True) -> dict:
    """Pre-processes all CUB images to (1, 28, 28) grayscale tensors and caches.

    Returns dict with:
        images: (N, 1, 28, 28) tensor
        labels: (N,) tensor of class IDs (1-indexed, matching CUB convention)
        is_train: (N,) bool tensor (True = train split, False = test/eval split)
    """
    cub_root = Path(cub_root)
    cache_path = Path(cache_path)

    if cache_path.exists():
        if verbose:
            print(f"  [cub-cache] loading from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    if verbose:
        print(f"  [cub-cache] building cache from {cub_root}/images/ ...")

    meta = _parse_metadata(cub_root)
    transform = _build_transform()
    images_dir = cub_root / "images"

    n = len(meta["id_to_path"])
    imgs = torch.zeros(n, 1, 28, 28, dtype=torch.float32)
    labels = torch.zeros(n, dtype=torch.long)
    is_train = torch.zeros(n, dtype=torch.bool)

    sorted_ids = sorted(meta["id_to_path"].keys())
    for i, img_id in enumerate(sorted_ids):
        rel_path = meta["id_to_path"][img_id]
        img_path = images_dir / rel_path
        if not img_path.exists():
            raise FileNotFoundError(f"Missing CUB image: {img_path}")
        with Image.open(img_path) as pil_img:
            t = transform(pil_img.convert("RGB"))  # ensure 3-channel input to grayscale
        imgs[i] = t
        labels[i] = meta["id_to_class"][img_id]
        is_train[i] = meta["id_to_train"][img_id]

        if verbose and (i + 1) % 2000 == 0:
            print(f"  [cub-cache] processed {i+1}/{n}")

    cache = {"images": imgs, "labels": labels, "is_train": is_train}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    if verbose:
        print(f"  [cub-cache] saved {n} images to {cache_path} "
              f"({imgs.element_size() * imgs.numel() / 1e6:.1f} MB)")
    return cache


@dataclass
class CUBSplit:
    images: torch.Tensor      # (N, 1, 28, 28)
    labels: torch.Tensor      # (N,) class IDs
    by_class: dict[int, list[int]]  # class_id → list of indices into images/labels


class CUBDataset(Dataset):
    """CUB-200-2011 with 28×28 grayscale preprocessing + cached tensors.

    Supports 3 splits:
        "train": is_train=True (used for ProtoNet retreinado em CUB, sessão #56)
        "test":  is_train=False (used for cross-domain eval, sessão #54+)
        "all":   both (used if not splitting train/eval)
    """

    def __init__(self, split: str = "test", cub_root: Path = CUB_ROOT_DEFAULT,
                  cache_path: Path = CACHE_PATH_DEFAULT, verbose: bool = True):
        if split not in ("train", "test", "all"):
            raise ValueError(f"split must be 'train', 'test', or 'all'; got {split!r}")
        self.split = split

        cache = build_cache(cub_root=cub_root, cache_path=cache_path, verbose=verbose)
        all_imgs = cache["images"]
        all_lbls = cache["labels"]
        is_train = cache["is_train"]

        if split == "train":
            mask = is_train
        elif split == "test":
            mask = ~is_train
        else:
            mask = torch.ones_like(is_train)

        self.images = all_imgs[mask]
        self.labels = all_lbls[mask]

        self.by_class: dict[int, list[int]] = defaultdict(list)
        for idx, lbl in enumerate(self.labels.tolist()):
            self.by_class[int(lbl)].append(idx)

        if verbose:
            n_classes = len(self.by_class)
            min_per_class = min(len(v) for v in self.by_class.values())
            max_per_class = max(len(v) for v in self.by_class.values())
            print(f"  [CUBDataset split={split}] {len(self.images)} images, "
                  f"{n_classes} classes, {min_per_class}-{max_per_class} per class")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], int(self.labels[idx])


if __name__ == "__main__":
    # Quick sanity: load test split, print shape
    ds = CUBDataset(split="test")
    img, lbl = ds[0]
    print(f"  sample[0]: shape={tuple(img.shape)}, dtype={img.dtype}, label={lbl}")
    print(f"  pixel range: min={img.min():.3f} max={img.max():.3f}")
