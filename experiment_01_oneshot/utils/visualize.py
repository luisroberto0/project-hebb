"""
Visualizações diagnósticas pós-pretreino.

Funções principais:
  visualize_filters(model, layer_idx, save_path)
      Mostra filtros conv da camada como grade de imagens. Para layer 1
      (1×kH×kW), cada filtro vira um patch monocromático esperado tipo
      Gabor/strokes se STDP aprendeu features locais.

  plot_weight_histogram(model, save_path)
      Distribuição final dos pesos por camada. Bimodal (perto de w_min e
      w_max) é assinatura de STDP convergente — pesos saturam pra "selecionado"
      ou "podado".

  tsne_embeddings(model, dataset, n_classes, save_path)
      Roda extract_features em N classes do dataset e projeta embeddings
      em 2D via t-SNE. Cluster bem separados = features discriminativas.

Uso típico (em notebook ou script):
    from utils.visualize import visualize_filters, plot_weight_histogram
    model.load_state_dict(torch.load("checkpoints/stdp_model.pt"))
    visualize_filters(model, layer_idx=0, save_path="figs/filters_layer1.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")  # backend sem display, salva em arquivo
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _ensure_mpl():
    if not HAS_MPL:
        raise RuntimeError("matplotlib não instalado. Rode: pip install matplotlib")


def visualize_filters(model, layer_idx: int = 0, save_path: Optional[str] = None,
                      cols: int = 8, figsize_per: float = 1.0) -> None:
    """
    Plota filtros conv como grade. layer_idx=0 → layer1, 1 → layer2.
    Pra layer2 (in_channels > 1), mostra média sobre canais de entrada.
    """
    _ensure_mpl()
    layer = model.layer1 if layer_idx == 0 else model.layer2
    weights = layer.conv.weight.data.cpu().numpy()  # (C_out, C_in, kH, kW)
    C_out, C_in, kH, kW = weights.shape
    rows = (C_out + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per, rows * figsize_per))
    axes = np.atleast_2d(axes)
    for i in range(C_out):
        ax = axes[i // cols, i % cols]
        if C_in == 1:
            img = weights[i, 0]
        else:
            img = weights[i].mean(axis=0)  # média sobre canais
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
    # Esconde eixos extras
    for i in range(C_out, rows * cols):
        axes[i // cols, i % cols].axis("off")
    fig.suptitle(f"Filtros — layer {layer_idx + 1}  ({C_out} filtros, {kH}×{kW})")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Filtros salvos em {save_path}")
    plt.close(fig)


def plot_weight_histogram(model, save_path: Optional[str] = None, bins: int = 50) -> None:
    """Distribuição de pesos por camada. STDP convergente tipicamente bimodal."""
    _ensure_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, layer, name in zip(axes, [model.layer1, model.layer2], ["layer 1", "layer 2"]):
        w = layer.conv.weight.data.cpu().numpy().ravel()
        ax.hist(w, bins=bins, color="#4477aa", edgecolor="white")
        ax.set_title(f"Pesos {name}\nμ={w.mean():.3f}  σ={w.std():.3f}")
        ax.set_xlabel("w")
        ax.set_ylabel("contagem")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Histograma salvo em {save_path}")
    plt.close(fig)


def tsne_embeddings(model, dataset, n_classes: int = 10, samples_per_class: int = 20,
                    device: torch.device | None = None, save_path: Optional[str] = None,
                    perplexity: float = 15.0) -> None:
    """
    Coleta features de N classes e projeta em 2D via t-SNE.
    Usa só dataset rotulado (Omniglot evaluation tem labels por character).
    """
    _ensure_mpl()
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise RuntimeError("scikit-learn não instalado. Rode: pip install scikit-learn")

    device = device or next(model.parameters()).device
    model.eval()

    # Coleta features
    by_class = {}
    flat = getattr(dataset, "_flat_character_images", None)
    if flat:
        for idx, (_, label) in enumerate(flat):
            by_class.setdefault(label, []).append(idx)
    classes = sorted(by_class.keys())[:n_classes]

    feats, labels = [], []
    with torch.no_grad():
        for c in classes:
            indices = by_class[c][:samples_per_class]
            imgs = torch.stack([dataset[i][0].squeeze(0) for i in indices]).to(device)
            emb = model.extract_features(imgs, train_stdp=False).cpu().numpy()
            feats.append(emb)
            labels.extend([c] * len(indices))

    feats = np.concatenate(feats, axis=0)
    labels = np.array(labels)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    proj = tsne.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.get_cmap("tab20", n_classes)
    for i, c in enumerate(classes):
        mask = (labels == c)
        ax.scatter(proj[mask, 0], proj[mask, 1], s=14, color=cmap(i), label=f"cls {c}")
    ax.set_title(f"t-SNE de embeddings ({n_classes} classes × {samples_per_class})")
    ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2")
    ax.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.7)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"t-SNE salvo em {save_path}")
    plt.close(fig)


def main():
    """CLI rápida: carrega checkpoint, gera figuras."""
    import argparse
    from config import default_config
    from model import STDPHopfieldModel

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/stdp_model.pt")
    p.add_argument("--out-dir", type=str, default="figs")
    args = p.parse_args()

    cfg = default_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STDPHopfieldModel(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"Checkpoint carregado: {args.checkpoint}")

    visualize_filters(model, layer_idx=0, save_path=f"{args.out_dir}/filters_layer1.png")
    visualize_filters(model, layer_idx=1, save_path=f"{args.out_dir}/filters_layer2.png")
    plot_weight_histogram(model, save_path=f"{args.out_dir}/weight_hist.png")


if __name__ == "__main__":
    main()
