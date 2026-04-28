"""
Sessão #11 — Visualização e análise quantitativa dos filtros.

Compara checkpoints Iter 1 (treinado, saturado em 0.999) e random U(0,1)
sem treino (32.89% acurácia). Pergunta: o que diferencia +3 p.p. residuais
do STDP? Os filtros saturados são ruído estrutural, têm padrão sistemático,
ou capturam algo Gabor-like?

Saídas em figs/sessao_11/:
  layer1_filters_iter1.png    — 8 filtros 5x5 do Iter 1 (raw)
  layer1_filters_random.png   — 8 filtros 5x5 do random U(0,1)
  layer1_filters_iter1_centered.png   — Iter 1 menos média per-filter (z-score)
  layer1_filters_delta.png    — Iter 1 menos média(random) — padrão sistemático
  layer2_filters_iter1.png    — 16 × 8 grid (16 filtros, 8 channels)
  layer2_filters_random.png   — idem random
  cosine_matrix_layer1.png    — 8×8 cossenos par-a-par (Iter 1 e random lado a lado)
  cosine_matrix_layer2.png    — 16×16 idem
  spatial_std_per_filter.png  — barplot de σ espacial por filtro

Imprime tabela quantitativa em stdout pra documentação.

Uso:
    python tests/visualize_filters_session_11.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # backend sem display

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ITER1_CKPT = "checkpoints/stdp_model_iter1_seed42.pt"
RANDOM_CKPT = "checkpoints/stdp_model_random_u01.pt"
OUT_DIR = Path("figs/sessao_11")


def load_weights(ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu")
    return sd["layer1.conv.weight"].numpy(), sd["layer2.conv.weight"].numpy()


def cosine_matrix(filters: np.ndarray) -> np.ndarray:
    """Cosine matrix entre filtros, achatados."""
    F = filters.reshape(filters.shape[0], -1)
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    F_n = F / (norms + 1e-12)
    return F_n @ F_n.T


def plot_layer1_grid(filters: np.ndarray, title: str, fname: str, vmin=None, vmax=None, cmap="gray"):
    n = filters.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(n * 1.2, 1.5))
    for i in range(n):
        ax = axes[i]
        ax.imshow(filters[i, 0], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"f{i}", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_layer2_grid(filters: np.ndarray, title: str, fname: str, cmap="gray"):
    """16 filtros × 8 input channels = grid 16×8 de patches 5x5."""
    n_out, n_in = filters.shape[0], filters.shape[1]
    fig, axes = plt.subplots(n_out, n_in, figsize=(n_in * 0.7, n_out * 0.7))
    vmin, vmax = filters.min(), filters.max()
    for o in range(n_out):
        for i in range(n_in):
            ax = axes[o, i]
            ax.imshow(filters[o, i], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if o == 0:
                ax.set_title(f"in{i}", fontsize=6)
            if i == 0:
                ax.set_ylabel(f"out{o}", fontsize=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_cosine_pair(c_iter1, c_random, title: str, fname: str):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, mat, sub in zip(axes, [c_iter1, c_random], ["Iter 1 (saturado)", "Random U(0,1)"]):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(sub, fontsize=9)
        ax.set_xticks(range(mat.shape[0]))
        ax.set_yticks(range(mat.shape[0]))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_std(stds_iter1_l1, stds_rand_l1, stds_iter1_l2, stds_rand_l2):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    n1 = len(stds_iter1_l1); n2 = len(stds_iter1_l2)
    x1 = np.arange(n1); x2 = np.arange(n2)
    w = 0.4
    axes[0].bar(x1 - w/2, stds_iter1_l1, w, label="Iter 1", color="C0")
    axes[0].bar(x1 + w/2, stds_rand_l1, w, label="Random", color="C1")
    axes[0].set_title("Layer 1 — σ espacial por filtro (25 pesos)")
    axes[0].set_xlabel("Filtro"); axes[0].set_ylabel("σ"); axes[0].legend()
    axes[0].set_yscale("log")
    axes[1].bar(x2 - w/2, stds_iter1_l2, w, label="Iter 1", color="C0")
    axes[1].bar(x2 + w/2, stds_rand_l2, w, label="Random", color="C1")
    axes[1].set_title("Layer 2 — σ espacial por filtro (200 pesos)")
    axes[1].set_xlabel("Filtro"); axes[1].set_ylabel("σ"); axes[1].legend()
    axes[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spatial_std_per_filter.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def report_table(name, w):
    """Imprime estatísticas globais e por-filtro."""
    print(f"\n=== {name} ===")
    print(f"  Shape: {w.shape}")
    print(f"  Global: μ={w.mean():.4f}  σ={w.std():.4f}  min={w.min():.4f}  max={w.max():.4f}")
    n_out = w.shape[0]
    flat = w.reshape(n_out, -1)
    means = flat.mean(axis=1)
    stds = flat.std(axis=1)
    print(f"  Per-filter μ: range [{means.min():.4f}, {means.max():.4f}]")
    print(f"  Per-filter σ: range [{stds.min():.4f}, {stds.max():.4f}]  mean={stds.mean():.5f}")
    return stds


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saída: {OUT_DIR}")

    w1_iter1, w2_iter1 = load_weights(ITER1_CKPT)
    w1_rand, w2_rand = load_weights(RANDOM_CKPT)

    # === Layer 1 ===
    stds_iter1_l1 = report_table("Iter 1 — Layer 1", w1_iter1)
    stds_rand_l1 = report_table("Random U(0,1) — Layer 1", w1_rand)

    plot_layer1_grid(w1_iter1, "Iter 1 — Layer 1 (raw, saturado ~0.999)", "layer1_filters_iter1.png",
                     vmin=0.99, vmax=1.0, cmap="viridis")
    plot_layer1_grid(w1_rand, "Random U(0,1) — Layer 1 (raw)", "layer1_filters_random.png",
                     vmin=0.0, vmax=1.0, cmap="viridis")

    # Centered (per-filter zero-mean) — revela estrutura espacial fina
    w1_centered = w1_iter1 - w1_iter1.reshape(8, -1).mean(axis=1).reshape(8, 1, 1, 1)
    plot_layer1_grid(w1_centered, "Iter 1 — Layer 1 centrado (cada filtro zero-mean) — escala da σ=0.001",
                     "layer1_filters_iter1_centered.png", cmap="RdBu_r")

    w1_rand_centered = w1_rand - w1_rand.reshape(8, -1).mean(axis=1).reshape(8, 1, 1, 1)
    plot_layer1_grid(w1_rand_centered, "Random U(0,1) — Layer 1 centrado",
                     "layer1_filters_random_centered.png", cmap="RdBu_r")

    # Delta: Iter 1 - mean(random) — padrão sistemático
    rand_mean_l1 = w1_rand.mean(axis=0, keepdims=True)  # (1, 1, 5, 5)
    delta_l1 = w1_iter1 - rand_mean_l1
    plot_layer1_grid(delta_l1, "Iter 1 − média(Random) — Layer 1 (offset sistemático)",
                     "layer1_filters_delta.png", cmap="RdBu_r")

    # Cosine matrices Layer 1
    cos_iter1_l1 = cosine_matrix(w1_iter1)
    cos_rand_l1 = cosine_matrix(w1_rand)
    plot_cosine_pair(cos_iter1_l1, cos_rand_l1,
                     "Cosine entre filtros — Layer 1 (8×8)",
                     "cosine_matrix_layer1.png")

    print(f"\n=== Cosine matrix Layer 1 ===")
    print(f"  Iter 1: off-diagonal range [{cos_iter1_l1[~np.eye(8, dtype=bool)].min():.4f}, "
          f"{cos_iter1_l1[~np.eye(8, dtype=bool)].max():.4f}], mean={cos_iter1_l1[~np.eye(8, dtype=bool)].mean():.4f}")
    print(f"  Random: off-diagonal range [{cos_rand_l1[~np.eye(8, dtype=bool)].min():.4f}, "
          f"{cos_rand_l1[~np.eye(8, dtype=bool)].max():.4f}], mean={cos_rand_l1[~np.eye(8, dtype=bool)].mean():.4f}")

    # === Layer 2 ===
    stds_iter1_l2 = report_table("Iter 1 — Layer 2", w2_iter1)
    stds_rand_l2 = report_table("Random U(0,1) — Layer 2", w2_rand)

    plot_layer2_grid(w2_iter1, "Iter 1 — Layer 2 (16 filtros × 8 channels)",
                     "layer2_filters_iter1.png", cmap="viridis")
    plot_layer2_grid(w2_rand, "Random U(0,1) — Layer 2",
                     "layer2_filters_random.png", cmap="viridis")

    cos_iter1_l2 = cosine_matrix(w2_iter1)
    cos_rand_l2 = cosine_matrix(w2_rand)
    plot_cosine_pair(cos_iter1_l2, cos_rand_l2,
                     "Cosine entre filtros — Layer 2 (16×16)",
                     "cosine_matrix_layer2.png")

    print(f"\n=== Cosine matrix Layer 2 ===")
    print(f"  Iter 1: off-diagonal range [{cos_iter1_l2[~np.eye(16, dtype=bool)].min():.4f}, "
          f"{cos_iter1_l2[~np.eye(16, dtype=bool)].max():.4f}], mean={cos_iter1_l2[~np.eye(16, dtype=bool)].mean():.4f}")
    print(f"  Random: off-diagonal range [{cos_rand_l2[~np.eye(16, dtype=bool)].min():.4f}, "
          f"{cos_rand_l2[~np.eye(16, dtype=bool)].max():.4f}], mean={cos_rand_l2[~np.eye(16, dtype=bool)].mean():.4f}")

    # Spatial std plot
    plot_spatial_std(stds_iter1_l1, stds_rand_l1, stds_iter1_l2, stds_rand_l2)

    # === Análise do delta sistemático ===
    print(f"\n=== Delta sistemático (Iter 1 − média(Random)) ===")
    print(f"  Layer 1 delta: μ={delta_l1.mean():.4f}  σ={delta_l1.std():.4f}")
    print(f"  Layer 1 delta range: [{delta_l1.min():.4f}, {delta_l1.max():.4f}]")

    rand_mean_l2 = w2_rand.mean(axis=0, keepdims=True)
    delta_l2 = w2_iter1 - rand_mean_l2
    print(f"  Layer 2 delta: μ={delta_l2.mean():.4f}  σ={delta_l2.std():.4f}")
    print(f"  Layer 2 delta range: [{delta_l2.min():.4f}, {delta_l2.max():.4f}]")

    print(f"\nFiguras salvas em {OUT_DIR}")
    files = sorted(OUT_DIR.iterdir())
    print(f"Total {len(files)} arquivos:")
    total_bytes = 0
    for f in files:
        sz = f.stat().st_size
        total_bytes += sz
        print(f"  {f.name}: {sz/1024:.1f} KB")
    print(f"Total: {total_bytes/1024:.1f} KB ({total_bytes/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
