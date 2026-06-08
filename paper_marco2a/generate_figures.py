"""
Generate figures for paper Marco 2-A.

Reusa dados já no repo (sessões #52-#57, experiment_03_crossdomain/WEEKLY-1.md).
NÃO rerodam experimentos — números hardcoded da WEEKLY-1 consolidada.
Todos os valores conferem 1-pra-1 com Table 1 (experiments.md) e WEEKLY-1.md.

Uso:
    cd paper_marco2a
    python generate_figures.py

Saídas (300 DPI, PNG + PDF):
    figs/fig1_crossdomain_bars.{png,pdf}    -- 8 condições + chance, cluster k-WTA destacado
    figs/fig2_effect_collapse.{png,pdf}     -- dual-panel in-domain vs cross-domain
    figs/fig3_bottleneck_waterfall.{png,pdf}-- random -> C3 -> retrained 28 -> retrained 84
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path

FIGS_DIR = Path(__file__).resolve().parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)

# Cores consistentes com paper C3 (#1f77b4 blue, #d62728 red) + extensão para Marco 2-A
C_RETRAINED = "#2ca02c"   # verde: retrained-on-target (upper bound)
C_PIXEL = "#ff7f0e"       # laranja: pixel kNN (no encoder)
C_C3 = "#1f77b4"          # azul: source-trained C3 sparsities
C_RANDOM = "#7f7f7f"      # cinza: random encoder (no training)
C_CHANCE = "#999999"      # cinza claro: chance line

# ---------------------------------------------------------------------------
# Table 1 (experiments.md / WEEKLY-1.md): cross-domain CUB-200 5w1s, 5 seeds.
# (label, input, mean, inter_seed_std, ci_low, ci_high, category)
# ---------------------------------------------------------------------------
TABLE1 = [
    ("ProtoNet retrained\nCUB 84x84 RGB",  "(3,84,84)",     49.84, 0.82, 49.38, 50.59, "retrained"),
    ("ProtoNet retrained\nCUB 28x28 gray", "(1,28,28)",     34.31, 0.31, 34.06, 34.55, "retrained"),
    ("Pixel kNN\ncross-domain",            "(1,28,28) raw", 22.81, 0.18, 22.69, 22.97, "pixel"),
    ("C3 k=32 (50%)\nOmniglot frozen",     "(1,28,28)",     22.20, 0.53, 21.77, 22.57, "c3"),
    ("C3 k=64 (vanilla)\nOmniglot frozen", "(1,28,28)",     22.13, 0.30, 21.90, 22.36, "c3"),
    ("C3 k=16 (75%)\nOmniglot frozen",     "(1,28,28)",     22.09, 0.32, 21.84, 22.34, "c3"),
    ("Random encoder\n+ k-WTA k=16",       "(1,28,28)",     21.91, 0.17, 21.76, 22.03, "random"),
    ("C3 k=8 (87.5%)\nOmniglot frozen",    "(1,28,28)",     21.68, 0.44, 21.34, 22.04, "c3"),
]
CHANCE = 20.00

# Table 2 (effect-collapse): in-domain Omniglot vs cross-domain CUB at each k.
# (k, sparsity_pct, omniglot_acc, cub_acc)
TABLE2 = [
    (8,  87.5, 90.77, 21.68),
    (16, 75.0, 93.10, 22.09),
    (32, 50.0, 93.35, 22.20),
    (64, 0.0,  94.55, 22.13),
]
SPREAD_INDOMAIN = 94.55 - 90.77   # 3.78 p.p.
SPREAD_CROSSDOMAIN = 22.20 - 21.68  # 0.52 p.p.

CAT_COLOR = {"retrained": C_RETRAINED, "pixel": C_PIXEL, "c3": C_C3, "random": C_RANDOM}


def _save(fig_stem: str):
    for ext in ("png", "pdf"):
        plt.savefig(FIGS_DIR / f"{fig_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig_stem}.{{png,pdf}}")


def fig1_crossdomain_bars():
    """Horizontal bar chart: 8 conditions + chance, sorted descending.

    Highlights that the five CNN-forward conditions (4 C3 sparsities + random)
    collapse into a 0.52 p.p. band while retrained baselines tower above.
    """
    labels = [r[0] for r in TABLE1]
    means = np.array([r[2] for r in TABLE1])
    ci_low = np.array([r[2] - r[4] for r in TABLE1])
    ci_high = np.array([r[5] - r[2] for r in TABLE1])
    colors = [CAT_COLOR[r[6]] for r in TABLE1]

    # Plot top-to-bottom in table order (highest first).
    y = np.arange(len(labels))[::-1]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bars = ax.barh(y, means, xerr=[ci_low, ci_high], color=colors,
                   capsize=3, height=0.62, error_kw=dict(ecolor="#333333", lw=1))
    # Hatch the no-training conditions (random encoder).
    for bar, r in zip(bars, TABLE1):
        if r[6] == "random":
            bar.set_hatch("//")
            bar.set_alpha(0.75)

    # Chance reference.
    ax.axvline(x=CHANCE, color=C_CHANCE, linestyle="--", lw=1.2, alpha=0.8)
    ax.text(CHANCE + 0.15, len(labels) - 0.35, "chance (20%)",
            color="#555555", fontsize=8, rotation=90, va="top")

    # Value labels at bar ends.
    for yi, m in zip(y, means):
        ax.text(m + 0.7, yi, f"{m:.2f}%", va="center", fontsize=8, color="#222222")

    # Shaded band over the collapsed cross-domain cluster (the five (1,28,28)
    # CNN-forward conditions: C3 k=32/64/16/8 + random encoder).
    cluster_vals = [r[2] for r in TABLE1 if r[6] in ("c3", "random")]
    ax.axvspan(min(cluster_vals) - 0.4, max(cluster_vals) + 0.4,
               color=C_C3, alpha=0.07, zorder=0)
    ax.annotate(f"k-WTA spread = {SPREAD_CROSSDOMAIN:.2f} p.p.\n(within CI overlap)",
                xy=(22.0, 1.0), xytext=(30.0, 2.3), fontsize=9, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8),
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7))

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("CUB-200 5-way 1-shot accuracy (%)", fontsize=11)
    ax.set_title("Cross-domain few-shot accuracy (Omniglot$\\rightarrow$CUB-200)",
                 fontsize=11)
    ax.set_xlim(18, 54)
    ax.grid(True, alpha=0.3, axis="x")

    legend_handles = [
        Patch(fc=C_RETRAINED, label="Retrained on CUB (upper bound)"),
        Patch(fc=C_PIXEL, label="Pixel kNN (no encoder)"),
        Patch(fc=C_C3, label="C3 source-trained (k-WTA)"),
        Patch(fc=C_RANDOM, hatch="//", label="Random encoder (no training)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5)

    plt.tight_layout()
    _save("fig1_crossdomain_bars")


def fig2_effect_collapse():
    """Dual-panel: in-domain (Omniglot) vs cross-domain (CUB) across sparsity.

    Left panel shows the 3.78 p.p. in-domain spread; right panel shows it
    collapsing to 0.52 p.p. of noise around ~22%.
    """
    k = np.array([r[0] for r in TABLE2])
    omni = np.array([r[2] for r in TABLE2])
    cub = np.array([r[3] for r in TABLE2])
    xlabels = [f"k={r[0]}\n({r[1]:.1f}%)" if r[1] > 0 else f"k={r[0]}\n(vanilla)"
               for r in TABLE2]
    x = np.arange(len(k))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 4.2))

    # Left: Omniglot in-domain.
    axL.plot(x, omni, marker="o", ms=8, lw=2, color=C_C3)
    for xi, v in zip(x, omni):
        axL.text(xi, v + 0.25, f"{v:.2f}", ha="center", fontsize=8, color="#222222")
    axL.set_title("In-domain (Omniglot)", fontsize=11)
    axL.set_ylabel("5-way 1-shot accuracy (%)", fontsize=11)
    axL.set_ylim(88, 96)
    axL.set_xticks(x)
    axL.set_xticklabels(xlabels, fontsize=8)
    axL.grid(True, alpha=0.3)
    axL.annotate(f"spread = {SPREAD_INDOMAIN:.2f} p.p.",
                 xy=(0.5, 92.6), xytext=(0.5, 89.2), fontsize=9, ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8))

    # Right: CUB cross-domain (same delta-axis span = 8 p.p. to make the
    # collapse visually honest; chance line included).
    axR.plot(x, cub, marker="s", ms=8, lw=2, color=C_RETRAINED)
    for xi, v in zip(x, cub):
        axR.text(xi, v + 0.22, f"{v:.2f}", ha="center", fontsize=8, color="#222222")
    axR.axhline(y=CHANCE, color=C_CHANCE, linestyle="--", lw=1.2, alpha=0.8)
    axR.text(2.55, CHANCE + 0.12, "chance (20%)", color="#555555", fontsize=8)
    axR.set_title("Cross-domain (CUB-200)", fontsize=11)
    axR.set_ylim(19.5, 27.5)   # 8 p.p. span, matching left panel's range
    axR.set_xticks(x)
    axR.set_xticklabels(xlabels, fontsize=8)
    axR.grid(True, alpha=0.3)
    axR.annotate(f"spread = {SPREAD_CROSSDOMAIN:.2f} p.p.",
                 xy=(1.5, 22.2), xytext=(1.5, 25.4), fontsize=9, ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8))

    fig.suptitle("k-WTA effect collapse under extreme domain shift", fontsize=12)
    fig.supxlabel("k (embedding sparsity)", fontsize=10)
    plt.tight_layout()
    _save("fig2_effect_collapse")


def fig3_bottleneck_waterfall():
    """Waterfall: random -> C3 k=16 -> retrained 28x28 -> retrained 84x84 RGB.

    Shows the two large, roughly additive contributors (target training +12.22,
    resolution/color +15.53) versus the negligible training-on-source step.
    """
    stages = [
        ("Random\nencoder", 21.91, None),
        ("+ Omniglot\ntraining\n(C3 k=16)", 22.09, +0.18),
        ("+ Target\ntraining\n(CUB 28x28)", 34.31, +12.22),
        ("+ Resolution\n& color\n(CUB 84x84)", 49.84, +15.53),
    ]
    tops = np.array([s[1] for s in stages])
    x = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(8.0, 4.6))

    floor = CHANCE
    # Base stack (chance -> first value) for the anchor bar.
    ax.bar(0, tops[0] - floor, bottom=floor, width=0.6, color=C_RANDOM,
           hatch="//", alpha=0.75, edgecolor="white")
    # Incremental floating bars.
    inc_colors = [C_C3, C_RETRAINED, C_RETRAINED]
    for i in range(1, len(stages)):
        delta = tops[i] - tops[i - 1]
        bottom = tops[i - 1]
        ax.bar(i, delta, bottom=bottom, width=0.6, color=inc_colors[i - 1],
               alpha=0.9, edgecolor="white")
        # Connector line between stage tops.
        ax.plot([i - 1 + 0.3, i - 0.3], [tops[i - 1], tops[i - 1]],
                color="#888888", lw=1, ls="--")

    # Labels: cumulative value above each bar top; delta inside floating bars.
    for i, s in enumerate(stages):
        ax.text(i, tops[i] + 0.6, f"{s[1]:.2f}%", ha="center", fontsize=9,
                fontweight="bold", color="#222222")
        if s[2] is not None:
            sign = "+" if s[2] >= 0 else ""
            if abs(s[2]) >= 2.0:
                # Bar tall enough to hold the delta label inside.
                mid = (tops[i] + tops[i - 1]) / 2
                ax.text(i, mid, f"{sign}{s[2]:.2f}\np.p.", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
            else:
                # Negligible step: bar is invisibly thin, annotate to the side.
                ax.text(i + 0.36, tops[i - 1], f"{sign}{s[2]:.2f} p.p.\n(negligible)",
                        ha="left", va="center", fontsize=8, color="#555555")

    ax.axhline(y=CHANCE, color=C_CHANCE, linestyle="--", lw=1.2, alpha=0.8)
    ax.text(len(stages) - 0.5, CHANCE - 1.1, "chance (20%)",
            color="#555555", fontsize=8, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in stages], fontsize=9)
    ax.set_ylabel("CUB-200 5-way 1-shot accuracy (%)", fontsize=11)
    ax.set_title("Bottleneck decomposition: what actually moves cross-domain accuracy",
                 fontsize=11)
    ax.set_ylim(18, 54)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _save("fig3_bottleneck_waterfall")


def main():
    print(f"Generating figures into {FIGS_DIR}")
    fig1_crossdomain_bars()
    fig2_effect_collapse()
    fig3_bottleneck_waterfall()
    print("Done.")


if __name__ == "__main__":
    main()
