"""
Generate figures for paper Marco 2-C (temporal / SHD).

Números hardcoded de experiment_05_temporal/WEEKLY-1.md e results_*.txt (sessões #72-#77b).
Não rerodam experimentos.

Uso:
    cd paper_marco2c
    python generate_figures.py

Saídas (300 DPI, PNG + PDF):
    figs/fig1_shd_conditions.{png,pdf}     -- barras das 3 condições SHD
    figs/fig2_resolution.{png,pdf}         -- acc vs resolução temporal (timing genuíno)
    figs/fig3_kwta_temporal.{png,pdf}      -- acc vs sparsity k-WTA temporal
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)

C_BLIND = "#7f7f7f"   # cinza: timing-blind
C_FF = "#ff7f0e"      # laranja: feedforward
C_REC = "#1f77b4"     # azul: recurrent
C_REF = "#999999"

CEGO_SHD = 49.41      # baseline cego (3-seed runs de bins/kwta)


def _save(stem):
    for ext in ("png", "pdf"):
        plt.savefig(FIGS / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {stem}.{{png,pdf}}")


def fig1_conditions():
    """Barras das 3 condições SHD (5 seeds, IC95%)."""
    labels = ["Timing-blind\n(histogram+MLP)", "SNN\nfeedforward", "SNN\nrecurrent"]
    accs = [51.56, 61.02, 71.27]
    err = [0.77, 0.85, 0.45]  # inter-seed std
    colors = [C_BLIND, C_FF, C_REC]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bars = ax.bar(range(3), accs, yerr=err, color=colors, capsize=4,
                  width=0.6, error_kw=dict(ecolor="#333", lw=1.2))
    bars[0].set_hatch("//")
    ax.axhline(5.0, color=C_REF, ls="--", lw=1.1, alpha=0.8)
    ax.text(2.4, 6.5, "chance (5%)", color="#555", fontsize=8)
    for i, a in enumerate(accs):
        ax.text(i, a + 1.2, f"{a:.2f}%", ha="center", fontsize=9, fontweight="bold")
    # anotação do gap bruto (texto discreto, sem cobrir as barras)
    ax.text(0.5, 76, "raw gap (blind$\\to$rec): +19.7 p.p.\n(~half is timing — see Fig. 2)",
            ha="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.85))
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("SHD 20-way accuracy (%)", fontsize=11)
    ax.set_title("SHD: timing-blind baseline vs. SNN", fontsize=11)
    ax.set_ylim(0, 80); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); _save("fig1_shd_conditions")


def fig2_resolution():
    """Acc vs nº de bins temporais (timing genuíno = bins 1 -> 100)."""
    bins = np.array([1, 4, 8, 16, 32, 64, 100])
    acc = np.array([58.92, 63.06, 63.49, 67.29, 68.45, 68.54, 69.10])
    std = np.array([0.70, 1.06, 1.85, 0.62, 1.49, 0.85, 0.81])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(bins, acc, yerr=std, marker="o", ms=7, lw=2, capsize=4, color=C_REC)
    ax.axhline(CEGO_SHD, color=C_BLIND, ls="--", lw=1.3, alpha=0.8)
    ax.text(8, CEGO_SHD + 0.4, "timing-blind baseline (49.41%)", color="#555", fontsize=8.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(bins); ax.set_xticklabels([str(b) for b in bins])
    # anotação timing genuíno
    ax.annotate("", xy=(1, 58.92), xytext=(1, 69.10),
                arrowprops=dict(arrowstyle="<->", color="#1f77b4", alpha=0.7))
    ax.text(1.5, 64.0, "+10.18 p.p.\ngenuine timing\n(same network,\n1$\\to$100 bins)",
            fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#cfe8ff", alpha=0.9))
    ax.text(1.05, 54.5, "+9.5 p.p.\narchitectural\n(bins=1 vs blind)", fontsize=8,
            color="#555")
    ax.set_xlabel("Temporal resolution (time bins)", fontsize=11)
    ax.set_ylabel("SHD accuracy (%)", fontsize=11)
    ax.set_title("Accuracy rises with temporal resolution", fontsize=11)
    ax.set_ylim(46, 73); ax.grid(True, alpha=0.3)
    plt.tight_layout(); _save("fig2_resolution")


def fig3_kwta():
    """Acc vs sparsity k-WTA temporal (tolerância 75% + colapso)."""
    # (k, sparsity%, acc, std)
    data = [(128, 50.0, 69.60, 0.95), (64, 75.0, 67.59, 0.93), (32, 87.5, 62.71, 2.10),
            (16, 93.8, 55.54, 0.43), (8, 96.9, 48.98, 0.97), (4, 98.4, 42.64, 1.75)]
    dense = 69.10
    spars = np.array([0.0] + [d[1] for d in data])
    acc = np.array([dense] + [d[2] for d in data])
    std = np.array([0.81] + [d[3] for d in data])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(spars, acc, yerr=std, marker="s", ms=7, lw=2, capsize=4, color=C_REC)
    ax.axhline(CEGO_SHD, color=C_BLIND, ls="--", lw=1.3, alpha=0.8)
    ax.text(2, CEGO_SHD - 2.2, "timing-blind baseline (49.41%)", color="#555", fontsize=8.5)
    # marcar o ponto 75%
    ax.annotate("75% sparsity:\n$-$1.50 p.p.\n(≈ spatial k-WTA, C3)",
                xy=(75, 67.59), xytext=(40, 58),
                fontsize=8.5, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#cfe8ff", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7))
    ax.text(96, 45, "collapse\n(>96%)", fontsize=8, color="#a33", ha="center")
    ax.set_xlabel("Temporal k-WTA sparsity (%)", fontsize=11)
    ax.set_ylabel("SHD accuracy (%)", fontsize=11)
    ax.set_title("Temporal k-WTA: tolerant to 75%, collapses beyond 96%", fontsize=11)
    ax.set_ylim(40, 73); ax.set_xlim(-4, 100); ax.grid(True, alpha=0.3)
    plt.tight_layout(); _save("fig3_kwta_temporal")


def main():
    print(f"Generating figures into {FIGS}")
    fig1_conditions(); fig2_resolution(); fig3_kwta()
    print("Done.")


if __name__ == "__main__":
    main()
