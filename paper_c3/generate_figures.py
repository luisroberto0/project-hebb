"""
Generate figures for paper C3.

Reusa dados já no repo (sessão #20, c3_protonet_sparse.py outputs).
Não rerodam experimentos — números hardcoded de WEEKLY-2.md sessão #20.

Uso:
    cd paper_c3
    python generate_figures.py

Saídas:
    figs/fig1_sparsity_curve.png  (300 DPI)
    figs/fig1_sparsity_curve.pdf
    figs/fig2_validation.png       (300 DPI)
    figs/fig2_validation.pdf
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGS_DIR = Path(__file__).resolve().parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)

# Data from session #20 (commit fc75495):
# Format: (sparsity_pct, acc_5w1s, ic95_low_5w, ic95_high_5w, acc_20w1s, ic95_low_20w, ic95_high_20w)
DATA = [
    (0.0,  94.55, 94.10, 95.00, np.nan, np.nan, np.nan),  # ProtoNet baseline (5w1s only reported with std±6.40)
    (50.0, 93.35, 92.89, 93.77, 81.87, 81.52, 82.20),     # C3a k=32
    (75.0, 93.10, 92.67, 93.55, 80.72, 80.36, 81.09),     # C3b k=16
    (87.5, 90.77, 90.20, 91.34, 75.44, 75.00, 75.87),     # C3c k=8
]


def fig1_sparsity_curve():
    """Sparsity-Accuracy Trade-off Curve.
    X: sparsity %. Y: accuracy %. Two curves (5w1s, 20w1s) with IC95% error bars.
    """
    sparsity = np.array([d[0] for d in DATA])
    acc_5w = np.array([d[1] for d in DATA])
    err_5w_low = np.array([d[1] - d[2] for d in DATA])
    err_5w_high = np.array([d[3] - d[1] for d in DATA])

    acc_20w = np.array([d[4] for d in DATA])
    err_20w_low = np.array([d[4] - d[5] if not np.isnan(d[5]) else 0 for d in DATA])
    err_20w_high = np.array([d[6] - d[4] if not np.isnan(d[6]) else 0 for d in DATA])

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # 5-way 1-shot
    ax.errorbar(sparsity, acc_5w, yerr=[err_5w_low, err_5w_high],
                marker='o', markersize=8, linewidth=2, capsize=4,
                label='5-way 1-shot', color='#1f77b4')

    # 20-way 1-shot (skip baseline point — not reported in 20w1s)
    valid_20w = ~np.isnan(acc_20w)
    ax.errorbar(sparsity[valid_20w], acc_20w[valid_20w],
                yerr=[err_20w_low[valid_20w], err_20w_high[valid_20w]],
                marker='s', markersize=8, linewidth=2, capsize=4,
                label='20-way 1-shot', color='#d62728')

    # Reference lines
    ax.axhline(y=94.55, color='#1f77b4', linestyle=':', alpha=0.4, linewidth=1)
    ax.text(2, 94.9, 'ProtoNet baseline (5w1s)', color='#1f77b4', fontsize=8, alpha=0.7)

    # Annotations for key points
    ax.annotate('C3b: 75% sparse\n−1.45 p.p. vs baseline',
                xy=(75, 93.10), xytext=(45, 88),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6))

    ax.set_xlabel('Sparsity (%)', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Sparsity-Accuracy Trade-off in ProtoNet + k-WTA on Omniglot',
                 fontsize=11)
    ax.set_xlim(-5, 95)
    ax.set_ylim(70, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=10)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(FIGS_DIR / f"fig1_sparsity_curve.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved fig1_sparsity_curve.{{png,pdf}}")


def fig2_validation():
    """Validation comparison: ProtoNet baseline vs C3a/b/c vs Random+kWTA.
    Bar chart highlighting +55.50 p.p. gap between trained and random+kWTA.
    """
    labels = ['ProtoNet\nbaseline\n(0% sparse)',
              'C3a\nk=32\n(50% sparse)',
              'C3b\nk=16\n(75% sparse)',
              'C3c\nk=8\n(87.5% sparse)',
              'Random encoder\n+ k-WTA\n(75% sparse)']
    accs_5w = [94.55, 93.35, 93.10, 90.77, 37.60]
    accs_20w = [np.nan, 81.87, 80.72, 75.44, 16.73]
    err_5w_low = [0.45, 0.46, 0.43, 0.57, 0.71]
    err_5w_high = [0.45, 0.42, 0.45, 0.57, 0.74]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    bars5 = ax.bar(x - width/2, accs_5w, width,
                   yerr=err_5w_low,  # symmetric is fine for visualization
                   label='5-way 1-shot', color='#1f77b4', capsize=3)
    bars20 = ax.bar(x + width/2, accs_20w, width,
                    label='20-way 1-shot', color='#d62728', capsize=3)

    # Distinguish trained from random with hatching
    bars5[-1].set_hatch('//')
    bars5[-1].set_alpha(0.6)
    bars20[-1].set_hatch('//')
    bars20[-1].set_alpha(0.6)

    # Chance lines
    ax.axhline(y=20, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(4.5, 22, 'chance (5w)', color='gray', fontsize=8)
    ax.axhline(y=5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(4.5, 7, 'chance (20w)', color='gray', fontsize=8)

    # Gap annotation
    ax.annotate('', xy=(2, 93.10), xytext=(4, 37.60),
                arrowprops=dict(arrowstyle='<->', color='black', alpha=0.5))
    ax.text(3.0, 67, '+55.50 p.p.\n(training gap)', ha='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('C3 trained vs random encoder validation', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(FIGS_DIR / f"fig2_validation.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved fig2_validation.{{png,pdf}}")


def main():
    print(f"Generating figures into {FIGS_DIR}")
    fig1_sparsity_curve()
    fig2_validation()
    print("Done.")


if __name__ == "__main__":
    main()
