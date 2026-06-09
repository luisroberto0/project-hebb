"""
Marco 2-C extensão (#73) — sweep de resolução temporal: a vantagem da SNN é o TIMING?

Teste mecanístico decisivo da ressalva "SHD é temporal por construção": variamos o nº de
bins temporais. Se a vantagem vem do timing, a acc da SNN recorrente deve CRESCER com a
resolução e, em bins=1 (= histograma total, sem ordem temporal), CONVERGIR para o baseline
cego. Isso prova que a SNN ganha proporcionalmente à resolução temporal disponível.

Reusa o harness. bag_of_spikes (cego) é invariante ao nº de bins (soma total) -> referência fixa.
Salva results_bins.txt.

Uso:
    python sweep_bins.py --epochs 8 --seeds 3
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from temporal_bench import SNN_Rec, BlindMLP, train, evaluate
from shd_data import SHDDataset

BINS = [1, 4, 8, 16, 32, 64, 100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    # carrega spikes esparsos uma vez; binning é lazy (muda ds.n_bins)
    tr_ds = SHDDataset("train", n_bins=BINS[0])
    te_ds = SHDDataset("test", n_bins=BINS[0])

    out = [f"Sweep bins (resolucao temporal) — SHD, {args.seeds} seeds, {args.epochs} epochs, device={dev}", ""]

    # baseline cego: invariante ao nº de bins (histograma total) -> 1 medida
    te_ds.n_bins = 100; tr_ds.n_bins = 100
    cego_accs = []
    for s in range(args.seeds):
        torch.manual_seed(s)
        m = BlindMLP().to(dev)
        train(m, DataLoader(tr_ds, args.batch, shuffle=True), dev, args.epochs)
        cego_accs.append(evaluate(m, DataLoader(te_ds, args.batch), dev))
    cego = float(np.mean(cego_accs))
    out.append(f"baseline cego (sem timing, referencia): {cego:.2f}% ±{np.std(cego_accs):.2f}")
    out.append("")
    out.append("bins | SNN recorrente acc (mean±std) | margem sobre cego")
    print("\n".join(out))

    curve = []
    for nb in BINS:
        tr_ds.n_bins = nb; te_ds.n_bins = nb
        accs = []
        for s in range(args.seeds):
            torch.manual_seed(s)
            m = SNN_Rec(gain=1.0).to(dev)
            train(m, DataLoader(tr_ds, args.batch, shuffle=True), dev, args.epochs)
            accs.append(evaluate(m, DataLoader(te_ds, args.batch), dev))
        mu, sd = float(np.mean(accs)), float(np.std(accs))
        curve.append((nb, mu, sd))
        line = f"{nb:4d} | {mu:5.2f} ±{sd:.2f} | {mu-cego:+.2f} p.p."
        out.append(line); print(line)

    out.append("")
    out.append("Predicao: acc cresce com bins; em bins=1 (histograma) SNN_rec ~ cego (margem ~0).")
    out.append(f"bins=1 margem={curve[0][1]-cego:+.2f}  bins=100 margem={curve[-1][1]-cego:+.2f}")
    with open(os.path.join(os.path.dirname(__file__), "results_bins.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nbins=1 margem={curve[0][1]-cego:+.2f}pp  bins=100 margem={curve[-1][1]-cego:+.2f}pp")
    print("Salvo em results_bins.txt")


if __name__ == "__main__":
    main()
