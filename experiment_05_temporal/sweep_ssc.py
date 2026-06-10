"""
Marco 2-C extensão (#76) — generalização para SSC (Spiking Speech Commands).

SSC: dataset irmão do SHD (Cramer/Zenke 2020), 35 classes (~75k train), mesmo formato
(700 canais, ~1s). Testa se o achado de timing do SHD GENERALIZA para um benchmark maior
e mais difícil. Usa subset balanceado (max_per_class) — 75k estouraria a RAM.

Compara cego vs SNN feedforward vs SNN recorrente. chance = 1/35 = 2.86%.
Salva results_ssc.txt.

Uso:
    python sweep_ssc.py --epochs 8 --seeds 3 --per-class 400
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from temporal_bench import BlindMLP, SNN_FF, SNN_Rec, train, evaluate
from shd_data import SHDDataset

N_SSC = 35


def run_seeds(build, tr, te, dev, epochs, seeds, batch):
    accs = []
    for s in range(seeds):
        torch.manual_seed(s)
        m = build().to(dev)
        train(m, DataLoader(tr, batch, shuffle=True), dev, epochs)
        accs.append(evaluate(m, DataLoader(te, batch), dev))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--per-class", type=int, default=400, help="subset balanceado do train")
    ap.add_argument("--per-class-test", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    print("carregando SSC (subset balanceado)...")
    tr = SHDDataset("train", args.bins, dataset="ssc", max_per_class=args.per_class)
    te = SHDDataset("test", args.bins, dataset="ssc", max_per_class=args.per_class_test)
    print(f"  train={len(tr)} test={len(te)} n_classes={tr.n_classes}")

    out = [f"SSC (Spiking Speech Commands) — {args.seeds} seeds, {args.epochs} epochs, "
           f"subset {args.per_class}/classe train, bins={args.bins}, device={dev}",
           f"train={len(tr)} test={len(te)} classes={tr.n_classes} (chance={100/N_SSC:.2f}%)", ""]

    cego_m, cego_s = run_seeds(lambda: BlindMLP(N_SSC), tr, te, dev, args.epochs, args.seeds, args.batch)
    ff_m, ff_s = run_seeds(lambda: SNN_FF(n_classes=N_SSC), tr, te, dev, args.epochs, args.seeds, args.batch)
    rec_m, rec_s = run_seeds(lambda: SNN_Rec(n_classes=N_SSC), tr, te, dev, args.epochs, args.seeds, args.batch)
    out.append(f"BlindMLP (cego)  acc={cego_m:.2f} ±{cego_s:.2f}")
    out.append(f"SNN feedforward  acc={ff_m:.2f} ±{ff_s:.2f}")
    out.append(f"SNN recorrente   acc={rec_m:.2f} ±{rec_s:.2f}")
    out.append("")
    timing = rec_m - cego_m; recur = rec_m - ff_m
    out.append(f"timing (rec-cego)={timing:+.2f}pp | recorrencia (rec-ff)={recur:+.2f}pp | rec={rec_m:.2f}%")
    verdict = "GENERALIZA (timing agrega em SSC)" if timing >= 10.0 else "NAO GENERALIZA (margem <10pp)"
    out.append(f"-> {verdict}")
    txt = "\n".join(out)
    with open(os.path.join(os.path.dirname(__file__), "results_ssc.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)
    print("Salvo em results_ssc.txt")


if __name__ == "__main__":
    main()
