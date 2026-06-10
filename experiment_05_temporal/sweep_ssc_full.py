"""
Marco 2-C extensão (#77) — SSC COMPLETO (resolver a generalização).

O subset (#76) deu margem +0.70pp com SNNs subtreinadas. Aqui: dataset SSC inteiro (~75k via
lazy loading do HDF5, sem estourar RAM) + muitos epochs, para responder definitivamente se o
timing generaliza. Foca cego vs SNN recorrente (a pergunta central). chance = 1/35 = 2.86%.

Salva results_ssc_full.txt.

Uso:
    python sweep_ssc_full.py --epochs 30 --seeds 1     # completo
    python sweep_ssc_full.py --epochs 1 --seeds 1      # smoke de velocidade
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from temporal_bench import BlindMLP, SNN_Rec, train, evaluate

N_SSC = 35


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    from shd_data import SHDDataset
    print("abrindo SSC completo (lazy)...")
    tr = SHDDataset("train", args.bins, dataset="ssc", lazy=True)
    te = SHDDataset("test", args.bins, dataset="ssc", lazy=True)
    print(f"  train={len(tr)} test={len(te)} classes={tr.n_classes}")

    out = [f"SSC COMPLETO — {args.seeds} seed(s), {args.epochs} epochs, bins={args.bins}, "
           f"train={len(tr)} test={len(te)}, device={dev} (chance={100/N_SSC:.2f}%)", ""]

    def run(name, build):
        accs = []
        for s in range(args.seeds):
            torch.manual_seed(s)
            m = build().to(dev)
            t0 = time.time()
            train(m, DataLoader(tr, args.batch, shuffle=True), dev, args.epochs)
            a = evaluate(m, DataLoader(te, args.batch), dev)
            accs.append(a)
            print(f"  {name} seed={s} acc={a:.2f} ({time.time()-t0:.0f}s)")
        return float(np.mean(accs)), float(np.std(accs))

    cego_m, cego_s = run("cego", lambda: BlindMLP(N_SSC))
    rec_m, rec_s = run("rec ", lambda: SNN_Rec(n_classes=N_SSC))
    out.append(f"BlindMLP (cego)  acc={cego_m:.2f} ±{cego_s:.2f}")
    out.append(f"SNN recorrente   acc={rec_m:.2f} ±{rec_s:.2f}")
    out.append("")
    timing = rec_m - cego_m
    out.append(f"timing (rec-cego)={timing:+.2f}pp | rec={rec_m:.2f}%")
    verdict = "GENERALIZA (timing >=10pp em SSC completo)" if timing >= 10.0 else \
              ("GENERALIZACAO FRACA (margem 3-10pp)" if timing >= 3.0 else "NAO GENERALIZA (margem <3pp)")
    out.append(f"-> {verdict}")
    txt = "\n".join(out)
    with open(os.path.join(os.path.dirname(__file__), "results_ssc_full.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
