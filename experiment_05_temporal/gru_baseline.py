"""
Marco 2-C — controle do peer review (#78): GRU não-spiking no mesmo input (T,700).

O reviewer apontou que sem um baseline recorrente NÃO-spiking não dá para saber se o ~+10 p.p.
de timing é específico de spiking/LIF ou genérico de qualquer recorrência. Este script roda um
GRU no mesmo tensor (T,700) que a SNN-rec, mesmo budget (3 seeds, 8 epochs), para comparar com
os valores conhecidos: cego 49.41% / SNN-rec 69.10% (sweep de bins, 3 seeds).

Interpretação:
  - GRU >= SNN-rec  -> o timing é genérico de recorrência, NÃO spiking-específico.
  - GRU << SNN-rec  -> o spiking/LIF agrega além da recorrência.

Salva results_gru.txt.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from shd_data import SHDDataset, N_UNITS, N_CLASSES
from temporal_bench import train, evaluate, HID

CEGO_KNOWN = 49.41   # 3-seed bins sweep
REC_KNOWN = 69.10    # SNN-rec, bins=100, 3 seeds


class GRUBaseline(nn.Module):
    """GRU não-spiking sobre o tensor (B,T,700); readout sobre a média temporal."""
    def __init__(self, n_classes=N_CLASSES, hid=HID):
        super().__init__()
        self.gru = nn.GRU(N_UNITS, hid, batch_first=True)
        self.fc = nn.Linear(hid, n_classes)

    def forward(self, x):              # x: (B, T, 700)
        out, _ = self.gru(x)           # (B, T, hid)
        return self.fc(out.mean(dim=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    tr = SHDDataset("train", args.bins, dataset="shd")
    te = SHDDataset("test", args.bins, dataset="shd")

    accs = []
    for s in range(args.seeds):
        torch.manual_seed(s)
        m = GRUBaseline().to(dev)
        t0 = time.time()
        train(m, DataLoader(tr, args.batch, shuffle=True), dev, args.epochs)
        a = evaluate(m, DataLoader(te, args.batch), dev)
        accs.append(a)
        print(f"  GRU seed={s} acc={a:.2f} ({time.time()-t0:.0f}s)")

    gru_m, gru_s = float(np.mean(accs)), float(np.std(accs))
    lines = [
        f"GRU baseline (nao-spiking) — {args.seeds} seeds, {args.epochs} epochs, bins={args.bins}, SHD",
        "",
        f"GRU (nao-spiking)  acc={gru_m:.2f} +/-{gru_s:.2f}",
        f"[ref] timing-blind (cego)  {CEGO_KNOWN:.2f}",
        f"[ref] SNN recorrente       {REC_KNOWN:.2f}",
        "",
        f"GRU - cego = {gru_m - CEGO_KNOWN:+.2f}pp (timing recuperado por recorrencia generica)",
        f"GRU - SNN  = {gru_m - REC_KNOWN:+.2f}pp",
    ]
    if gru_m >= REC_KNOWN - 1.0:
        lines.append("-> timing e GENERICO de recorrencia (GRU iguala/supera a SNN): NAO spiking-especifico")
    elif gru_m <= CEGO_KNOWN + 2.0:
        lines.append("-> GRU nao recupera o timing: o spiking/recorrencia-SNN e que agrega")
    else:
        lines.append("-> GRU recupera PARTE do timing; SNN-rec ainda a frente (spiking agrega algo)")
    txt = "\n".join(lines)
    with open(os.path.join(os.path.dirname(__file__), "results_gru.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
