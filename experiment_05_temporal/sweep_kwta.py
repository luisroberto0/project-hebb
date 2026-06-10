"""
Marco 2-C extensão (#75) — k-WTA temporal: a esparsificação temporal preserva o timing?

Aplica k-WTA por timestep na hidden da SNN recorrente (≤k spikes ativos/timestep, de 256).
Fecha a narrativa de k-WTA do projeto: C3 (espacial, tolerante in-domain), Marco 2-A
(cross-domain, effect collapse), agora temporal (preserva ou destrói o raciocínio temporal?).

Predição: se o timing é robusto à esparsificação (como C3 in-domain), a acc se mantém até k
pequeno; se o timing exige muitos neurônios ativos, cai cedo. Sweep de k. Salva results_kwta.txt.

Uso:
    python sweep_kwta.py --epochs 8 --seeds 3
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from temporal_bench import SNN_Rec, BlindMLP, train, evaluate, HID
from shd_data import SHDDataset

KS = [None, 128, 64, 32, 16, 8, 4]  # None = denso (256); k de 256


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
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    tr = SHDDataset("train", args.bins); te = SHDDataset("test", args.bins)

    out = [f"k-WTA temporal — SHD, {args.seeds} seeds, {args.epochs} epochs, bins={args.bins}, hidden={HID}, device={dev}", ""]
    cego_m, _ = run_seeds(BlindMLP, tr, te, dev, args.epochs, args.seeds, args.batch)
    out.append(f"baseline cego (referencia): {cego_m:.2f}%")
    out.append("")
    out.append("k (de 256) | sparsity | SNN-rec acc | margem sobre cego")
    print("\n".join(out))

    dense = None
    for k in KS:
        mu, sd = run_seeds(lambda: SNN_Rec(gain=1.0, k_wta=k), tr, te, dev, args.epochs, args.seeds, args.batch)
        if k is None:
            dense = mu
        spars = "0% (denso)" if k is None else f"{100*(1-k/HID):.1f}%"
        klab = "denso" if k is None else str(k)
        drop = "" if k is None else f"  (vs denso {mu-dense:+.2f})"
        line = f"{klab:>6} | {spars:>10} | {mu:5.2f} ±{sd:.2f} | {mu-cego_m:+.2f} p.p.{drop}"
        out.append(line); print(line)

    out.append("")
    out.append("Predicao: se o timing e robusto a esparsificacao (como C3 in-domain), acc se mantem")
    out.append("ate k pequeno; se exige muitos neuronios ativos, cai cedo.")
    with open(os.path.join(os.path.dirname(__file__), "results_kwta.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\nSalvo em results_kwta.txt")


if __name__ == "__main__":
    main()
