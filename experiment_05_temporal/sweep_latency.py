"""
Marco 2-C extensão (#74) — latency coding (time-to-first-spike): a SNN explora o timing FINO?

rate coding = contagem de spikes por bin (atual). latency coding = 1 spike por canal, no bin
do PRIMEIRO spike. Se a SNN recorrente mantém alta acc sob latency, a informação está no
TIMING do onset (latência), não só na contagem. Compara cego vs SNN-rec sob os dois encodings.

O cego (soma sobre o tempo): sob rate = histograma de contagem; sob latency = presença binária
por canal (perde a latência). Margem SNN−cego sob latency isola o uso da latência pela dinâmica.

Salva results_latency.txt.

Uso:
    python sweep_latency.py --epochs 8 --seeds 3
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from temporal_bench import SNN_Rec, BlindMLP, train, evaluate
from shd_data import SHDDataset


def run_seeds(build, tr_ds, te_ds, batch, epochs, dev, seeds):
    accs = []
    for s in range(seeds):
        torch.manual_seed(s)
        m = build().to(dev)
        train(m, DataLoader(tr_ds, batch, shuffle=True), dev, epochs)
        accs.append(evaluate(m, DataLoader(te_ds, batch), dev))
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

    tr_ds = SHDDataset("train", n_bins=args.bins)
    te_ds = SHDDataset("test", n_bins=args.bins)

    out = [f"Latency vs rate coding — SHD, {args.seeds} seeds, {args.epochs} epochs, bins={args.bins}, device={dev}", ""]
    summary = {}
    for coding in ["rate", "latency"]:
        tr_ds.coding = coding; te_ds.coding = coding
        cego_m, cego_s = run_seeds(BlindMLP, tr_ds, te_ds, args.batch, args.epochs, dev, args.seeds)
        snn_m, snn_s = run_seeds(lambda: SNN_Rec(gain=1.0), tr_ds, te_ds, args.batch, args.epochs, dev, args.seeds)
        summary[coding] = (cego_m, snn_m)
        out.append(f"[{coding}]  cego {cego_m:.2f}±{cego_s:.2f}  |  SNN-rec {snn_m:.2f}±{snn_s:.2f}  |  margem {snn_m-cego_m:+.2f} p.p.")
        print(out[-1])

    out.append("")
    r_cego, r_snn = summary["rate"]; l_cego, l_snn = summary["latency"]
    out.append(f"SNN-rec: rate {r_snn:.2f}% vs latency {l_snn:.2f}%  (delta {l_snn-r_snn:+.2f} p.p.)")
    out.append("Interpretacao: se latency preserva a acc/margem, o timing fino (onset) carrega a info;")
    out.append("se cai muito, a contagem de spikes (rate) era o que importava.")
    with open(os.path.join(os.path.dirname(__file__), "results_latency.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nSNN-rec rate={r_snn:.2f} latency={l_snn:.2f} delta={l_snn-r_snn:+.2f}pp | Salvo em results_latency.txt")


if __name__ == "__main__":
    main()
