"""
Marco 2-C (#72) — sweep formal: 5 seeds, IC95% bootstrap, SHD completo.
Confirma o smoke positivo (timing agrega +21pp) com rigor antes de declarar Sucesso.

Reusa o harness de temporal_bench.py. Salva tabela em results_temporal.txt.

Uso:
    python sweep_temporal.py --epochs 15 --seeds 5
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from temporal_bench import BlindMLP, SNN_FF, SNN_Rec, train, evaluate
from shd_data import get_shd_loaders

CONFIGS = [
    ("BlindMLP (cego)", lambda: BlindMLP()),
    ("SNN feedforward", lambda: SNN_FF(gain=1.0)),
    ("SNN recorrente",  lambda: SNN_Rec(gain=1.0)),
]


def boot_ci(vals, n=1000, seed=123):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    means = [rng.choice(vals, vals.size, replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    tr, te = get_shd_loaders(args.batch, args.bins)

    out = [f"Sweep formal temporal — SHD, {args.seeds} seeds, {args.epochs} epochs, bins={args.bins}, device={dev}", ""]
    res = {}
    for name, build in CONFIGS:
        accs = []
        for s in range(args.seeds):
            torch.manual_seed(s)
            m = build().to(dev)
            train(m, tr, dev, args.epochs)
            a = evaluate(m, te, dev)
            accs.append(a)
            print(f"  {name:18s} seed={s} acc={a:.2f}")
        m_, s_ = float(np.mean(accs)), float(np.std(accs))
        lo, hi = boot_ci(accs)
        res[name] = (m_, s_, lo, hi, accs)
        line = f"{name:18s} acc={m_:5.2f} ±{s_:.2f}  CI95[{lo:.2f}, {hi:.2f}]  seeds={[round(a,1) for a in accs]}"
        out.append(line); print(line)

    cego = res["BlindMLP (cego)"][0]
    ff = res["SNN feedforward"][0]
    rec = res["SNN recorrente"][0]
    out += ["", "=== Critério: SNN recorrente − cego ≥ 10 p.p. E SNN recorrente ≥ 65% ==="]
    timing = rec - cego; recur = rec - ff
    out.append(f"  timing (rec − cego):    {timing:+.2f} p.p.   (criterio >=10)")
    out.append(f"  recorrencia (rec − ff): {recur:+.2f} p.p.")
    out.append(f"  SNN recorrente:         {rec:.2f}%            (criterio >=65)")
    verdict = "SUCESSO" if (timing >= 10.0 and rec >= 65.0) else ("MEDIANO" if (timing >= 10.0 or rec >= 65.0) else "FALHA")
    out.append(f"  -> {verdict}")
    txt = "\n".join(out)
    with open(os.path.join(os.path.dirname(__file__), "results_temporal.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt.split("=== Critério")[-1])
    print("Salvo em results_temporal.txt")


if __name__ == "__main__":
    main()
