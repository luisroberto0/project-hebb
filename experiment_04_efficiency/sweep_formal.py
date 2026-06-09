"""
Marco 2-B (#70) — sweep formal: 5 seeds, IC95% bootstrap, Fashion-MNIST completo.
Confirma (ou refuta) o achado negativo do smoke com rigor estatístico antes do fechamento.

Reusa o harness de efficiency_bench.py. Treina cada (config, seed), agrega acc + SynOps
sobre seeds com IC95% bootstrap; latência CPU medida uma vez por config (determinística
pela arquitetura). Salva tabela em results_sweep.txt.

Uso:
    python sweep_formal.py --epochs 5 --seeds 5
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch

from efficiency_bench import DenseMLP, SNN, get_loaders, train, evaluate, cpu_latency

CONFIGS = [
    ("DenseMLP",                 lambda: DenseMLP(),                  False),
    ("SNN vanilla T=25",         lambda: SNN(T=25),                   True),
    ("SNN vanilla T=10",         lambda: SNN(T=10),                   True),
    ("SNN kWTAin T=10 ki=64",    lambda: SNN(T=10, k=32, k_in=64),    True),
    ("SNN kWTAin T=5 ki=32",     lambda: SNN(T=5,  k=32, k_in=32),    True),
]


def boot_ci(vals, n=1000, seed=123):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    means = [rng.choice(vals, vals.size, replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    train_loader, test_loader = get_loaders(args.batch, quick=False)
    dense_macs = float(DenseMLP.macs)

    out = [f"Sweep formal — Fashion-MNIST, {args.seeds} seeds, {args.epochs} epochs, device={dev}",
           f"Dense MACs/sample = {dense_macs:.0f}", ""]
    rows = []
    for name, build, is_snn in CONFIGS:
        accs, synops_list = [], []
        for s in range(args.seeds):
            torch.manual_seed(s)
            model = build().to(dev)
            train(model, train_loader, dev, args.epochs, is_snn)
            acc, synops = evaluate(model, test_loader, dev, is_snn)
            accs.append(acc)
            if synops is not None:
                synops_list.append(synops)
            print(f"  {name:24s} seed={s} acc={acc:.2f} synops={synops}")
        lat = cpu_latency(build().to("cpu"), is_snn)  # latência: arquitetura, não treino
        acc_m, acc_s = float(np.mean(accs)), float(np.std(accs))
        lo, hi = boot_ci(accs)
        syn_m = float(np.mean(synops_list)) if synops_list else dense_macs
        cost_ratio = dense_macs / syn_m  # >1 = menos ops que denso
        rows.append((name, acc_m, acc_s, lo, hi, syn_m, cost_ratio, lat, is_snn))
        line = (f"{name:24s} acc={acc_m:5.2f}±{acc_s:.2f} CI[{lo:.2f},{hi:.2f}]  "
                f"{'SynOps' if is_snn else 'MACs'}={syn_m:9.0f} "
                f"({cost_ratio:5.2f}x vs denso)  cpu_lat={lat:6.3f}ms")
        out.append(line); print(line)

    # Veredicto vs critério (acc -2pp E SynOps >=5x menores E latencia <= denso)
    dense = rows[0]
    out += ["", "=== Critério: acc dentro de -2pp E SynOps>=5x menores E latencia CPU<=denso ==="]
    for r in rows[1:]:
        name, acc_m, _, _, _, _, cost_ratio, lat, _ = r
        acc_ok = (dense[1] - acc_m) <= 2.0
        syn_ok = cost_ratio >= 5.0
        lat_ok = lat <= dense[7]
        verdict = "SUCESSO" if (acc_ok and syn_ok and lat_ok) else "FALHA"
        out.append(f"{name:24s} acc_ok={acc_ok!s:5} syn>=5x={syn_ok!s:5} "
                   f"lat<=denso={lat_ok!s:5}  -> {verdict}")
    txt = "\n".join(out)
    with open(os.path.join(os.path.dirname(__file__), "results_sweep.txt"), "w") as f:
        f.write(txt + "\n")
    print("\n" + txt.split("=== Critério")[1] if "=== Critério" in txt else "")
    print("\nSalvo em results_sweep.txt")


if __name__ == "__main__":
    main()
