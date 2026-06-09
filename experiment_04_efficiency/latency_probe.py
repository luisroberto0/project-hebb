"""
Marco 2-B (#69) — teste decisivo de latência: inferência event-driven (sparse) realiza
a vantagem teórica de SynOps em CPU comum?

O smoke (efficiency_bench.py) mostrou que a SNN com runtime DENSO é 60-300x mais lenta que
o MLP, apesar de (no melhor caso) fazer menos SynOps. Este probe testa o caminho que restaria
pra um Sucesso: uma inferência EVENT-DRIVEN que só computa contribuições de neurônios que
disparam (W[:, ativos].sum), pulando os silenciosos. É o "co-design de runtime" que o achado aponta.

A latência depende da arquitetura + esparsidade, NÃO dos valores dos pesos — então medimos com
pesos random e esparsidade controlada (válido pra a pergunta de latência). Validamos que o
forward event-driven dá o MESMO output que o denso.

Uso:
    python latency_probe.py --T 10 --k-in 32 --k 32
"""
from __future__ import annotations
import argparse, time
import torch
import torch.nn as nn

IN, HID, OUT = 28 * 28, 256, 10


def topk_mask(v: torch.Tensor, k: int) -> torch.Tensor:
    m = torch.zeros_like(v)
    return m.scatter_(0, v.topk(k).indices, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--k-in", type=int, default=32, help="input spikes ativos/timestep")
    ap.add_argument("--k", type=int, default=32, help="hidden spikes ativos/timestep (k-WTA)")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(1)

    W1 = torch.randn(HID, IN);  b1 = torch.randn(HID)
    W2 = torch.randn(OUT, HID); b2 = torch.randn(OUT)
    beta = 0.95

    # Pré-gera spike trains com esparsidade controlada (T timesteps), batch=1.
    inten = torch.rand(IN)
    spk_in = torch.stack([topk_mask(inten * torch.rand(IN), args.k_in) for _ in range(args.T)])  # (T, IN)

    # --- dense MLP (1 matmul) ---
    def dense_mlp():
        x = torch.rand(IN)
        h = torch.relu(W1 @ x + b1)
        return W2 @ h + b2

    # --- SNN dense runtime (T matmuls completos) ---
    def snn_dense():
        mem1 = torch.zeros(HID); mem2 = torch.zeros(OUT); out = torch.zeros(OUT)
        for t in range(args.T):
            cur1 = W1 @ spk_in[t] + b1
            mem1 = beta * mem1 + cur1
            spk1 = (mem1 > 1.0).float(); mem1 = mem1 * (1 - spk1)
            spk1 = spk1 * topk_mask(mem1 + spk1, args.k)  # k-WTA hidden
            cur2 = W2 @ spk1 + b2
            mem2 = beta * mem2 + cur2
            spk2 = (mem2 > 1.0).float(); mem2 = mem2 * (1 - spk2)
            out = out + spk2
        return out

    # --- SNN event-driven (só colunas de neurônios ativos) ---
    def snn_eventdriven():
        mem1 = torch.zeros(HID); mem2 = torch.zeros(OUT); out = torch.zeros(OUT)
        for t in range(args.T):
            active_in = spk_in[t].nonzero(as_tuple=True)[0]
            cur1 = W1[:, active_in].sum(dim=1) + b1            # pula entradas silenciosas
            mem1 = beta * mem1 + cur1
            spk1 = (mem1 > 1.0).float(); mem1 = mem1 * (1 - spk1)
            spk1 = spk1 * topk_mask(mem1 + spk1, args.k)
            active_hid = spk1.nonzero(as_tuple=True)[0]
            cur2 = W2[:, active_hid].sum(dim=1) + b2           # pula hidden silenciosos
            mem2 = beta * mem2 + cur2
            spk2 = (mem2 > 1.0).float(); mem2 = mem2 * (1 - spk2)
            out = out + spk2
        return out

    # Validar equivalência dense vs event-driven (mesma dinâmica)
    o_d, o_e = snn_dense(), snn_eventdriven()
    assert torch.allclose(o_d, o_e, atol=1e-4), f"event-driven != dense: {(o_d-o_e).abs().max()}"

    def bench(fn, reps):
        for _ in range(20):
            fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1000.0

    lat_mlp = bench(dense_mlp, args.reps)
    lat_snn_d = bench(snn_dense, args.reps)
    lat_snn_e = bench(snn_eventdriven, args.reps)

    print(f"T={args.T}  k_in={args.k_in}  k={args.k}  (CPU single-thread, batch=1)")
    print(f"  DenseMLP (1 matmul)        : {lat_mlp:7.4f} ms   (1.00x)")
    print(f"  SNN dense runtime (T loop) : {lat_snn_d:7.4f} ms   ({lat_snn_d/lat_mlp:5.1f}x)")
    print(f"  SNN event-driven (sparse)  : {lat_snn_e:7.4f} ms   ({lat_snn_e/lat_mlp:5.1f}x)")
    print(f"  event-driven vs MLP        : {'FASTER' if lat_snn_e < lat_mlp else 'SLOWER'} "
          f"({lat_mlp/lat_snn_e:.2f}x the MLP speed)")


if __name__ == "__main__":
    main()
