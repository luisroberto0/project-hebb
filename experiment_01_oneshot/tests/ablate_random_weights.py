"""
Sessão #10 — Ablação A3: pesos random U(0,1) sem treino.

Cria STDPHopfieldModel novo, sobrescreve pesos com U(0,1) (matching
o range pós-treino do Iter 1, que satura em ~0.999), zera theta
(modelo nunca foi treinado), salva ckpt, roda eval.

Predição: se ≈36%, STDP não está aprendendo nada útil — toda a
estrutura vem da arquitetura + Hopfield + Poisson. Se cai pra chance,
o treino contribui (pesos OU theta).

Nota: este teste mistura "pesos não-treinados" + "theta=0". Se A3 for
chance e A2 também (já é chance — confirmado), podemos pedir A3b
adicional: random pesos + theta_iter1 (preservada do checkpoint),
pra decompor se o que importa é peso treinado, theta treinada, ou ambos.

Uso:
    python tests/ablate_random_weights.py --episodes 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_config
from model import STDPHopfieldModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="checkpoints/stdp_model_random_u01.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = default_config()
    torch.manual_seed(args.seed)

    model = STDPHopfieldModel(cfg)

    # Override init: U(0,1) em vez do U(0, 0.3) padrão
    with torch.no_grad():
        model.layer1.conv.weight.uniform_(0.0, 1.0)
        model.layer2.conv.weight.uniform_(0.0, 1.0)
        # theta começa em zero (modelo fresh, nunca recebeu spikes)
        model.layer1.theta.zero_()
        model.layer2.theta.zero_()

    sd = model.state_dict()
    print(f"Modelo random U(0,1) gerado:")
    print(f"  layer1.conv.weight  μ={sd['layer1.conv.weight'].mean().item():.4f}  σ={sd['layer1.conv.weight'].std().item():.4f}")
    print(f"  layer2.conv.weight  μ={sd['layer2.conv.weight'].mean().item():.4f}  σ={sd['layer2.conv.weight'].std().item():.4f}")
    print(f"  layer1.theta = {sd['layer1.theta'].max().item():.4f}")
    print(f"  layer2.theta = {sd['layer2.theta'].max().item():.4f}")

    torch.save(sd, args.out)
    print(f"\nSalvo em {args.out}")


if __name__ == "__main__":
    main()
