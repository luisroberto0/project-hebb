"""
Sessão #10 — Ablação A3b: random U(0,1) + theta_iter1 preservada.

Carrega checkpoint Iter 1 pra extrair theta1 e theta2 treinadas, sobrescreve
APENAS os conv weights com U(0,1) random. Preserva theta_iter1 (~20.6, ~30.8).

Critério interpretativo:
- ≈36%: theta agrega 3 p.p. sobre random magnitude → STDP da conv é totalmente irrelevante
- ≈33%: theta também não agrega → 3 p.p. vêm de algo nos pesos saturados

Uso:
    python tests/ablate_random_with_theta.py --base checkpoint_iter1.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, required=True, help="Checkpoint pra extrair theta")
    p.add_argument("--out", type=str, default="checkpoints/stdp_model_random_with_theta.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    sd = torch.load(args.base, map_location="cpu")

    print(f"Base (theta preservada): {args.base}")
    print(f"  layer1.theta μ={sd['layer1.theta'].mean().item():.4f} max={sd['layer1.theta'].max().item():.4f}")
    print(f"  layer2.theta μ={sd['layer2.theta'].mean().item():.4f} max={sd['layer2.theta'].max().item():.4f}")

    # Sobrescreve só os conv weights com U(0,1) random
    sd['layer1.conv.weight'] = torch.empty_like(sd['layer1.conv.weight']).uniform_(0.0, 1.0)
    sd['layer2.conv.weight'] = torch.empty_like(sd['layer2.conv.weight']).uniform_(0.0, 1.0)

    print(f"\nConv weights sobrescritos com U(0,1):")
    print(f"  layer1.conv.weight μ={sd['layer1.conv.weight'].mean().item():.4f} σ={sd['layer1.conv.weight'].std().item():.4f}")
    print(f"  layer2.conv.weight μ={sd['layer2.conv.weight'].mean().item():.4f} σ={sd['layer2.conv.weight'].std().item():.4f}")

    torch.save(sd, args.out)
    print(f"\nSalvo em {args.out}")


if __name__ == "__main__":
    main()
