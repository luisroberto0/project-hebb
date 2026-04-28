"""
Sessão #10 — Ablação A2: zera conv weights, mantém theta treinada.

Carrega checkpoint Iter 1, zera layer1.conv.weight + layer2.conv.weight,
preserva todos os demais buffers (theta1, theta2). Salva como
stdp_model_zeroed.pt e roda evaluate normal.

Predição rigorosa: w=0 ⇒ I=conv(spk)=0 ⇒ mem só recebe leakage ⇒
nunca atinge v_thresh+theta (~21) ⇒ zero spikes pós ⇒ embedding zero
⇒ acc = chance (~20%). Se for ~36%, há bypass não-óbvio no pipeline.

Uso:
    python tests/ablate_zero_conv.py --checkpoint checkpoints/stdp_model_iter1_seed42.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, default="checkpoints/stdp_model_zeroed.pt")
    args = p.parse_args()

    sd = torch.load(args.checkpoint, map_location="cpu")
    print(f"Carregado: {args.checkpoint}")
    print(f"Chaves: {list(sd.keys())}")

    print(f"\nAntes:")
    print(f"  layer1.conv.weight  μ={sd['layer1.conv.weight'].mean().item():.4f}  σ={sd['layer1.conv.weight'].std().item():.4f}")
    print(f"  layer2.conv.weight  μ={sd['layer2.conv.weight'].mean().item():.4f}  σ={sd['layer2.conv.weight'].std().item():.4f}")
    print(f"  layer1.theta        μ={sd['layer1.theta'].mean().item():.4f}  max={sd['layer1.theta'].max().item():.4f}")
    print(f"  layer2.theta        μ={sd['layer2.theta'].mean().item():.4f}  max={sd['layer2.theta'].max().item():.4f}")

    sd['layer1.conv.weight'] = torch.zeros_like(sd['layer1.conv.weight'])
    sd['layer2.conv.weight'] = torch.zeros_like(sd['layer2.conv.weight'])

    print(f"\nDepois (zerados):")
    print(f"  layer1.conv.weight  μ={sd['layer1.conv.weight'].mean().item():.4f}  max={sd['layer1.conv.weight'].max().item():.4f}")
    print(f"  layer2.conv.weight  μ={sd['layer2.conv.weight'].mean().item():.4f}  max={sd['layer2.conv.weight'].max().item():.4f}")
    print(f"  layer1.theta (preservada) μ={sd['layer1.theta'].mean().item():.4f}")

    torch.save(sd, args.out)
    print(f"\nSalvo em {args.out}")


if __name__ == "__main__":
    main()
