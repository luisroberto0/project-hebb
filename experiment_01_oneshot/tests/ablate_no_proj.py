"""
Sessão #10 — Ablação A1: bypass de _proj.

Monkey-patcha STDPHopfieldModel.extract_features pra retornar `flat`
(784D pós-pool, taxa de spikes) direto, sem a projeção ortogonal de 784→64.

Hopfield com cosseno deveria ter sinal ainda — mas como _proj reduz dim
descartando 720 direções, o resultado pode subir, descer ou ficar igual.

Predição: ambígua a priori. Ponto é medir, não prever.

Uso:
    python tests/ablate_no_proj.py --checkpoint checkpoints/stdp_model_iter1_seed42.pt --episodes 1000
"""
from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

# Garante que o módulo do experimento entra no path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_config
from data import load_evaluation, EpisodeSampler, encode
from model import STDPHopfieldModel


def extract_features_no_proj(self, images: torch.Tensor, train_stdp: bool = False) -> torch.Tensor:
    """Cópia de STDPHopfieldModel.extract_features SEM a projeção final."""
    B, H, W = images.shape
    device = images.device
    x = images.unsqueeze(1)
    spikes_in = encode(x, self.cfg)
    T = spikes_in.shape[0]

    out1_shape, out2_shape, pool1_shape, pool2_shape = self._compute_shapes((H, W))
    mem1 = torch.zeros((B,) + out1_shape, device=device)
    mem2 = torch.zeros((B,) + out2_shape, device=device)
    spike_count_final = torch.zeros((B,) + pool2_shape, device=device)

    for t in range(T):
        pre1 = spikes_in[t]
        spk1, mem1 = self.layer1(pre1, mem1)
        spk1_pool = self.pool1(spk1)
        spk2, mem2 = self.layer2(spk1_pool, mem2)
        spk2_pool = self.pool2(spk2)
        spike_count_final = spike_count_final + spk2_pool

    rate = spike_count_final / T
    flat = rate.flatten(start_dim=1)
    return flat  # SEM _proj


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05):
    rng = np.random.default_rng(0)
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--ways", type=int, default=5)
    p.add_argument("--shots", type=int, default=1)
    p.add_argument("--queries", type=int, default=5)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = default_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    torch.manual_seed(args.seed)

    print("Carregando Omniglot evaluation...")
    dataset = load_evaluation(cfg)
    sampler = EpisodeSampler(
        dataset, n_way=args.ways, k_shot=args.shots,
        n_query=args.queries, seed=args.seed,
    )
    print(f"  {len(sampler.classes)} classes disponíveis")

    model = STDPHopfieldModel(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"  Checkpoint: {args.checkpoint}")

    # Monkey-patch
    model.extract_features = types.MethodType(extract_features_no_proj, model)
    print("  ⚠️  extract_features patched: bypass de _proj (flat 784D direto)")
    model.eval()

    print(f"\nAvaliação no-proj: {args.ways}w{args.shots}s, {args.episodes} eps")
    accs = []
    t0 = time.time()
    with torch.no_grad():
        for ep in range(args.episodes):
            episode = sampler.sample()
            support = episode.support.to(device)
            support_labels = episode.support_labels.to(device)
            query = episode.query.to(device)
            query_labels = episode.query_labels.to(device)
            logits = model(support, support_labels, query, n_classes=args.ways)
            preds = logits.argmax(dim=-1)
            accs.append((preds == query_labels).float().mean().item())
            if (ep + 1) % 100 == 0:
                print(f"  ep {ep+1:4d}  acc rolling={np.mean(accs)*100:.2f}%")

    accs_arr = np.array(accs)
    mean = accs_arr.mean()
    lo, hi = bootstrap_ci(accs_arr)
    chance = 1.0 / args.ways
    z = (mean - chance) / accs_arr.std() if accs_arr.std() > 0 else float("inf")
    print(f"\nResultado A1 (no_proj, {args.episodes} eps em {time.time()-t0:.1f}s):")
    print(f"  Acurácia: {mean*100:.2f}%   IC95%: [{lo*100:.2f}%, {hi*100:.2f}%]")
    print(f"  Chance: {chance*100:.2f}%   Δchance: {(mean-chance)*100:.2f} p.p.  z≈{z:.1f}")


if __name__ == "__main__":
    main()
