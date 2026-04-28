"""
Sessão #15 — Caminho C1: baseline puro de Modern Hopfield Memory em Omniglot.

Pergunta: dado que Modern Hopfield Memory (Ramsauer 2020) é matematicamente
equivalente à atenção do Transformer com capacidade exponencial, qual é a
performance em few-shot quando alimentada com features triviais (sem feature
learning bio-inspirado)?

3 variações de encoding:
  C1a — Pixels flatten + L2-norm
  C1b — PCA-32 (sklearn, treinado em background set)
  C1c — Random Projection-32 (matriz ortogonal random)

Cada variação roda 5w1s + 20w1s × 1000 episódios cada, com IC95% bootstrap.

Não modifica model.py nem config.py. Reutiliza HopfieldMemory existente.

Uso:
    python c1_hopfield_baselines.py --device cuda --episodes 1000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import default_config
from data import load_evaluation, load_background, EpisodeSampler
from model import HopfieldMemory


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05):
    rng = np.random.default_rng(0)
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def fit_pca(cfg, n_components: int, n_samples: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit PCA no background set. Retorna (components, mean) como tensors no device.

    Usa sklearn pra robustez numérica, depois converte pra torch — eval roda em GPU.
    """
    from sklearn.decomposition import PCA

    print(f"[PCA-{n_components}] carregando background set...")
    bg = load_background(cfg)
    if n_samples < len(bg):
        idx = torch.randperm(len(bg), generator=torch.Generator().manual_seed(0))[:n_samples].tolist()
        bg = torch.utils.data.Subset(bg, idx)

    print(f"[PCA-{n_components}] flatten {len(bg)} amostras...")
    images = []
    for img, _ in bg:
        images.append(img.flatten().numpy())
    X = np.array(images, dtype=np.float32)

    print(f"[PCA-{n_components}] fit em X.shape={X.shape}...")
    pca = PCA(n_components=n_components)
    pca.fit(X)
    print(f"[PCA-{n_components}] explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")

    components = torch.from_numpy(pca.components_).float().to(device)  # (n_components, in_dim)
    mean = torch.from_numpy(pca.mean_).float().to(device)  # (in_dim,)
    return components, mean


def random_projection(in_dim: int, out_dim: int, seed: int, device) -> torch.Tensor:
    """Matriz ortogonal random de in_dim → out_dim, no device."""
    g = torch.Generator().manual_seed(seed)
    M = torch.empty(out_dim, in_dim)
    torch.nn.init.orthogonal_(M, generator=g)
    return M.to(device)


def encode_pixels(images: torch.Tensor) -> torch.Tensor:
    """C1a: flatten + L2-norm."""
    flat = images.flatten(start_dim=1)
    return F.normalize(flat, dim=-1)


def encode_pca(images: torch.Tensor, pca_components: torch.Tensor, pca_mean: torch.Tensor) -> torch.Tensor:
    """C1b: PCA-32 via tensor (pré-computado em GPU)."""
    flat = images.flatten(start_dim=1)
    centered = flat - pca_mean
    proj = centered @ pca_components.T  # (B, n_components)
    return F.normalize(proj, dim=-1)


def encode_random_proj(images: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """C1c: random orthogonal projection."""
    flat = images.flatten(start_dim=1)
    proj = flat @ M.T
    return F.normalize(proj, dim=-1)


def run_evaluation(encoder_fn, name: str, cfg, dataset, n_way: int, k_shot: int,
                   n_queries: int, n_episodes: int, seed: int, device):
    sampler = EpisodeSampler(dataset, n_way=n_way, k_shot=k_shot, n_query=n_queries, seed=seed)
    memory = HopfieldMemory(cfg).to(device)
    accs = []
    centered_cosines = []
    t0 = time.time()
    with torch.no_grad():
        for ep in range(n_episodes):
            episode = sampler.sample()
            support = episode.support.to(device)
            support_labels = episode.support_labels.to(device)
            query = episode.query.to(device)
            query_labels = episode.query_labels.to(device)

            sup_emb = encoder_fn(support)
            qry_emb = encoder_fn(query)

            # Centered cosine entre support embeddings (medida de diversidade)
            sup_centered = sup_emb - sup_emb.mean(dim=0, keepdim=True)
            sup_n = F.normalize(sup_centered, dim=-1)
            cos = sup_n @ sup_n.T
            n = cos.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool, device=device)
            centered_cosines.append(cos[mask].mean().item())

            memory.store(sup_emb, support_labels, n_classes=n_way)
            logits = memory.query(qry_emb)
            preds = logits.argmax(dim=-1)
            accs.append((preds == query_labels).float().mean().item())

            if (ep + 1) % 250 == 0:
                print(f"  {name} {n_way}w{k_shot}s ep {ep+1:4d}  acc rolling={np.mean(accs)*100:.2f}%")

    accs_arr = np.array(accs)
    mean = accs_arr.mean()
    lo, hi = bootstrap_ci(accs_arr)
    chance = 1.0 / n_way
    z = (mean - chance) / accs_arr.std() if accs_arr.std() > 0 else float("inf")
    avg_cos = float(np.mean(centered_cosines))
    elapsed = time.time() - t0
    print(f"  {name} {n_way}w{k_shot}s: {mean*100:.2f}% IC[{lo*100:.2f},{hi*100:.2f}] z≈{z:.1f} cos_cent={avg_cos:.4f} ({elapsed:.1f}s)")
    return {"acc": mean, "lo": lo, "hi": hi, "z": z, "cos_cent": avg_cos}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pca-samples", type=int, default=5000,
                   help="Amostras do background set pra fit do PCA (5000 é razoável)")
    args = p.parse_args()

    cfg = default_config()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Dispositivo: {device}")
    torch.manual_seed(args.seed)

    # Fit PCA UMA vez (no background set, separado do evaluation set)
    pca_components, pca_mean = fit_pca(cfg, n_components=32, n_samples=args.pca_samples, device=device)

    # Random projection ortogonal UMA vez
    R = random_projection(in_dim=28*28, out_dim=32, seed=args.seed, device=device)
    print(f"[RandomProj] matriz ortogonal shape={R.shape}")

    # Carrega evaluation set UMA vez
    print("\nCarregando Omniglot evaluation set...")
    eval_dataset = load_evaluation(cfg)

    encoders = {
        "C1a (Pixels)":      lambda x: encode_pixels(x),
        "C1b (PCA-32)":      lambda x: encode_pca(x, pca_components, pca_mean),
        "C1c (RandomProj-32)": lambda x: encode_random_proj(x, R),
    }

    print(f"\nMemory config: beta={cfg.memory.beta}, distance={cfg.memory.distance}, normalize_keys={cfg.memory.normalize_keys}")

    print("\n" + "="*72)
    print(f"AVALIAÇÃO 5-WAY 1-SHOT × {args.episodes} eps")
    print("="*72)
    results_5w = {}
    for name, enc in encoders.items():
        results_5w[name] = run_evaluation(enc, name, cfg, eval_dataset,
                                          n_way=5, k_shot=1, n_queries=5,
                                          n_episodes=args.episodes, seed=args.seed, device=device)

    print("\n" + "="*72)
    print(f"AVALIAÇÃO 20-WAY 1-SHOT × {args.episodes} eps")
    print("="*72)
    results_20w = {}
    for name, enc in encoders.items():
        results_20w[name] = run_evaluation(enc, name, cfg, eval_dataset,
                                           n_way=20, k_shot=1, n_queries=5,
                                           n_episodes=args.episodes, seed=args.seed, device=device)

    print("\n" + "="*72)
    print("TABELA COMPARATIVA — Hopfield Memory + features triviais")
    print("="*72)
    print(f"{'Encoder':22s} | {'5w1s acc':>10s} | {'5w1s IC95%':>16s} | {'20w1s acc':>10s} | {'cos cent':>10s}")
    print("-" * 72)
    for name in encoders:
        r5 = results_5w[name]
        r20 = results_20w[name]
        print(f"{name:22s} | {r5['acc']*100:>9.2f}% | [{r5['lo']*100:>5.2f},{r5['hi']*100:>5.2f}] | {r20['acc']*100:>9.2f}% | {r5['cos_cent']:>10.4f}")
    print("-" * 72)
    print(f"{'Refs (sessão #7)':22s} | {'45.76%':>10s} (Pixel kNN)        | {'~':>10s}        |")
    print(f"{'':22s} | {'85.88%':>10s} (ProtoNet)         | {'~':>10s}        |")
    print(f"{'Iter 1 STDP (#9)':22s} | {'35.98%':>10s} (saturado)         | {'9.80%':>10s}     |")


if __name__ == "__main__":
    main()
