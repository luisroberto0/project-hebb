"""
Sessão #16 — C1d: Hopfield Memory + autoencoder MLP simples como encoder.

Pergunta: dado que C1b (Hopfield + PCA-32) atinge 56.28% 5w1s, features
APRENDIDAS via autoencoder não-linear simples (reconstrução MSE, sem
meta-objetivo) podem capturar estrutura adicional (invariâncias, simetrias)
e bater C1b?

Arquitetura (fixa, não modificar mid-sessão conforme protocolo):
  encoder: 784 → 128 → 32 (ReLU)
  decoder: 32 → 128 → 784 (sigmoid no fim)
  loss: MSE
  optim: Adam lr=1e-3, batch=64
  data: 5000 imgs do background set, 30 epochs

Eval: encoder(image) → L2-norm → 32D → HopfieldMemory.

Não modifica model.py nem config.py. Reusa HopfieldMemory + EpisodeSampler.

Uso:
    python c1d_autoencoder_baseline.py --device cuda --episodes 1000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import default_config
from data import load_evaluation, load_background, EpisodeSampler
from model import HopfieldMemory


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05):
    rng = np.random.default_rng(0)
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


class AutoencoderMLP(nn.Module):
    """Autoencoder MLP simples 784 → 128 → 32 → 128 → 784."""

    def __init__(self, in_dim: int = 784, hidden: int = 128, latent: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def train_autoencoder(cfg, n_samples: int, epochs: int, batch_size: int, lr: float,
                      latent: int, device, seed: int) -> tuple[AutoencoderMLP, list[float]]:
    """Treina AE com MSE no background set. Retorna modelo + lista de losses por epoch."""
    print(f"[AE] carregando background set...")
    bg = load_background(cfg)
    if n_samples < len(bg):
        idx = torch.randperm(len(bg), generator=torch.Generator().manual_seed(seed))[:n_samples].tolist()
        bg = Subset(bg, idx)
    print(f"[AE] {len(bg)} amostras de treino, {epochs} epochs, batch={batch_size}, lr={lr}, latent={latent}")

    loader = DataLoader(bg, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                        generator=torch.Generator().manual_seed(seed))

    torch.manual_seed(seed)
    model = AutoencoderMLP(in_dim=28*28, hidden=128, latent=latent).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[AE] params: {n_params}")

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    losses_per_epoch = []

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for img, _ in loader:
            img = img.to(device).flatten(start_dim=1)  # (B, 784)
            x_hat, _ = model(img)
            loss = F.mse_loss(x_hat, img)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())
        avg = float(np.mean(epoch_losses))
        losses_per_epoch.append(avg)
        elapsed = time.time() - t0
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"[AE] epoch {epoch+1:2d}/{epochs}  MSE={avg:.5f}  elapsed={elapsed:.1f}s")
        # Watchdog: se passar de 20 min, para
        if elapsed > 1200:
            print(f"[AE] ⚠️ Treino passou de 20 min ({elapsed:.0f}s), parando antecipadamente em epoch {epoch+1}")
            break

    print(f"[AE] treino concluído em {time.time()-t0:.1f}s. Loss final: {losses_per_epoch[-1]:.5f}")
    return model, losses_per_epoch


def encode_with_ae(images: torch.Tensor, ae: AutoencoderMLP) -> torch.Tensor:
    """Aplica encoder do AE + L2-norm. Eval mode (sem grad)."""
    ae.eval()
    with torch.no_grad():
        flat = images.flatten(start_dim=1)
        z = ae.encoder(flat)
    return F.normalize(z, dim=-1)


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
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent", type=int, default=32)
    args = p.parse_args()

    cfg = default_config()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Dispositivo: {device}")
    torch.manual_seed(args.seed)

    # Treina AE
    ae, losses = train_autoencoder(cfg, n_samples=args.n_train, epochs=args.epochs,
                                   batch_size=args.batch, lr=args.lr, latent=args.latent,
                                   device=device, seed=args.seed)

    # Carrega evaluation
    print("\nCarregando Omniglot evaluation set...")
    eval_dataset = load_evaluation(cfg)

    # Eval
    print(f"\nMemory config: beta={cfg.memory.beta}, distance={cfg.memory.distance}, normalize_keys={cfg.memory.normalize_keys}")

    print("\n" + "="*72)
    print(f"AVALIAÇÃO C1d (Hopfield + Autoencoder-{args.latent}) × {args.episodes} eps")
    print("="*72)
    r5w = run_evaluation(lambda x: encode_with_ae(x, ae), f"C1d (AE-{args.latent})",
                        cfg, eval_dataset, n_way=5, k_shot=1, n_queries=5,
                        n_episodes=args.episodes, seed=args.seed, device=device)
    r20w = run_evaluation(lambda x: encode_with_ae(x, ae), f"C1d (AE-{args.latent})",
                         cfg, eval_dataset, n_way=20, k_shot=1, n_queries=5,
                         n_episodes=args.episodes, seed=args.seed, device=device)

    print("\n" + "="*72)
    print("TABELA C1d vs família C1 (sessão #15)")
    print("="*72)
    print(f"{'Encoder':25s} | {'5w1s':>10s} | {'IC95% 5w1s':>16s} | {'20w1s':>10s} | {'cos cent':>10s}")
    print("-" * 80)
    print(f"{'C1a Pixels+L2 (#15)':25s} | {'50.17%':>10s} | {'[49.34, 50.96]':>16s} | {'30.30%':>10s} | {'-0.2485':>10s}")
    print(f"{'C1b PCA-32 (#15)':25s} | {'56.28%':>10s} | {'[55.50, 57.11]':>16s} | {'35.37%':>10s} | {'-0.2476':>10s}")
    print(f"{'C1c RandomProj-32 (#15)':25s} | {'41.23%':>10s} | {'[40.51, 41.95]':>16s} | {'20.05%':>10s} | {'-0.2462':>10s}")
    print(f"{'C1d AE-' + str(args.latent) + ' (#16)':25s} | {r5w['acc']*100:>9.2f}% | [{r5w['lo']*100:>5.2f}, {r5w['hi']*100:>5.2f}] | {r20w['acc']*100:>9.2f}% | {r5w['cos_cent']:>10.4f}")
    print("-" * 80)
    print(f"\nReferências:")
    print(f"  Pixel kNN (sessão #7):  45.76%")
    print(f"  Iter 1 STDP (sessão #9): 35.98% / 9.80%")
    print(f"  ProtoNet (sessão #7):   85.88%")
    print(f"\nLoss MSE final do AE: {losses[-1]:.5f} (epoch {len(losses)})")

    # Critério pré-definido
    acc5 = r5w["acc"] * 100
    print(f"\n=== Critério de decisão (acc 5w1s = {acc5:.2f}%) ===")
    if acc5 >= 65:
        print("  ✅ FORTE (≥65%): features aprendidas valem investimento. Próxima: C2.")
    elif acc5 >= 58:
        print("  ✓ MÉDIO (58-65%): C1d agrega marginalmente sobre C1b (56.28%). Decisão C2/C3 pra próxima sessão.")
    elif acc5 >= 54:
        print("  ≈ EMPATE (54-58%): PCA-32 é fronteira realista sem meta-objetivo. Próxima precisa de C2 ou C3.")
    else:
        print(f"  ⚠️ PIOR (<54%): considera bottleneck 64D. Se ainda <54%, descarta C1d.")


if __name__ == "__main__":
    main()
