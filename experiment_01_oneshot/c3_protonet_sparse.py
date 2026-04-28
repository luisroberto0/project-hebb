"""
Sessão #20 — Caminho C3: ProtoNet com features esparsas (k-WTA).

Pré-condição: família C2 saturou em ~64% (sessões #17-#19). ProtoNet
baseline reproduzido nesta sessão = 94.55% (vs 85.88% anteriormente
registrado, que era smoke test 500 eps; aqui rodamos os 5000 eps reais).

Pergunta: k-WTA esparso no embedding final do ProtoNet preserva a alta
performance enquanto agrega princípio neuro-inspirado defensável?

3 níveis de esparsidade (k de 64 dimensões):
  C3a: k=32 (50% esparso)
  C3b: k=16 (75% esparso)
  C3c: k=8  (87.5% esparso)

Validação obrigatória: random encoder (sem treino) + k-WTA + ProtoNet
classifier — confirma se ganho vem do treino ou da estrutura k-WTA.

Uso:
    python c3_protonet_sparse.py --device cuda --train-episodes 5000 --eval-eps 1000
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import default_config
from data import load_evaluation, load_background, EpisodeSampler


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05):
    rng = np.random.default_rng(0)
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def kwta(z: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k Winner-Take-All: mantém top-k ativações por exemplo, zera o resto.

    z: (B, D). Como ProtoEncoder termina em MaxPool (output ≥ 0 via ReLU),
    fazemos top-k por valor direto (não abs).
    """
    if k >= z.shape[-1]:
        return z
    topk_vals, topk_idx = z.topk(k, dim=-1)
    mask = torch.zeros_like(z).scatter_(-1, topk_idx, 1.0)
    return z * mask


class ProtoEncoder(nn.Module):
    """CNN-4 padrão de Snell et al. — duplicado de baselines.py por isolamento."""

    def __init__(self):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
        self.net = nn.Sequential(
            block(1, 64), block(64, 64), block(64, 64), block(64, 64),
        )

    def forward(self, x):  # x: (B, 1, H, W)
        z = self.net(x)
        return z.flatten(1)


class ProtoEncoderSparse(nn.Module):
    """ProtoEncoder + k-WTA no embedding final."""

    def __init__(self, k: int):
        super().__init__()
        self.encoder = ProtoEncoder()
        self.k = k

    def forward(self, x):
        z = self.encoder(x)
        return kwta(z, self.k)


def proto_episode_loss(encoder: nn.Module, episode, n_classes: int, device: torch.device):
    sup = episode.support.unsqueeze(1).to(device)
    qry = episode.query.unsqueeze(1).to(device)
    sup_lbl = episode.support_labels.to(device)
    qry_lbl = episode.query_labels.to(device)

    sup_emb = encoder(sup)
    qry_emb = encoder(qry)

    protos = []
    for c in range(n_classes):
        mask = sup_lbl == c
        protos.append(sup_emb[mask].mean(dim=0))
    protos = torch.stack(protos)

    logits = -torch.cdist(qry_emb, protos).pow(2)
    loss = F.cross_entropy(logits, qry_lbl)
    acc = (logits.argmax(-1) == qry_lbl).float().mean().item()
    return loss, acc


def train_and_eval(name: str, encoder: nn.Module, cfg, device, args, skip_train: bool = False):
    """Treina (se skip_train=False) e avalia em 5w1s + 20w1s."""
    print(f"\n{'='*72}\n[{name}]\n{'='*72}")

    if not skip_train:
        train_ds = load_background(cfg)
        train_sampler = EpisodeSampler(train_ds, n_way=5, k_shot=1, n_query=5, seed=args.seed)
        opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        encoder.train()

        t0 = time.time()
        log_every = max(1, args.train_episodes // 10)
        for step in range(args.train_episodes):
            ep = train_sampler.sample()
            loss, acc = proto_episode_loss(encoder, ep, n_classes=5, device=device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % log_every == 0:
                elapsed = time.time() - t0
                print(f"  [{name}] step {step+1:5d}/{args.train_episodes}  loss={loss.item():.3f}  acc={acc*100:.1f}%  ({elapsed:.1f}s)")
            if time.time() - t0 > args.watchdog:
                print(f"  [{name}] ⚠️ watchdog {args.watchdog}s — parando em step {step+1}")
                break

        train_time = time.time() - t0
        print(f"  [{name}] treino concluído em {train_time:.1f}s")
    else:
        train_time = 0.0
        print(f"  [{name}] SEM TREINO (encoder random + classificador)")

    eval_ds = load_evaluation(cfg)
    encoder.eval()
    results = {}
    for n_way in (5, 20):
        sampler = EpisodeSampler(eval_ds, n_way=n_way, k_shot=1, n_query=5, seed=args.seed + 1)
        accs = []
        t_eval = time.time()
        with torch.no_grad():
            for _ in range(args.eval_eps):
                ep = sampler.sample()
                _, acc = proto_episode_loss(encoder, ep, n_classes=n_way, device=device)
                accs.append(acc)
        arr = np.array(accs)
        mean = arr.mean()
        lo, hi = bootstrap_ci(arr)
        chance = 1.0 / n_way
        z = (mean - chance) / arr.std() if arr.std() > 0 else float("inf")
        results[f"{n_way}w"] = {"acc": mean, "lo": lo, "hi": hi, "z": z, "time": time.time()-t_eval}
        print(f"  [{name}] EVAL {n_way}w1s: {mean*100:.2f}% IC[{lo*100:.2f}, {hi*100:.2f}] z≈{z:.1f}")

    return {"name": name, "train_time": train_time, **results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--train-episodes", type=int, default=5000)
    p.add_argument("--eval-eps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--watchdog", type=int, default=900,
                   help="Segundos antes de matar treino individual")
    args = p.parse_args()

    cfg = default_config()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Dispositivo: {device}, seed={args.seed}, train_eps={args.train_episodes}, eval_eps={args.eval_eps}")

    torch.manual_seed(args.seed)

    sess_t0 = time.time()
    results = {}

    # C3a: k=32 (50% esparso)
    enc_a = ProtoEncoderSparse(k=32).to(device)
    results["C3a (k=32, 50%)"] = train_and_eval("C3a k=32", enc_a, cfg, device, args)

    # C3b: k=16 (75% esparso)
    torch.manual_seed(args.seed)
    enc_b = ProtoEncoderSparse(k=16).to(device)
    results["C3b (k=16, 75%)"] = train_and_eval("C3b k=16", enc_b, cfg, device, args)

    # C3c: k=8 (87.5% esparso)
    torch.manual_seed(args.seed)
    enc_c = ProtoEncoderSparse(k=8).to(device)
    results["C3c (k=8, 87.5%)"] = train_and_eval("C3c k=8", enc_c, cfg, device, args)

    # Validação obrigatória: random encoder + k-WTA + ProtoNet classifier
    # Usa o mesmo k=16 (ponto médio) com encoder NÃO treinado
    print(f"\n{'='*72}")
    print("VALIDAÇÃO OBRIGATÓRIA — random encoder + k-WTA (sem treino)")
    print(f"{'='*72}")
    torch.manual_seed(args.seed)
    enc_random = ProtoEncoderSparse(k=16).to(device)
    results["RandomEnc + k-WTA k=16"] = train_and_eval("Random+kWTA", enc_random, cfg, device, args, skip_train=True)

    total = time.time() - sess_t0
    print(f"\n\nTempo total da sessão: {total:.1f}s ({total/60:.1f} min)")

    # Tabela final
    print(f"\n{'='*84}")
    print("TABELA FAMÍLIA C COMPLETA + ProtoNet baselines")
    print(f"{'='*84}")
    print(f"{'Modelo':30s} | {'5w1s':>10s} | {'IC95% 5w1s':>16s} | {'20w1s':>10s} | {'sparsity':>10s}")
    print("-" * 90)
    print(f"{'C1b PCA-32 (#15)':30s} | {'56.28%':>10s} | {'[55.50, 57.11]':>16s} | {'35.37%':>10s} | {'-':>10s}")
    print(f"{'C2 baseline (#17)':30s} | {'63.22%':>10s} | {'[62.41, 64.06]':>16s} | {'37.30%':>10s} | {'-':>10s}")
    print(f"{'C2-simplified (#19)':30s} | {'64.08%':>10s} | {'[63.24, 64.92]':>16s} | {'-':>10s} | {'-':>10s}")
    print(f"{'ProtoNet baseline (#20)':30s} | {'94.55%':>10s} | (este run)         | {'-':>10s} | {'0%':>10s}")
    for name, r in results.items():
        if "RandomEnc" in name:
            continue
        sparsity = name.split('(')[1].split(')')[0] if '(' in name else "?"
        # Pega só o % de sparsity
        sparsity = sparsity.split(', ')[1] if ', ' in sparsity else sparsity
        print(f"{name:30s} | {r['5w']['acc']*100:>9.2f}% | [{r['5w']['lo']*100:>5.2f}, {r['5w']['hi']*100:>5.2f}] | {r['20w']['acc']*100:>9.2f}% | {sparsity:>10s}")
    rnd = results["RandomEnc + k-WTA k=16"]
    print("-" * 90)
    print(f"{'RandomEnc + k-WTA (k=16)':30s} | {rnd['5w']['acc']*100:>9.2f}% | [{rnd['5w']['lo']*100:>5.2f}, {rnd['5w']['hi']*100:>5.2f}] | {rnd['20w']['acc']*100:>9.2f}% | {'(validação)':>10s}")
    print("-" * 90)

    # Decisão pelo critério
    best_acc = max(results[k]["5w"]["acc"] for k in results if "RandomEnc" not in k) * 100
    print(f"\n=== Critério de decisão (melhor C3 5w1s = {best_acc:.2f}%) ===")
    if best_acc >= 80:
        print("  ✅ FORTE (≥80%): k-WTA preserva ProtoNet. Sparsity compatível com high performance.")
    elif best_acc >= 60:
        print(f"  ✓ TRADE-OFF (60-80%): caracteriza curva sparsity × acurácia.")
    else:
        print(f"  ⚠️ COLAPSO (<60%): k-WTA quebra ProtoNet. Esparsidade incompatível com prototype-based.")

    rnd_acc = rnd["5w"]["acc"] * 100
    print(f"\n=== Validação: random encoder + k-WTA = {rnd_acc:.2f}% ===")
    if rnd_acc < 45:
        print(f"  → 35-45%: ganho de C3 vem do TREINO, não da estrutura k-WTA.")
    elif rnd_acc < 55:
        print(f"  → 45-55%: parte arquitetural moderada.")
    else:
        print(f"  → >55%: parte do sinal é arquitetural, ajusta interpretação.")


if __name__ == "__main__":
    main()
