"""
Sessão #17 — C2: meta-learning bio-inspirado, primeira iteração conservadora.

Inspiração: Najarro & Risi 2020 — redes random com plasticidade local Hebbian
aprendem few-shot via meta-objetivo. Aqui:
  - Encoder MLP: 784 → 128 → 32 (tanh)
  - Pesos iniciais (W1, W2) fixos random — NÃO meta-aprendidos
  - Plasticidade Hebbian local por peso: ΔW_ij = η × (A·pre_i·post_j + B·pre_i + C·post_j + D)
  - A, B, C, D são meta-aprendidos por peso (4 × N² params por layer)
  - Inner loop: aplica plasticidade n_inner=5 vezes sobre o support
  - Outer loop: cross-entropy do query, Adam lr=1e-3 nos params de plasticidade

C2 NÃO usa Hopfield — substitui por classificador prototípico direto
(prototypes per class no espaço latente adaptado, cosine similarity, β=8).

Validação obrigatória pós-treino: eval com inner loop desativado pra
confirmar que o sinal vem da plasticidade, não dos pesos iniciais.

Uso:
    python c2_meta_hebbian.py --device cuda --meta-train-eps 5000 --eval-eps 1000
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


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05):
    rng = np.random.default_rng(0)
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def init_random_weights(in_dim: int, hidden: int, latent: int, device, seed: int):
    """Pesos iniciais fixos (não meta-aprendidos). Std=0.1 pra tanh estável."""
    g = torch.Generator(device=device).manual_seed(seed)
    W1 = torch.randn(hidden, in_dim, generator=g, device=device) * 0.1
    W2 = torch.randn(latent, hidden, generator=g, device=device) * 0.1
    return W1, W2


def init_plasticity_params(shape1: tuple, shape2: tuple, device, seed: int):
    """4 parâmetros (A, B, C, D) por peso em cada layer. Meta-aprendidos."""
    g = torch.Generator(device=device).manual_seed(seed + 1)
    def _p(shape):
        # std pequena pra começar perto de "no plasticity" — meta-train aprende escala
        return (torch.randn(shape, generator=g, device=device) * 0.01).requires_grad_(True)
    return [_p(shape1) for _ in range(4)] + [_p(shape2) for _ in range(4)]


def forward_encode(x: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor):
    """Forward 2-layer tanh. x: (B, in_dim)."""
    h = torch.tanh(x @ W1.T)        # (B, hidden)
    z = torch.tanh(h @ W2.T)        # (B, latent)
    return h, z


def hebbian_update(W: torch.Tensor, pre: torch.Tensor, post: torch.Tensor,
                   A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                   eta: float) -> torch.Tensor:
    """
    ΔW_ij = η × (A_ij · mean(pre_j · post_i) + B_ij · mean(pre_j) + C_ij · mean(post_i) + D_ij)
    W: (out, in); pre: (Bsz, in); post: (Bsz, out); A,B,C,D: (out, in).
    """
    Bsz = pre.shape[0]
    hebb = (post.t() @ pre) / Bsz                    # (out, in)
    pre_avg = pre.mean(dim=0, keepdim=True)          # (1, in) — broadcast
    post_avg = post.mean(dim=0).unsqueeze(1)         # (out, 1) — broadcast
    dW = eta * (A * hebb + B * pre_avg + C * post_avg + D)
    return W + dW


def adapt(W1, W2, support, plasticity, n_inner: int, eta: float):
    """Inner loop: aplica plasticidade n_inner vezes sobre o support."""
    A1, B1, C1, D1, A2, B2, C2, D2 = plasticity
    for _ in range(n_inner):
        h1 = torch.tanh(support @ W1.T)
        W1 = hebbian_update(W1, support, h1, A1, B1, C1, D1, eta)
        h1_new = torch.tanh(support @ W1.T)          # re-forward com W1 atualizado
        z = torch.tanh(h1_new @ W2.T)
        W2 = hebbian_update(W2, h1_new, z, A2, B2, C2, D2, eta)
    return W1, W2


def episode_pass(W1, W2, plasticity, support, support_labels, query, query_labels,
                 n_classes: int, n_inner: int, eta: float, beta: float, skip_inner: bool = False):
    """
    Adapt W via plasticity (ou pula), encoda support+query, classifica via prototypes
    (cosine sim), retorna loss + acc + logits.
    """
    if skip_inner:
        W1_a, W2_a = W1, W2
    else:
        W1_a, W2_a = adapt(W1, W2, support, plasticity, n_inner, eta)

    _, z_sup = forward_encode(support, W1_a, W2_a)
    _, z_qry = forward_encode(query, W1_a, W2_a)

    # Prototypes per class (mean of K-shot embeddings)
    prototypes = torch.zeros(n_classes, z_sup.shape[1], device=z_sup.device)
    for c in range(n_classes):
        mask = (support_labels == c)
        prototypes[c] = z_sup[mask].mean(dim=0)

    z_qry_n = F.normalize(z_qry, dim=-1)
    proto_n = F.normalize(prototypes, dim=-1)
    logits = beta * z_qry_n @ proto_n.T
    loss = F.cross_entropy(logits, query_labels)
    preds = logits.argmax(dim=-1)
    acc = (preds == query_labels).float().mean().item()
    return loss, acc, logits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--meta-train-eps", type=int, default=5000)
    p.add_argument("--eval-eps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-inner", type=int, default=5)
    p.add_argument("--eta", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=8.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--latent", type=int, default=32)
    args = p.parse_args()

    cfg = default_config()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Dispositivo: {device}")
    torch.manual_seed(args.seed)

    # Pesos iniciais e parâmetros de plasticidade
    W1, W2 = init_random_weights(28*28, args.hidden, args.latent, device, args.seed)
    plasticity = init_plasticity_params(W1.shape, W2.shape, device, args.seed)
    n_init = W1.numel() + W2.numel()
    n_plast = sum(p.numel() for p in plasticity)
    print(f"Arquitetura: 784 → {args.hidden} → {args.latent} (tanh)")
    print(f"Pesos iniciais (FIXOS): {n_init}")
    print(f"Parâmetros plasticidade (meta-aprendidos): {n_plast}")
    print(f"Hyperparams: n_inner={args.n_inner}, eta={args.eta}, lr={args.lr}, beta={args.beta}")

    optim = torch.optim.Adam(plasticity, lr=args.lr)

    # Meta-train no background set
    print("\nCarregando background set...")
    bg = load_background(cfg)
    train_sampler = EpisodeSampler(bg, n_way=5, k_shot=1, n_query=5, seed=args.seed)

    print(f"\nMeta-train: {args.meta_train_eps} episodes")
    losses = []
    accs_train = []
    t0 = time.time()
    log_every = 250
    for ep in range(args.meta_train_eps):
        episode = train_sampler.sample()
        support = episode.support.to(device).flatten(start_dim=1)
        support_labels = episode.support_labels.to(device)
        query = episode.query.to(device).flatten(start_dim=1)
        query_labels = episode.query_labels.to(device)

        loss, acc, _ = episode_pass(W1, W2, plasticity, support, support_labels, query, query_labels,
                                    n_classes=5, n_inner=args.n_inner, eta=args.eta, beta=args.beta)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(plasticity, max_norm=1.0)  # estabilidade
        optim.step()

        losses.append(loss.item())
        accs_train.append(acc)

        if (ep + 1) % log_every == 0:
            elapsed = time.time() - t0
            recent_loss = np.mean(losses[-log_every:])
            recent_acc = np.mean(accs_train[-log_every:]) * 100
            print(f"  ep {ep+1:5d}/{args.meta_train_eps}  loss={recent_loss:.4f}  acc={recent_acc:.2f}%  elapsed={elapsed:.1f}s")

        if time.time() - t0 > 1800:  # 30 min watchdog
            print(f"  ⚠️ Meta-train passou de 30 min, parando em ep {ep+1}")
            break

    train_elapsed = time.time() - t0
    final_train_acc = float(np.mean(accs_train[-log_every:])) * 100
    print(f"\nMeta-train concluído em {train_elapsed:.1f}s. Último bloco {log_every} eps acc={final_train_acc:.2f}%")

    # Eval
    print("\nCarregando evaluation set...")
    eval_dataset = load_evaluation(cfg)

    def evaluate(n_way: int, n_eps: int, skip_inner: bool, label: str):
        sampler = EpisodeSampler(eval_dataset, n_way=n_way, k_shot=1, n_query=5, seed=args.seed)
        accs = []
        cos_centered_list = []
        t_eval = time.time()
        with torch.no_grad():
            for ep in range(n_eps):
                episode = sampler.sample()
                support = episode.support.to(device).flatten(start_dim=1)
                support_labels = episode.support_labels.to(device)
                query = episode.query.to(device).flatten(start_dim=1)
                query_labels = episode.query_labels.to(device)

                if skip_inner:
                    W1_a, W2_a = W1, W2
                else:
                    W1_a, W2_a = adapt(W1, W2, support, plasticity, args.n_inner, args.eta)

                _, z_sup = forward_encode(support, W1_a, W2_a)
                _, z_qry = forward_encode(query, W1_a, W2_a)

                prototypes = torch.zeros(n_way, z_sup.shape[1], device=device)
                for c in range(n_way):
                    mask = (support_labels == c)
                    prototypes[c] = z_sup[mask].mean(dim=0)
                z_qry_n = F.normalize(z_qry, dim=-1)
                proto_n = F.normalize(prototypes, dim=-1)
                logits = args.beta * z_qry_n @ proto_n.T
                preds = logits.argmax(dim=-1)
                accs.append((preds == query_labels).float().mean().item())

                # Centered cosine entre support embeddings (medida de diversidade)
                sup_centered = z_sup - z_sup.mean(dim=0, keepdim=True)
                sup_n = F.normalize(sup_centered, dim=-1)
                cos = sup_n @ sup_n.T
                n = cos.shape[0]
                mask_eye = ~torch.eye(n, dtype=torch.bool, device=device)
                cos_centered_list.append(cos[mask_eye].mean().item())

                if (ep + 1) % 250 == 0:
                    print(f"  [{label}] {n_way}w1s ep {ep+1:4d}  acc rolling={np.mean(accs)*100:.2f}%")

        accs_arr = np.array(accs)
        mean = accs_arr.mean()
        lo, hi = bootstrap_ci(accs_arr)
        chance = 1.0 / n_way
        z = (mean - chance) / accs_arr.std() if accs_arr.std() > 0 else float("inf")
        avg_cos = float(np.mean(cos_centered_list))
        print(f"  [{label}] {n_way}w1s: {mean*100:.2f}% IC[{lo*100:.2f},{hi*100:.2f}] z≈{z:.1f} cos_cent={avg_cos:.4f} ({time.time()-t_eval:.1f}s)")
        return {"acc": mean, "lo": lo, "hi": hi, "z": z, "cos_cent": avg_cos}

    print(f"\n{'='*72}\nEvaluação principal C2 (com inner loop)\n{'='*72}")
    r5w = evaluate(5, args.eval_eps, skip_inner=False, label="C2")
    r20w = evaluate(20, args.eval_eps, skip_inner=False, label="C2")

    print(f"\n{'='*72}\nValidação obrigatória — eval SEM inner loop (pesos iniciais)\n{'='*72}")
    r5w_no = evaluate(5, args.eval_eps, skip_inner=True, label="C2-no_inner")
    r20w_no = evaluate(20, args.eval_eps, skip_inner=True, label="C2-no_inner")

    # Tabela final
    print(f"\n{'='*72}")
    print("TABELA C2 vs família C1 (sessões #15, #16)")
    print(f"{'='*72}")
    print(f"{'Encoder':30s} | {'5w1s':>10s} | {'IC95% 5w1s':>16s} | {'20w1s':>10s} | {'cos cent':>10s}")
    print("-" * 90)
    print(f"{'C1a Pixels+L2 (#15)':30s} | {'50.17%':>10s} | {'[49.34, 50.96]':>16s} | {'30.30%':>10s} | {'-0.2485':>10s}")
    print(f"{'C1b PCA-32 (#15, ref)':30s} | {'56.28%':>10s} | {'[55.50, 57.11]':>16s} | {'35.37%':>10s} | {'-0.2476':>10s}")
    print(f"{'C1c RandomProj-32 (#15)':30s} | {'41.23%':>10s} | {'[40.51, 41.95]':>16s} | {'20.05%':>10s} | {'-0.2462':>10s}")
    print(f"{'C1d AE-32 (#16)':30s} | {'50.57%':>10s} | {'[49.76, 51.37]':>16s} | {'30.59%':>10s} | {'-0.2438':>10s}")
    print(f"{'C1d AE-64 (#16)':30s} | {'52.64%':>10s} | {'[51.77, 53.47]':>16s} | {'32.54%':>10s} | {'-0.2463':>10s}")
    print(f"{'C2 (com inner)':30s} | {r5w['acc']*100:>9.2f}% | [{r5w['lo']*100:>5.2f}, {r5w['hi']*100:>5.2f}] | {r20w['acc']*100:>9.2f}% | {r5w['cos_cent']:>10.4f}")
    print(f"{'C2 SEM inner (validação)':30s} | {r5w_no['acc']*100:>9.2f}% | [{r5w_no['lo']*100:>5.2f}, {r5w_no['hi']*100:>5.2f}] | {r20w_no['acc']*100:>9.2f}% | {r5w_no['cos_cent']:>10.4f}")
    print("-" * 90)

    # Critério
    acc5 = r5w["acc"] * 100
    print(f"\n=== Critério de decisão (acc 5w1s = {acc5:.2f}%) ===")
    if acc5 >= 70:
        print("  ✅ FORTE (≥70%): C2 valida hipótese central. Próximas refinam (mais inner steps, encoder maior).")
    elif acc5 >= 60:
        print("  ✓ MÉDIO (60-70%): agrega sobre C1b mas não fecha gap até ProtoNet (85.88%). Decidir continuar refinando ou C3.")
    elif acc5 >= 54:
        print("  ≈ EMPATE (54-58%): plasticidade meta-aprendida não agrega sobre PCA-32. Falha informativa.")
    else:
        print(f"  ⚠️ PIOR (<54%): algo errado na implementação. Investigar antes de aceitar.")

    # Diagnóstico via no-inner
    delta = (r5w["acc"] - r5w_no["acc"]) * 100
    print(f"\n=== Diagnóstico via validação sem inner loop ===")
    print(f"  C2 (com inner):    {r5w['acc']*100:.2f}%")
    print(f"  C2 (sem inner):    {r5w_no['acc']*100:.2f}%")
    print(f"  Delta (inner agrega): {delta:+.2f} p.p.")
    if delta > 5:
        print(f"  → Plasticidade contribui significativamente.")
    elif delta > 0:
        print(f"  → Plasticidade contribui marginalmente.")
    else:
        print(f"  → Plasticidade NÃO contribui ou regride. Encoder random sozinho carrega o sinal.")


if __name__ == "__main__":
    main()
