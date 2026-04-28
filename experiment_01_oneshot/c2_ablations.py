"""
Sessão #18 — Ablações sobre C2 meta-Hebbian.

Pré-condição: C2 baseline (sessão #17) = 63.22% 5w1s. Este script isola
qual componente da regra de plasticidade carrega o sinal.

4 ablações (cada uma re-roda meta-train + eval com mesmo seed=42):
  A1: só termo Hebbian (ΔW = η·A·pre·post, sem B, C, D)
  A2: pesos iniciais zero em vez de random
  A3: inner_loop=1 em vez de 5
  A4: encoder linear (sem tanh)

Não modifica c2_meta_hebbian.py (restrição protocolo). Reusa lógica
duplicada aqui com flags de ablação. eval 5w1s 1000 eps seed=42.

Uso:
    python c2_ablations.py --device cuda --meta-train-eps 5000 --eval-eps 1000
    python c2_ablations.py --device cuda --meta-train-eps 3000  # se tempo apertar
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


def init_weights(in_dim, hidden, latent, device, seed, zero_init: bool = False):
    """Pesos iniciais. zero_init=True força tudo zero (A2)."""
    g = torch.Generator(device=device).manual_seed(seed)
    if zero_init:
        W1 = torch.zeros(hidden, in_dim, device=device)
        W2 = torch.zeros(latent, hidden, device=device)
    else:
        W1 = torch.randn(hidden, in_dim, generator=g, device=device) * 0.1
        W2 = torch.randn(latent, hidden, generator=g, device=device) * 0.1
    return W1, W2


def init_plasticity_params(shape1, shape2, device, seed, only_A: bool = False):
    """Plasticidade. only_A=True aloca só A (B=C=D=0 fixo, não treinados) (A1)."""
    g = torch.Generator(device=device).manual_seed(seed + 1)
    def _p(shape):
        return (torch.randn(shape, generator=g, device=device) * 0.01).requires_grad_(True)

    if only_A:
        # Só A é treinável; B, C, D são zero fixos (não-trainable)
        A1 = _p(shape1); A2 = _p(shape2)
        zeros1 = torch.zeros(shape1, device=device)
        zeros2 = torch.zeros(shape2, device=device)
        # Retorna no mesmo formato (8 tensors), mas B, C, D são zeros sem grad
        return [A1, zeros1, zeros1, zeros1, A2, zeros2, zeros2, zeros2], [A1, A2]
    else:
        params = [_p(shape1) for _ in range(4)] + [_p(shape2) for _ in range(4)]
        return params, params  # treináveis = todos


def forward_encode(x, W1, W2, linear: bool = False):
    """Forward 2-layer. linear=True remove tanh (A4)."""
    if linear:
        h = x @ W1.T
        z = h @ W2.T
    else:
        h = torch.tanh(x @ W1.T)
        z = torch.tanh(h @ W2.T)
    return h, z


def hebbian_update(W, pre, post, A, B, C, D, eta):
    Bsz = pre.shape[0]
    hebb = (post.t() @ pre) / Bsz
    pre_avg = pre.mean(dim=0, keepdim=True)
    post_avg = post.mean(dim=0).unsqueeze(1)
    dW = eta * (A * hebb + B * pre_avg + C * post_avg + D)
    return W + dW


def adapt(W1, W2, support, plasticity, n_inner: int, eta: float, linear: bool = False):
    A1, B1, C1, D1, A2, B2, C2, D2 = plasticity
    for _ in range(n_inner):
        h1, _ = forward_encode(support, W1, W2, linear=linear)  # só usamos h1 aqui
        W1 = hebbian_update(W1, support, h1, A1, B1, C1, D1, eta)
        # Re-forward com W1 atualizado pra computar z corretamente
        if linear:
            h1_new = support @ W1.T
            z = h1_new @ W2.T
        else:
            h1_new = torch.tanh(support @ W1.T)
            z = torch.tanh(h1_new @ W2.T)
        W2 = hebbian_update(W2, h1_new, z, A2, B2, C2, D2, eta)
    return W1, W2


def episode_pass(W1, W2, plasticity, support, support_labels, query, query_labels,
                 n_classes, n_inner, eta, beta, linear=False, skip_inner=False):
    if skip_inner:
        W1_a, W2_a = W1, W2
    else:
        W1_a, W2_a = adapt(W1, W2, support, plasticity, n_inner, eta, linear=linear)

    _, z_sup = forward_encode(support, W1_a, W2_a, linear=linear)
    _, z_qry = forward_encode(query, W1_a, W2_a, linear=linear)

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


def run_ablation(name: str, cfg, device, seed: int, n_eps_train: int, n_eps_eval: int,
                 hidden: int, latent: int, n_inner: int, eta: float, lr: float, beta: float,
                 only_A: bool = False, zero_init: bool = False, linear: bool = False,
                 watchdog_seconds: int = 600) -> dict:
    """Roda uma ablação completa. Retorna dict com resultados."""
    print(f"\n{'='*72}\n[{name}] only_A={only_A} zero_init={zero_init} linear={linear} n_inner={n_inner}")
    print(f"{'='*72}")
    torch.manual_seed(seed)

    W1, W2 = init_weights(28*28, hidden, latent, device, seed, zero_init=zero_init)
    plasticity, trainable = init_plasticity_params(W1.shape, W2.shape, device, seed, only_A=only_A)
    n_train_params = sum(p.numel() for p in trainable)
    print(f"  Pesos iniciais: {W1.numel() + W2.numel()} (zero_init={zero_init})")
    print(f"  Plasticidade treinável: {n_train_params}")

    optim = torch.optim.Adam(trainable, lr=lr)

    # Meta-train
    bg = load_background(cfg)
    train_sampler = EpisodeSampler(bg, n_way=5, k_shot=1, n_query=5, seed=seed)
    losses = []
    accs_train = []
    t0 = time.time()
    log_every = max(1, n_eps_train // 10)
    aborted = False
    for ep in range(n_eps_train):
        episode = train_sampler.sample()
        support = episode.support.to(device).flatten(start_dim=1)
        support_labels = episode.support_labels.to(device)
        query = episode.query.to(device).flatten(start_dim=1)
        query_labels = episode.query_labels.to(device)

        loss, acc, _ = episode_pass(W1, W2, plasticity, support, support_labels, query, query_labels,
                                    n_classes=5, n_inner=n_inner, eta=eta, beta=beta, linear=linear)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optim.step()
        losses.append(loss.item())
        accs_train.append(acc)

        if (ep + 1) % log_every == 0:
            elapsed = time.time() - t0
            recent_acc = np.mean(accs_train[-log_every:]) * 100
            recent_loss = np.mean(losses[-log_every:])
            print(f"  [{name}] ep {ep+1:4d}  loss={recent_loss:.4f}  acc={recent_acc:.2f}%  ({elapsed:.1f}s)")
        if time.time() - t0 > watchdog_seconds:
            print(f"  [{name}] ⚠️ watchdog {watchdog_seconds}s — parando em ep {ep+1}")
            aborted = True
            break

    train_time = time.time() - t0
    final_train_acc = float(np.mean(accs_train[-log_every:])) * 100
    print(f"  [{name}] meta-train concluído em {train_time:.1f}s, último bloco acc={final_train_acc:.2f}%")

    # Eval 5w1s
    eval_dataset = load_evaluation(cfg)
    sampler = EpisodeSampler(eval_dataset, n_way=5, k_shot=1, n_query=5, seed=seed)
    eval_accs = []
    t_eval = time.time()
    with torch.no_grad():
        for ep in range(n_eps_eval):
            episode = sampler.sample()
            support = episode.support.to(device).flatten(start_dim=1)
            support_labels = episode.support_labels.to(device)
            query = episode.query.to(device).flatten(start_dim=1)
            query_labels = episode.query_labels.to(device)
            _, acc, _ = episode_pass(W1, W2, plasticity, support, support_labels, query, query_labels,
                                     n_classes=5, n_inner=n_inner, eta=eta, beta=beta, linear=linear)
            eval_accs.append(acc)

    accs_arr = np.array(eval_accs)
    mean = accs_arr.mean()
    lo, hi = bootstrap_ci(accs_arr)
    z = (mean - 0.2) / accs_arr.std() if accs_arr.std() > 0 else float("inf")
    print(f"  [{name}] EVAL 5w1s: {mean*100:.2f}% IC[{lo*100:.2f}, {hi*100:.2f}] z≈{z:.1f} ({time.time()-t_eval:.1f}s)")

    return {
        "name": name,
        "acc": mean, "lo": lo, "hi": hi, "z": z,
        "final_train_acc": final_train_acc,
        "train_time": train_time,
        "n_train_params": n_train_params,
        "aborted": aborted,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--meta-train-eps", type=int, default=5000)
    p.add_argument("--eval-eps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--watchdog", type=int, default=600,
                   help="Segundos antes de matar uma ablação individual")
    args = p.parse_args()

    cfg = default_config()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Dispositivo: {device}, seed={args.seed}, meta_train_eps={args.meta_train_eps}")

    common = dict(
        cfg=cfg, device=device, seed=args.seed,
        n_eps_train=args.meta_train_eps, n_eps_eval=args.eval_eps,
        hidden=128, latent=32, n_inner=5, eta=0.01, lr=1e-3, beta=8.0,
        watchdog_seconds=args.watchdog,
    )

    results = {}
    sess_t0 = time.time()

    # A1: só termo Hebbian (B, C, D fixos em zero)
    results["A1 (só Hebb)"] = run_ablation("A1 só Hebb", **{**common, "only_A": True})

    # A2: pesos iniciais zero
    results["A2 (W init=0)"] = run_ablation("A2 W=0", **{**common, "zero_init": True})

    # A3: inner_loop=1
    results["A3 (n_inner=1)"] = run_ablation("A3 inner=1", **{**common, "n_inner": 1})

    # A4: encoder linear (sem tanh)
    results["A4 (linear)"] = run_ablation("A4 linear", **{**common, "linear": True})

    total_time = time.time() - sess_t0
    print(f"\n\nTempo total das 4 ablações: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Tabela final
    print(f"\n{'='*78}")
    print("TABELA ABLAÇÕES C2 — sessão #18")
    print(f"{'='*78}")
    print(f"{'Ablação':25s} | {'5w1s':>10s} | {'IC95%':>16s} | {'z':>5s} | {'train acc':>10s} | {'time':>6s}")
    print("-" * 84)
    print(f"{'C2 baseline (#17, ref)':25s} | {'63.22%':>10s} | {'[62.41, 64.06]':>16s} | {'3.3':>5s} | {'69.62%':>10s} | {'132s':>6s}")
    for name, r in results.items():
        marker = " ⚠️" if r["aborted"] else ""
        print(f"{name+marker:25s} | {r['acc']*100:>9.2f}% | [{r['lo']*100:>5.2f}, {r['hi']*100:>5.2f}] | {r['z']:>5.1f} | {r['final_train_acc']:>9.2f}% | {r['train_time']:>5.1f}s")
    print("-" * 84)
    print(f"\nReferências: C1b PCA-32 = 56.28%, C1c RandomProj-32 = 41.23%, ProtoNet = 85.88%")

    # Interpretação automática
    print(f"\n{'='*78}\nINTERPRETAÇÃO\n{'='*78}")
    base = 63.22
    for name, r in results.items():
        acc = r["acc"] * 100
        delta = acc - base
        if abs(delta) < 3:
            cat = "≈ baseline"
        elif delta < -8:
            cat = "queda forte"
        elif delta < 0:
            cat = "queda moderada"
        else:
            cat = "ganho"
        print(f"  {name}: {acc:.2f}% (delta {delta:+.2f} p.p. vs C2 baseline) — {cat}")


if __name__ == "__main__":
    main()
