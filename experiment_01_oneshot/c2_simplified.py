"""
Sessão #19 — C2-simplified: combinação dos achados das ablações da sessão #18.

Achados pós-#18 que motivam esta variação:
  A2: W iniciais zero (= baseline) → init é dispensável
  A4: encoder linear (= baseline) → tanh é dispensável
  A3: inner_loop=1 perde 10 p.p. → profundidade importa, vamos testar n_inner=10

C2-simplified combina:
  - Encoder linear (sem tanh)
  - W iniciais zero (em vez de random std=0.1)
  - n_inner=10 (em vez de 5 do baseline)

Plus validação A1-invertida:
  - Mesmas modificações + A fixado em zero (treina só B, C, D)
  - Confirma que termo Hebbian A·pre·post é dispensável

Cada variação: 5000 eps meta-train + eval 1000 eps 5w1s seed=42.

Uso:
    python c2_simplified.py --device cuda --meta-train-eps 5000
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


def init_weights_zero(in_dim, hidden, latent, device):
    W1 = torch.zeros(hidden, in_dim, device=device)
    W2 = torch.zeros(latent, hidden, device=device)
    return W1, W2


def init_plasticity(shape1, shape2, device, seed, no_A: bool = False):
    """Inicializa A,B,C,D. Se no_A, A fica em zero fixo (não treinável)."""
    g = torch.Generator(device=device).manual_seed(seed + 1)
    def _p(shape):
        return (torch.randn(shape, generator=g, device=device) * 0.01).requires_grad_(True)

    if no_A:
        A1 = torch.zeros(shape1, device=device)
        A2 = torch.zeros(shape2, device=device)
        B1, C1, D1 = _p(shape1), _p(shape1), _p(shape1)
        B2, C2, D2 = _p(shape2), _p(shape2), _p(shape2)
        params = [A1, B1, C1, D1, A2, B2, C2, D2]
        trainable = [B1, C1, D1, B2, C2, D2]
    else:
        params = [_p(shape1) for _ in range(4)] + [_p(shape2) for _ in range(4)]
        trainable = params
    return params, trainable


def forward_linear(x, W1, W2):
    h = x @ W1.T
    z = h @ W2.T
    return h, z


def hebbian_update(W, pre, post, A, B, C, D, eta):
    Bsz = pre.shape[0]
    hebb = (post.t() @ pre) / Bsz
    pre_avg = pre.mean(dim=0, keepdim=True)
    post_avg = post.mean(dim=0).unsqueeze(1)
    return W + eta * (A * hebb + B * pre_avg + C * post_avg + D)


def adapt(W1, W2, support, plasticity, n_inner, eta):
    A1, B1, C1, D1, A2, B2, C2, D2 = plasticity
    for _ in range(n_inner):
        h1 = support @ W1.T
        W1 = hebbian_update(W1, support, h1, A1, B1, C1, D1, eta)
        h1_new = support @ W1.T
        z = h1_new @ W2.T
        W2 = hebbian_update(W2, h1_new, z, A2, B2, C2, D2, eta)
    return W1, W2


def episode_pass(W1, W2, plasticity, support, support_labels, query, query_labels,
                 n_classes, n_inner, eta, beta):
    W1_a, W2_a = adapt(W1, W2, support, plasticity, n_inner, eta)
    _, z_sup = forward_linear(support, W1_a, W2_a)
    _, z_qry = forward_linear(query, W1_a, W2_a)
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


def run_variation(name, cfg, device, seed, n_eps_train, n_eps_eval,
                  hidden, latent, n_inner, eta, lr, beta, no_A=False, watchdog=600):
    print(f"\n{'='*72}\n[{name}] linear+zero_init, n_inner={n_inner}, no_A={no_A}\n{'='*72}")
    torch.manual_seed(seed)

    W1, W2 = init_weights_zero(28*28, hidden, latent, device)
    plasticity, trainable = init_plasticity(W1.shape, W2.shape, device, seed, no_A=no_A)
    n_train = sum(p.numel() for p in trainable)
    print(f"  Pesos iniciais: zero ({W1.numel()+W2.numel()} valores)")
    print(f"  Plasticidade treinável: {n_train}")

    optim = torch.optim.Adam(trainable, lr=lr)

    bg = load_background(cfg)
    sampler = EpisodeSampler(bg, n_way=5, k_shot=1, n_query=5, seed=seed)

    losses, accs_train = [], []
    t0 = time.time()
    log_every = max(1, n_eps_train // 10)
    aborted = False
    for ep in range(n_eps_train):
        episode = sampler.sample()
        support = episode.support.to(device).flatten(start_dim=1)
        sl = episode.support_labels.to(device)
        query = episode.query.to(device).flatten(start_dim=1)
        ql = episode.query_labels.to(device)

        loss, acc, _ = episode_pass(W1, W2, plasticity, support, sl, query, ql,
                                    n_classes=5, n_inner=n_inner, eta=eta, beta=beta)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optim.step()
        losses.append(loss.item()); accs_train.append(acc)

        if (ep + 1) % log_every == 0:
            elapsed = time.time() - t0
            ra = np.mean(accs_train[-log_every:]) * 100
            rl = np.mean(losses[-log_every:])
            print(f"  [{name}] ep {ep+1:5d}  loss={rl:.4f}  acc={ra:.2f}%  ({elapsed:.1f}s)")
        if time.time() - t0 > watchdog:
            print(f"  [{name}] ⚠️ watchdog {watchdog}s — parando em ep {ep+1}")
            aborted = True
            break

    train_time = time.time() - t0
    final_train = float(np.mean(accs_train[-log_every:])) * 100
    print(f"  [{name}] meta-train concluído em {train_time:.1f}s, último bloco acc={final_train:.2f}%")

    eval_dataset = load_evaluation(cfg)
    eval_sampler = EpisodeSampler(eval_dataset, n_way=5, k_shot=1, n_query=5, seed=seed)
    eval_accs = []
    t_eval = time.time()
    with torch.no_grad():
        for ep in range(n_eps_eval):
            episode = eval_sampler.sample()
            support = episode.support.to(device).flatten(start_dim=1)
            sl = episode.support_labels.to(device)
            query = episode.query.to(device).flatten(start_dim=1)
            ql = episode.query_labels.to(device)
            _, acc, _ = episode_pass(W1, W2, plasticity, support, sl, query, ql,
                                     n_classes=5, n_inner=n_inner, eta=eta, beta=beta)
            eval_accs.append(acc)

    arr = np.array(eval_accs)
    mean = arr.mean()
    lo, hi = bootstrap_ci(arr)
    z = (mean - 0.2) / arr.std() if arr.std() > 0 else float("inf")
    print(f"  [{name}] EVAL 5w1s: {mean*100:.2f}% IC[{lo*100:.2f}, {hi*100:.2f}] z≈{z:.1f} ({time.time()-t_eval:.1f}s)")

    return {"name": name, "acc": mean, "lo": lo, "hi": hi, "z": z,
            "final_train_acc": final_train, "train_time": train_time,
            "n_train_params": n_train, "aborted": aborted}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--meta-train-eps", type=int, default=5000)
    p.add_argument("--eval-eps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--watchdog", type=int, default=600)
    args = p.parse_args()

    cfg = default_config()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Dispositivo: {device}, seed={args.seed}, meta_train_eps={args.meta_train_eps}")

    common = dict(
        cfg=cfg, device=device, seed=args.seed,
        n_eps_train=args.meta_train_eps, n_eps_eval=args.eval_eps,
        hidden=128, latent=32, n_inner=10, eta=0.01, lr=1e-3, beta=8.0,
        watchdog=args.watchdog,
    )

    results = {}
    sess_t0 = time.time()

    # C2-simplified: linear + W=0 + n_inner=10, com todos A,B,C,D treináveis
    results["C2-simplified"] = run_variation("C2-simplified", **common)

    # C2-simplified + no-A: confirma que A é dispensável
    results["C2-simplified-no-A"] = run_variation("C2-simplified-no-A", **{**common, "no_A": True})

    total = time.time() - sess_t0
    print(f"\n\nTempo total: {total:.1f}s ({total/60:.1f} min)")

    # Tabela final
    print(f"\n{'='*84}")
    print("TABELA C2-simplified — sessão #19")
    print(f"{'='*84}")
    print(f"{'Variação':30s} | {'5w1s':>10s} | {'IC95%':>16s} | {'z':>5s} | {'train acc':>10s} | {'time':>6s}")
    print("-" * 88)
    print(f"{'C2 baseline (#17, ref)':30s} | {'63.22%':>10s} | {'[62.41, 64.06]':>16s} | {'3.3':>5s} | {'69.62%':>10s} | {'132s':>6s}")
    print(f"{'C2 ablações:':30s} | {'':>10s} | {'':>16s} | {'':>5s} | {'':>10s} | {'':>6s}")
    print(f"{'  A2 W init=0 (#18)':30s} | {'63.97%':>10s} | {'[63.13, 64.80]':>16s} | {'3.3':>5s} | {'70.88%':>10s} | {'96s':>6s}")
    print(f"{'  A3 inner=1 (#18)':30s} | {'53.05%':>10s} | {'[52.25, 53.86]':>16s} | {'2.5':>5s} | {'55.73%':>10s} | {'73s':>6s}")
    print(f"{'  A4 linear (#18)':30s} | {'64.07%':>10s} | {'[63.27, 64.86]':>16s} | {'3.4':>5s} | {'69.32%':>10s} | {'93s':>6s}")
    for name, r in results.items():
        marker = " ⚠️" if r["aborted"] else ""
        print(f"{name+marker:30s} | {r['acc']*100:>9.2f}% | [{r['lo']*100:>5.2f}, {r['hi']*100:>5.2f}] | {r['z']:>5.1f} | {r['final_train_acc']:>9.2f}% | {r['train_time']:>5.1f}s")
    print("-" * 88)

    # Critério
    acc_simp = results["C2-simplified"]["acc"] * 100
    print(f"\n=== Critério de decisão (C2-simplified 5w1s = {acc_simp:.2f}%) ===")
    if acc_simp >= 65:
        print("  ✅ ≥65%: combinação destrava ganho. Próxima sessão: C2-with-Hopfield.")
    elif acc_simp >= 60:
        print("  ✓ 60-65%: saturação aparente. Próxima sessão: pivot pra C3.")
    else:
        print(f"  ⚠️ <60%: simplificação NÃO compensou. n_inner=10 não recuperou perdas combinadas.")

    # Comparação no-A
    acc_noA = results["C2-simplified-no-A"]["acc"] * 100
    delta_noA = acc_noA - acc_simp
    print(f"\n=== Validação A1-invertida (sem termo Hebbian A) ===")
    print(f"  C2-simplified:           {acc_simp:.2f}%")
    print(f"  C2-simplified-no-A:      {acc_noA:.2f}%")
    print(f"  Delta (remover A):       {delta_noA:+.2f} p.p.")
    if abs(delta_noA) < 2:
        print(f"  → A é dispensável (delta dentro do ruído). Confirma A1 invertida.")
    elif delta_noA < 0:
        print(f"  → A contribui mesmo que pequeno; remover prejudica.")
    else:
        print(f"  → Remover A *melhora* — A introduzia ruído.")


if __name__ == "__main__":
    main()
