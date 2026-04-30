"""
Possibilidade B — Camada única com regra híbrida (STDP-like trace + termos modulatórios).

Status: IMPLEMENTADO sessão #27 (sanity). Próxima sessão #28 roda 5 seeds
completos.

Regra de plasticidade unificada por peso:

    Δw_ij = η · (A_ij · pre_j · post_i · trace_j + B_ij · pre_j + C_ij · post_i + D_ij)

Onde:
  - pre_j: ativação pré-sináptica média (batch médio sobre os 5 support samples)
  - post_i: ativação pós-sináptica média
  - trace_j: STDP-like exponential trace de pre-spikes ao longo do inner loop:
            trace[t] = decay · trace[t-1] + pre_avg[t]
  - A, B, C, D: meta-parâmetros (1 por peso) treinados via outer loop
  - decay: meta-parâmetro escalar global (controla janela temporal do trace)
  - η: learning rate da plasticidade (fixo)

Hipótese central:
  Em sessão #18 (one-shot Omniglot), termo Hebbian puro `A·pre·post` era
  dispensável. SEM trace temporal, A só replica B·pre + C·post (correlação
  estática). Em CONTINUAL learning, trace introduz dependência temporal:
  Δw depende de QUANDO pre disparou em relação a post atual. Isso pode
  tornar o termo A não-trivial.

Setup:
  - Encoder linear: 784 → 128 → 32 (sem tanh — c2_simplified mostrou que
    linear basta; trace adiciona não-linearidade temporal)
  - Pesos iniciais zero (c2_simplified mostrou irrelevância)
  - Inner loop: n_inner=10 passes sobre support
  - Outer loop: cross-entropy do query, Adam lr=1e-3 nos meta-params
  - Continual: 50 alphabet tasks sequencial, sem replay, skip warmup
  - Classificador: prototype-based (cosine sim, β=8) — mesmo de c2_meta_hebbian

Critério sanity (1 seed, n_inner=5, finetune-eps=30, eval-eps=15):
  ACC ~50-85%: loop OK, prosseguir pra 5 seeds completos
  ACC <40%: regime degenerado (trace mata gradient ou plasticidade)
  ACC >95%: bug (provavelmente acessando query labels indevidamente)

Critério de validação completa (5 seeds, defaults):
  vs baseline naive ProtoNet (sessão #25): ACC 80.65% / BWT -9.26
  Sucesso: ACC ≥ 84% E (ablação A=0 cai ≥3 p.p. — confirma A contribui)
  Mediano: ACC 81-84% E A contribuição < 3 p.p.
  Falha: ACC < 81% (não bate baseline)

Ablações (executar após 5 seeds completos):
  B1: A=0 fixo (treina só B,C,D + decay) — testa se trace+A contribui
  B2: trace=0 fixo (treina A·pre·post puro + B,C,D) — testa se trace é
      necessário pra A virar útil. Equivalente a c2_simplified em CL.
  B3: decay fixo em 0 (sem trace memory) — match c2_meta_hebbian em CL
  B4: decay fixo em 1 (trace cumulativo) — extremo oposto

Não modifica baseline_naive.py nem nenhum arquivo do experiment_01.

Uso:
    # Sanity (1 seed reduzido)
    python c2_continual_arch_b.py --device cuda --seeds 1 --finetune-episodes 30 --eval-episodes 15 --n-inner 5
    # Completo (5 seeds defaults)
    python c2_continual_arch_b.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Imports do experiment_02 (mesmo dir) e experiment_01
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment_01_oneshot"))

from baseline_naive import (
    CombinedOmniglot,
    build_tasks_by_alphabet,
    TaskEpisodeSampler,
    bootstrap_ci,
    TaskData,
)
from config import default_config


# ---------------------------------------------------------------------------
# Plasticidade híbrida com trace
# ---------------------------------------------------------------------------
def init_plasticity_b(in_dim: int, hidden: int, latent: int, device, seed: int):
    """Meta-parâmetros A, B, C, D por peso em cada layer + decay scalar global.

    Pesos iniciais (W1, W2): zero fixo (c2_simplified default).
    """
    g = torch.Generator(device=device).manual_seed(seed + 1)

    def _p(shape):
        return (torch.randn(shape, generator=g, device=device) * 0.01).requires_grad_(True)

    shape1 = (hidden, in_dim)
    shape2 = (latent, hidden)
    params = [_p(shape1) for _ in range(4)] + [_p(shape2) for _ in range(4)]
    # decay scalar (sigmoid-parameterized pra ficar em [0,1])
    decay_logit = torch.zeros(1, device=device, requires_grad=True)  # sigmoid(0) = 0.5
    return params, decay_logit


def init_weights_zero(in_dim, hidden, latent, device):
    return (
        torch.zeros(hidden, in_dim, device=device),
        torch.zeros(latent, hidden, device=device),
    )


def hebbian_with_trace_update(W, pre_history, post, A, B, C, D, decay_scalar, eta):
    """Update híbrido com trace.

    pre_history: lista de pre_avg de inner steps anteriores (incluindo atual).
                 trace = sum(decay^k * pre_history[-1-k] for k=0..len-1)
    post: post_avg atual (out, 1)
    Outros: meta-params (out, in)

    Δw = η · (A · pre_current · post · trace + B · pre_current + C · post + D)
    """
    # Trace: combinação ponderada exponencial dos pre-pasados
    trace = torch.zeros_like(pre_history[-1])  # (1, in)
    for k, pre_k in enumerate(reversed(pre_history)):
        trace = trace + (decay_scalar ** k) * pre_k

    pre_current = pre_history[-1]  # (1, in)
    # Hebbian com trace: A * pre_current * post * trace
    hebb_with_trace = (post * pre_current) * trace  # (out, in) via broadcasting
    # Termos modulatórios sem trace
    pre_term = pre_current  # (1, in) broadcast pra (out, in)
    post_term = post  # (out, 1) broadcast pra (out, in)

    dW = eta * (A * hebb_with_trace + B * pre_term + C * post_term + D)
    return W + dW


def forward_linear(x, W1, W2):
    h = x @ W1.T
    z = h @ W2.T
    return h, z


def adapt_b(W1, W2, support, plasticity, decay_logit, n_inner: int, eta: float):
    """Inner loop com trace per layer."""
    A1, B1, C1, D1, A2, B2, C2, D2 = plasticity
    decay = torch.sigmoid(decay_logit)  # ∈ (0, 1)

    pre1_history = []
    pre2_history = []

    for inner_step in range(n_inner):
        # Layer 1 forward
        pre1_avg = support.mean(dim=0, keepdim=True)  # (1, in_dim)
        h1 = support @ W1.T  # (B, hidden)
        post1_avg = h1.mean(dim=0).unsqueeze(1)  # (hidden, 1)

        pre1_history.append(pre1_avg)
        # Atualiza W1 com trace
        W1 = hebbian_with_trace_update(W1, pre1_history, post1_avg,
                                        A1, B1, C1, D1, decay, eta)

        # Layer 2 forward (recompute h1 com W1 atualizado)
        h1_new = support @ W1.T
        pre2_avg = h1_new.mean(dim=0, keepdim=True)  # (1, hidden)
        z = h1_new @ W2.T  # (B, latent)
        post2_avg = z.mean(dim=0).unsqueeze(1)  # (latent, 1)

        pre2_history.append(pre2_avg)
        W2 = hebbian_with_trace_update(W2, pre2_history, post2_avg,
                                        A2, B2, C2, D2, decay, eta)

    return W1, W2


def episode_pass_b(W1, W2, plasticity, decay_logit, support, support_labels,
                   query, query_labels, n_classes: int, n_inner: int, eta: float, beta: float):
    """Adapt + classify via prototypes (cosine)."""
    W1_a, W2_a = adapt_b(W1, W2, support, plasticity, decay_logit, n_inner, eta)
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
    return loss, acc


def evaluate_task_b(W1, W2, plasticity, decay_logit, dataset, task: TaskData, device,
                    n_episodes: int, n_way: int, k_shot: int, n_query: int,
                    n_inner: int, eta: float, beta: float, seed: int) -> float:
    sampler = TaskEpisodeSampler(dataset, task.classes, task.test_indices_by_class,
                                  n_way=n_way, k_shot=k_shot, n_query=n_query, seed=seed)
    accs = []
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = sampler.sample()
            support = ep["support"].to(device).flatten(start_dim=1)
            sl = ep["support_labels"].to(device)
            query = ep["query"].to(device).flatten(start_dim=1)
            ql = ep["query_labels"].to(device)
            _, acc = episode_pass_b(W1, W2, plasticity, decay_logit, support, sl, query, ql,
                                    n_classes=n_way, n_inner=n_inner, eta=eta, beta=beta)
            accs.append(acc)
    return float(np.mean(accs))


# ---------------------------------------------------------------------------
# Run one seed
# ---------------------------------------------------------------------------
def run_one_seed(args, seed: int, dataset, device) -> dict:
    print(f"\n{'='*72}\n[seed={seed}] Possibilidade B (trace híbrido)\n{'='*72}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    # Build tasks (alphabet mode)
    tasks = build_tasks_by_alphabet(dataset, n_chars_per_task=args.n_chars_per_task,
                                     n_train_per_class=args.n_train_per_class,
                                     seed=seed, verbose=(seed == args.seed_start))
    if args.n_tasks < len(tasks):
        tasks = tasks[:args.n_tasks]
    n_actual = len(tasks)
    print(f"  [seed={seed}] {n_actual} tasks")

    # Init weights and plasticity
    in_dim = 28 * 28
    W1, W2 = init_weights_zero(in_dim, args.hidden, args.latent, device)
    plasticity, decay_logit = init_plasticity_b(in_dim, args.hidden, args.latent, device, seed)
    n_plast = sum(p.numel() for p in plasticity) + 1  # +decay
    print(f"  [seed={seed}] meta-params treináveis: {n_plast}")

    optim = torch.optim.Adam(plasticity + [decay_logit], lr=args.lr)

    acc_just_after = [0.0] * n_actual
    t0 = time.time()

    # Sequential fine-tune (skip warmup é default)
    for t in range(n_actual):
        train_sampler = TaskEpisodeSampler(
            dataset, tasks[t].classes, tasks[t].train_indices_by_class,
            n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query,
            seed=seed * 1000 + 200 + t,
        )
        for step in range(args.finetune_episodes):
            ep = train_sampler.sample()
            support = ep["support"].to(device).flatten(start_dim=1)
            sl = ep["support_labels"].to(device)
            query = ep["query"].to(device).flatten(start_dim=1)
            ql = ep["query_labels"].to(device)
            loss, _ = episode_pass_b(W1, W2, plasticity, decay_logit, support, sl, query, ql,
                                     n_classes=args.n_way, n_inner=args.n_inner,
                                     eta=args.eta, beta=args.beta)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(plasticity + [decay_logit], max_norm=1.0)
            optim.step()

        # Acc just_after pra task t
        acc_just_after[t] = evaluate_task_b(W1, W2, plasticity, decay_logit, dataset,
                                            tasks[t], device, args.eval_episodes,
                                            args.n_way, args.k_shot, args.n_query,
                                            args.n_inner, args.eta, args.beta,
                                            seed * 1000 + 100 + t)

        if (t + 1) % 10 == 0:
            elapsed = time.time() - t0
            decay_val = torch.sigmoid(decay_logit).item()
            print(f"  [seed={seed}] após task {t+1}/{n_actual}, just_after={acc_just_after[t]*100:.1f}%, "
                  f"decay={decay_val:.3f}, elapsed={elapsed:.1f}s")

    # Final eval em todas tasks
    print(f"  [seed={seed}] final eval em todas as {n_actual} tasks...")
    acc_final = [0.0] * n_actual
    for t in range(n_actual):
        acc_final[t] = evaluate_task_b(W1, W2, plasticity, decay_logit, dataset,
                                       tasks[t], device, args.eval_episodes,
                                       args.n_way, args.k_shot, args.n_query,
                                       args.n_inner, args.eta, args.beta,
                                       seed * 1000 + 500 + t)

    ACC = float(np.mean(acc_final))
    BWT = float(np.mean([acc_final[t] - acc_just_after[t] for t in range(n_actual)]))
    elapsed = time.time() - t0
    decay_val = torch.sigmoid(decay_logit).item()
    print(f"  [seed={seed}] FINAL: ACC={ACC*100:.2f}%  BWT={BWT*100:+.2f} p.p.  "
          f"decay={decay_val:.3f}  ({elapsed:.1f}s)")
    return {"seed": seed, "ACC": ACC, "BWT": BWT, "decay": decay_val,
            "acc_final": acc_final, "acc_just_after": acc_just_after, "elapsed": elapsed}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--n-tasks", type=int, default=50)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=1)
    p.add_argument("--n-query", type=int, default=5)
    p.add_argument("--n-chars-per-task", type=int, default=14)
    p.add_argument("--n-train-per-class", type=int, default=14)
    p.add_argument("--finetune-episodes", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--n-inner", type=int, default=10)
    p.add_argument("--eta", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=8.0)
    args = p.parse_args()

    cfg = default_config()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Dispositivo: {device}, seed_start={args.seed_start}, seeds={args.seeds}")
    print(f"Config: 50 alfabetos, n_inner={args.n_inner}, eta={args.eta}, lr={args.lr}, "
          f"finetune={args.finetune_episodes} eps/task, eval={args.eval_episodes}")

    print("\nCarregando Combined Omniglot...")
    dataset = CombinedOmniglot(cfg)
    print(f"  total chars: {len(dataset._characters)}")

    sess_t0 = time.time()
    results = []
    for s in range(args.seeds):
        seed = args.seed_start + s
        r = run_one_seed(args, seed, dataset, device)
        results.append(r)

    total = time.time() - sess_t0
    print(f"\n\nTempo total: {total:.1f}s ({total/60:.1f} min) pra {args.seeds} seeds")

    accs = np.array([r["ACC"] for r in results])
    bwts = np.array([r["BWT"] for r in results])
    decays = np.array([r["decay"] for r in results])

    print(f"\n{'='*72}")
    print(f"POSSIBILIDADE B — regra híbrida com trace")
    print(f"{'='*72}")
    accs_str = [f"{r['ACC']*100:.2f}%" for r in results]
    bwts_str = [f"{r['BWT']*100:+.2f}" for r in results]
    decays_str = [f"{r['decay']:.3f}" for r in results]
    print(f"  Seeds: {[r['seed'] for r in results]}")
    print(f"  ACC final: {accs_str}")
    print(f"  BWT:       {bwts_str}")
    print(f"  decay:     {decays_str}")

    if len(results) > 1:
        acc_lo, acc_hi = bootstrap_ci(accs)
        bwt_lo, bwt_hi = bootstrap_ci(bwts)
        print(f"\n  ACC média: {accs.mean()*100:.2f}%  IC95% [{acc_lo*100:.2f}, {acc_hi*100:.2f}]")
        print(f"  BWT média: {bwts.mean()*100:+.2f} p.p.  IC95% [{bwt_lo*100:+.2f}, {bwt_hi*100:+.2f}]")
    else:
        print(f"\n  ACC: {accs.mean()*100:.2f}%  BWT: {bwts.mean()*100:+.2f} p.p.  (1 seed)")

    print(f"\n=== Comparação com baseline naive ProtoNet (sessão #25) ===")
    print(f"  Baseline naive: ACC 80.65%, BWT -9.26 p.p.")
    delta_acc = accs.mean()*100 - 80.65
    delta_bwt = bwts.mean()*100 - (-9.26)
    print(f"  Possib. B:      ACC {accs.mean()*100:.2f}%, BWT {bwts.mean()*100:+.2f} p.p.")
    print(f"  Δ vs baseline:  ACC {delta_acc:+.2f} p.p., BWT {delta_bwt:+.2f} p.p.")


if __name__ == "__main__":
    main()
