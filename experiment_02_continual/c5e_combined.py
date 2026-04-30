"""
Caminho 5e — Arquitetura combinada (kitchen sink): CNN-4 + plasticidade meta-aprendida
+ trace STDP-like + k-WTA esparso + continual sequencial sem replay.

Status: scaffold + sanity (sessão #28). 5 seeds completos pra sessão #29.

Arquitetura (Opção 2 — hybrid backprop + plasticidade):

  Image (1, 28, 28)
    → CNN-4 (4 blocos Conv-BN-ReLU-MaxPool, SGD via backprop) → 64D features
    → Linear plasticity W (64×64, inner-loop adapted, reset zero per episode)
        Δw = η·(A·pre·post·trace + B·pre + C·post + D)
        A,B,C,D meta-aprendidos via outer loop. decay scalar global meta-aprendido.
    → k-WTA (k=16, 75% sparsity) no embedding 64D
    → Prototype classifier (cosine sim, β=8)

  Outer loop (sequencial 50 tasks): cross-entropy do query
    Backprop atualiza:
    - CNN-4 weights (slow, persistem cross-task → forgetting natural)
    - A, B, C, D, decay (slow, meta-params)
    NÃO atualiza W (W reseta zero a cada episode)

Mecanismos combinados:
  1. CNN-4 encoder      (de C3, sessão #20)
  2. Plasticidade local (de C2, sessão #19) — termos B, C, D são essenciais
  3. Trace STDP-like    (de Possib. B, sessão #27) — termo A com trace
  4. k-WTA esparso      (de C3b, sessão #20) — preserva 75% sparsity
  5. Continual sequencial sem replay (Marco 1)

Critérios de sanity (sessão #28):
  Sanity 5 tasks (1 seed reduzido, finetune=30, eval=15, n_inner=5):
    ACC entre 30-95% → loop OK, prosseguir pra sanity 50 tasks
    ACC <30% → BUG, parar
    ACC >95% → suspeita de bug (vazamento de query labels?)

  Sanity 50 tasks (1 seed, finetune=30 reduzido, eval=15, n_inner=5):
    ACC entre ~50-85% → loop OK pra escala completa
    Comparar com baseline naive (80.65%) e Possib. B (47.89%)

Critérios de sucesso (sessão #29, 5 seeds completos):
  ACC ≥85% E BWT ≥-7 → sucesso, prosseguir ablações
  ACC dentro de naive ± 2pp → sem ganho, encerrar Marco 1
  ACC <78% → pior que naive, descartar 5e

Não modifica nenhum arquivo do experiment_01 nem baseline_naive.py.

Uso:
    # Sanity 5 tasks
    python c5e_combined.py --device cuda --seeds 1 --n-tasks 5 --finetune-episodes 30 --eval-episodes 15 --n-inner 5
    # Sanity 50 tasks
    python c5e_combined.py --device cuda --seeds 1 --finetune-episodes 30 --eval-episodes 15 --n-inner 5
    # Completo (sessão #29+)
    python c5e_combined.py --device cuda
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

# Imports do experiment_02 (mesmo dir)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment_01_oneshot"))

from baseline_naive import (
    CombinedOmniglot,
    build_tasks_by_alphabet,
    TaskEpisodeSampler,
    bootstrap_ci,
    TaskData,
    ProtoEncoder,
)
from config import default_config


# ---------------------------------------------------------------------------
# k-WTA layer (de C3, sessão #20)
# ---------------------------------------------------------------------------
def kwta(z: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k WTA: mantém top-k ativações, zera resto."""
    if k >= z.shape[-1]:
        return z
    _, topk_idx = z.topk(k, dim=-1)
    mask = torch.zeros_like(z).scatter_(-1, topk_idx, 1.0)
    return z * mask


# ---------------------------------------------------------------------------
# Modelo combinado
# ---------------------------------------------------------------------------
class CombinedModel(nn.Module):
    """CNN-4 + plasticidade meta-aprendida + trace + k-WTA.

    Trainable params:
    - CNN-4 weights (via backprop standard)
    - Meta-plasticity: A, B, C, D (per peso da camada linear plast 64×64) + decay
    """

    def __init__(self, embed_dim: int = 64, k_wta: int = 16, eta: float = 0.01):
        super().__init__()
        self.embed_dim = embed_dim
        self.k_wta = k_wta
        self.eta = eta

        # CNN-4 (igual ProtoEncoder)
        self.cnn = ProtoEncoder()  # output 64D

        # Meta-plasticidade (per peso da camada linear 64×64)
        # Inicialização std=0.01 (matches c2_meta_hebbian)
        shape = (embed_dim, embed_dim)
        self.A = nn.Parameter(torch.randn(shape) * 0.01)
        self.B = nn.Parameter(torch.randn(shape) * 0.01)
        self.C = nn.Parameter(torch.randn(shape) * 0.01)
        self.D = nn.Parameter(torch.randn(shape) * 0.01)
        # decay scalar via sigmoid logit
        self.decay_logit = nn.Parameter(torch.tensor(0.0))

    def adapt_plasticity_W(self, sup_features: torch.Tensor, n_inner: int) -> torch.Tensor:
        """Inner loop: adapta W (64×64) usando support features. W reseta zero a cada chamada.

        sup_features: (B_sup, 64) embeddings pós-CNN.
        Retorna W adaptado (64, 64).
        """
        device = sup_features.device
        W = torch.zeros(self.embed_dim, self.embed_dim, device=device)
        decay = torch.sigmoid(self.decay_logit)

        pre_history = []
        for _ in range(n_inner):
            # Forward com W atual
            post = sup_features @ W.T  # (B_sup, 64)

            # Médias batch
            pre_avg = sup_features.mean(dim=0, keepdim=True)  # (1, 64)
            post_avg = post.mean(dim=0).unsqueeze(1)  # (64, 1)
            pre_history.append(pre_avg)

            # Trace exponencial
            trace = torch.zeros_like(pre_avg)  # (1, 64)
            for k_idx, pre_k in enumerate(reversed(pre_history)):
                trace = trace + (decay ** k_idx) * pre_k

            # Hebbian com trace + termos modulatórios
            hebb_with_trace = (post_avg * pre_avg) * trace  # (64, 64) via broadcast
            dW = self.eta * (
                self.A * hebb_with_trace +
                self.B * pre_avg +
                self.C * post_avg +
                self.D
            )
            W = W + dW

        return W

    def forward_episode(self, support: torch.Tensor, query: torch.Tensor, n_inner: int):
        """Pipeline completo: CNN → plasticidade adaptativa → k-WTA → embeddings.

        support: (B_sup, 1, H, W)
        query:   (B_qry, 1, H, W)
        Retorna (sup_emb, qry_emb) pós-k-WTA, shape (B, 64).
        """
        # CNN encode support and query
        sup_features = self.cnn(support)  # (B_sup, 64)
        qry_features = self.cnn(query)    # (B_qry, 64)

        # Adapt linear plasticity W via inner loop (using support features)
        W_plast = self.adapt_plasticity_W(sup_features, n_inner)

        # Apply adapted W
        sup_emb = sup_features @ W_plast.T  # (B_sup, 64)
        qry_emb = qry_features @ W_plast.T  # (B_qry, 64)

        # k-WTA esparso
        sup_emb = kwta(sup_emb, self.k_wta)
        qry_emb = kwta(qry_emb, self.k_wta)

        return sup_emb, qry_emb


def episode_loss(model: CombinedModel, episode: dict, n_classes: int, n_inner: int,
                 beta: float, device):
    """Forward completo + loss prototypical."""
    sup = episode["support"].unsqueeze(1).to(device)  # (B, 1, H, W)
    qry = episode["query"].unsqueeze(1).to(device)
    sup_lbl = episode["support_labels"].to(device)
    qry_lbl = episode["query_labels"].to(device)

    sup_emb, qry_emb = model.forward_episode(sup, qry, n_inner)

    # Prototypes per class
    prototypes = torch.zeros(n_classes, model.embed_dim, device=device)
    for c in range(n_classes):
        mask = sup_lbl == c
        prototypes[c] = sup_emb[mask].mean(dim=0)

    # Cosine sim classification (β=8)
    qry_n = F.normalize(qry_emb, dim=-1)
    proto_n = F.normalize(prototypes, dim=-1)
    logits = beta * qry_n @ proto_n.T

    loss = F.cross_entropy(logits, qry_lbl)
    acc = (logits.argmax(-1) == qry_lbl).float().mean().item()
    return loss, acc


def evaluate_task(model: CombinedModel, dataset, task: TaskData, device,
                  n_episodes: int, n_way: int, k_shot: int, n_query: int,
                  n_inner: int, beta: float, seed: int) -> float:
    sampler = TaskEpisodeSampler(dataset, task.classes, task.test_indices_by_class,
                                  n_way=n_way, k_shot=k_shot, n_query=n_query, seed=seed)
    model.eval()
    accs = []
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = sampler.sample()
            _, acc = episode_loss(model, ep, n_classes=n_way, n_inner=n_inner,
                                   beta=beta, device=device)
            accs.append(acc)
    return float(np.mean(accs))


def run_one_seed(args, seed: int, dataset, device) -> dict:
    print(f"\n{'='*72}\n[seed={seed}] Caminho 5e (CNN+plasticidade+trace+k-WTA)\n{'='*72}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    # Build tasks (alphabet mode pós-#24)
    tasks = build_tasks_by_alphabet(
        dataset, n_chars_per_task=args.n_chars_per_task,
        n_train_per_class=args.n_train_per_class,
        seed=seed, verbose=(seed == args.seed_start),
    )
    if args.n_tasks < len(tasks):
        tasks = tasks[:args.n_tasks]
    n_actual = len(tasks)
    print(f"  [seed={seed}] {n_actual} tasks")

    # Init model
    model = CombinedModel(embed_dim=64, k_wta=args.k_wta, eta=args.eta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_cnn = sum(p.numel() for p in model.cnn.parameters())
    n_meta = n_params - n_cnn
    print(f"  [seed={seed}] params: CNN={n_cnn}, meta-plast={n_meta}, total={n_params}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    acc_just_after = [0.0] * n_actual
    t0 = time.time()

    # Sequential fine-tune (skip warmup é default)
    for t in range(n_actual):
        train_sampler = TaskEpisodeSampler(
            dataset, tasks[t].classes, tasks[t].train_indices_by_class,
            n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query,
            seed=seed * 1000 + 200 + t,
        )
        model.train()
        for step in range(args.finetune_episodes):
            ep = train_sampler.sample()
            loss, _ = episode_loss(model, ep, n_classes=args.n_way, n_inner=args.n_inner,
                                    beta=args.beta, device=device)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

        # Acc just_after pra task t
        acc_just_after[t] = evaluate_task(
            model, dataset, tasks[t], device, args.eval_episodes,
            args.n_way, args.k_shot, args.n_query,
            args.n_inner, args.beta, seed * 1000 + 100 + t,
        )

        if (t + 1) % 5 == 0 or t == n_actual - 1:
            elapsed = time.time() - t0
            decay_val = torch.sigmoid(model.decay_logit).item()
            print(f"  [seed={seed}] após task {t+1}/{n_actual}, just_after={acc_just_after[t]*100:.1f}%, "
                  f"decay={decay_val:.3f}, elapsed={elapsed:.1f}s")

    # Final eval em todas tasks
    print(f"  [seed={seed}] final eval em todas as {n_actual} tasks...")
    acc_final = [0.0] * n_actual
    for t in range(n_actual):
        acc_final[t] = evaluate_task(
            model, dataset, tasks[t], device, args.eval_episodes,
            args.n_way, args.k_shot, args.n_query,
            args.n_inner, args.beta, seed * 1000 + 500 + t,
        )

    ACC = float(np.mean(acc_final))
    BWT = float(np.mean([acc_final[t] - acc_just_after[t] for t in range(n_actual)]))
    elapsed = time.time() - t0
    decay_val = torch.sigmoid(model.decay_logit).item()
    print(f"  [seed={seed}] FINAL: ACC={ACC*100:.2f}%  BWT={BWT*100:+.2f} p.p.  "
          f"decay={decay_val:.3f}  ({elapsed:.1f}s)")
    return {
        "seed": seed, "ACC": ACC, "BWT": BWT, "decay": decay_val,
        "acc_final": acc_final, "acc_just_after": acc_just_after, "elapsed": elapsed,
    }


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
    p.add_argument("--n-inner", type=int, default=10)
    p.add_argument("--eta", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=8.0)
    p.add_argument("--k-wta", type=int, default=16, help="k em top-k WTA (default 16 = 75%% sparsity em 64D)")
    args = p.parse_args()

    cfg = default_config()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Config: {args.n_tasks} tasks, n_inner={args.n_inner}, eta={args.eta}, lr={args.lr}, "
          f"k_wta={args.k_wta}, finetune={args.finetune_episodes}, eval={args.eval_episodes}, seeds={args.seeds}")

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

    print(f"\n{'='*72}")
    print(f"CAMINHO 5e — CNN + plasticidade meta + trace + k-WTA + continual")
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
        print(f"\n  ACC: {accs.mean()*100:.2f}%  BWT: {bwts.mean()*100:+.2f} p.p.  ({args.n_tasks} tasks, 1 seed)")

    print(f"\n=== Comparação ===")
    print(f"  Baseline naive ProtoNet (sessão #25):  ACC 80.65%, BWT -9.26 p.p.")
    print(f"  Possibilidade B encoder linear (#27): ACC 47.89%, BWT -2.05 p.p.")
    print(f"  ProtoNet vanilla one-shot (sessão #20): ACC 94.55% (sem continual)")
    print(f"  Caminho 5e (este):                    ACC {accs.mean()*100:.2f}%, BWT {bwts.mean()*100:+.2f} p.p.")
    print(f"  Δ vs baseline naive: ACC {accs.mean()*100 - 80.65:+.2f} p.p.")


if __name__ == "__main__":
    main()
