"""
Sessão #23 — Baseline naive sequential fine-tuning em Split-Omniglot 50-tasks.

Estabelece floor de performance pra família de propostas de continual learning
sem replay (#26-#28). Sem nenhuma defesa contra catastrophic forgetting:
encoder ProtoNet treinado sequencialmente em 50 tasks, fine-tuning livre.

Esperado pelo critério da sessão: ACC 30-50%, BWT −30 a −50%. Se acc vier
muito acima, suspeitar de bug (talvez não esteja realmente sequencial).

Setup:
- 50 tasks × 5 classes (250 classes do background set, sampled com seed)
- Por classe: 14 train + 6 test instances (deterministic split)
- Cada episódio: 5-way 1-shot 5-query (matches project convention)
- Warmup: 500 episodes joint sobre tasks 1-5
- Fine-tune: 100 episodes por task × tasks 6-50
- Eval: 50 episodes por task por medição (just_after + final)
- 5 seeds, IC95% bootstrap

Métricas (Lopez-Paz & Ranzato 2017):
- ACC = média de accuracy final em todas 50 tasks
- BWT = média de (acc_final[t] - acc_just_after[t]) — negativo = forgetting

Uso:
    python baseline_naive.py --device cuda
    python baseline_naive.py --seeds 3 --finetune-episodes 50  # debug rápido
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Imports do experimento 01
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment_01_oneshot"))
from config import default_config
from data import load_background


# ---------------------------------------------------------------------------
# ProtoEncoder duplicado (isolamento — não modificar baselines.py)
# ---------------------------------------------------------------------------
class ProtoEncoder(nn.Module):
    """CNN-4 padrão de Snell et al. — duplicado de baselines.py."""

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

    def forward(self, x):
        z = self.net(x)
        return z.flatten(1)


# ---------------------------------------------------------------------------
# Task setup: split classes em tasks, instances em train/test
# ---------------------------------------------------------------------------
@dataclass
class TaskData:
    classes: list[int]                                # 5 class IDs
    train_indices_by_class: dict[int, list[int]]      # class_id -> instance indices in dataset
    test_indices_by_class: dict[int, list[int]]


def build_tasks(dataset, n_tasks: int = 50, n_classes_per_task: int = 5,
                n_train_per_class: int = 14, seed: int = 42) -> list[TaskData]:
    """Constrói 50 tasks deterministicamente a partir do background set.

    Cada classe tem 20 instâncias em Omniglot. Splita em 14 train + 6 test
    via shuffle determinístico por seed.
    """
    flat = getattr(dataset, "_flat_character_images", None)
    if flat is None:
        raise ValueError("Dataset precisa ter _flat_character_images (Omniglot torchvision)")

    # Indexa instâncias por classe
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(flat):
        by_class[label].append(idx)
    all_classes = sorted(by_class.keys())

    n_total_classes = n_tasks * n_classes_per_task
    if len(all_classes) < n_total_classes:
        raise ValueError(f"Background tem {len(all_classes)} classes, precisa >= {n_total_classes}")

    # Sample 250 classes deterministicamente
    rng = random.Random(seed)
    selected = rng.sample(all_classes, n_total_classes)

    # Pra cada classe, splita instâncias deterministicamente
    train_by_class: dict[int, list[int]] = {}
    test_by_class: dict[int, list[int]] = {}
    for c in selected:
        instances = by_class[c]
        if len(instances) < n_train_per_class + 1:
            raise ValueError(f"Classe {c} tem {len(instances)} instâncias")
        instances_shuffled = instances.copy()
        rng.shuffle(instances_shuffled)
        train_by_class[c] = instances_shuffled[:n_train_per_class]
        test_by_class[c] = instances_shuffled[n_train_per_class:]

    # Agrupa em tasks
    tasks: list[TaskData] = []
    for t in range(n_tasks):
        cls = selected[t * n_classes_per_task : (t + 1) * n_classes_per_task]
        tasks.append(TaskData(
            classes=cls,
            train_indices_by_class={c: train_by_class[c] for c in cls},
            test_indices_by_class={c: test_by_class[c] for c in cls},
        ))
    return tasks


# ---------------------------------------------------------------------------
# Episode sampler restrito a um task (e a train ou test pool)
# ---------------------------------------------------------------------------
class TaskEpisodeSampler:
    """Sampler ProtoNet-style restrito a um conjunto de classes E um pool de instâncias.

    Diferente do EpisodeSampler de data.py em que aqui o pool de instâncias
    por classe é controlado externamente (split train/test).
    """

    def __init__(self, dataset, classes: list[int], indices_by_class: dict[int, list[int]],
                 n_way: int, k_shot: int, n_query: int, seed: int):
        self.dataset = dataset
        self.classes = classes
        self.indices_by_class = indices_by_class
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.rng = random.Random(seed)

        # Validação
        if len(classes) < n_way:
            raise ValueError(f"Task tem {len(classes)} classes, n_way={n_way}")
        for c in classes:
            if len(indices_by_class[c]) < k_shot + n_query:
                raise ValueError(f"Classe {c} tem {len(indices_by_class[c])} instâncias, "
                                 f"k_shot+n_query={k_shot+n_query}")

    def sample(self):
        chosen = self.rng.sample(self.classes, self.n_way)
        support_imgs, support_lbls = [], []
        query_imgs, query_lbls = [], []
        for new_label, cls in enumerate(chosen):
            indices = self.rng.sample(self.indices_by_class[cls], self.k_shot + self.n_query)
            for i, idx in enumerate(indices):
                img, _ = self.dataset[idx]
                if i < self.k_shot:
                    support_imgs.append(img.squeeze(0))
                    support_lbls.append(new_label)
                else:
                    query_imgs.append(img.squeeze(0))
                    query_lbls.append(new_label)
        return {
            "support": torch.stack(support_imgs),
            "support_labels": torch.tensor(support_lbls),
            "query": torch.stack(query_imgs),
            "query_labels": torch.tensor(query_lbls),
        }


class JointTasksEpisodeSampler(TaskEpisodeSampler):
    """Sampler que sorteia n_way classes do union de várias tasks (warmup)."""

    def __init__(self, dataset, tasks: list[TaskData], n_way: int, k_shot: int, n_query: int, seed: int):
        all_classes = []
        all_indices_by_class = {}
        for t in tasks:
            all_classes.extend(t.classes)
            all_indices_by_class.update(t.train_indices_by_class)
        super().__init__(dataset, all_classes, all_indices_by_class, n_way, k_shot, n_query, seed)


# ---------------------------------------------------------------------------
# ProtoNet episode loss (mesma de baselines.py)
# ---------------------------------------------------------------------------
def proto_episode_loss(encoder: nn.Module, episode: dict, n_classes: int, device):
    sup = episode["support"].unsqueeze(1).to(device)  # (N*K, 1, H, W)
    qry = episode["query"].unsqueeze(1).to(device)
    sup_lbl = episode["support_labels"].to(device)
    qry_lbl = episode["query_labels"].to(device)

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


def evaluate_task(encoder: nn.Module, dataset, task: TaskData, device,
                  n_episodes: int, n_way: int, k_shot: int, n_query: int, seed: int) -> float:
    """Evaluate accuracy on a task's test pool over n_episodes."""
    sampler = TaskEpisodeSampler(dataset, task.classes, task.test_indices_by_class,
                                  n_way=n_way, k_shot=k_shot, n_query=n_query, seed=seed)
    encoder.eval()
    accs = []
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = sampler.sample()
            _, acc = proto_episode_loss(encoder, ep, n_classes=n_way, device=device)
            accs.append(acc)
    return float(np.mean(accs))


# ---------------------------------------------------------------------------
# Run one seed: train + measure ACC + BWT
# ---------------------------------------------------------------------------
def run_one_seed(args, seed: int, dataset, device) -> dict:
    print(f"\n{'='*72}\n[seed={seed}]\n{'='*72}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build tasks (deterministic per seed)
    tasks = build_tasks(dataset, n_tasks=args.n_tasks, n_classes_per_task=args.n_way,
                        n_train_per_class=args.n_train_per_class, seed=seed)
    print(f"  [seed={seed}] {len(tasks)} tasks construídos")

    encoder = ProtoEncoder().to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    acc_just_after = [0.0] * args.n_tasks  # acc no fim do treino daquela task
    t0 = time.time()

    # Phase 1: warmup
    n_warmup = args.n_warmup_tasks
    warmup_sampler = JointTasksEpisodeSampler(
        dataset, tasks[:n_warmup], n_way=args.n_way, k_shot=args.k_shot,
        n_query=args.n_query, seed=seed * 1000 + 1,
    )
    encoder.train()
    for step in range(args.warmup_episodes):
        ep = warmup_sampler.sample()
        loss, _ = proto_episode_loss(encoder, ep, n_classes=args.n_way, device=device)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"  [seed={seed}] warmup ({args.warmup_episodes} eps) concluído em {time.time()-t0:.1f}s")

    # Acc just_after pra warmup tasks
    for t in range(n_warmup):
        acc_just_after[t] = evaluate_task(encoder, dataset, tasks[t], device,
                                          n_episodes=args.eval_episodes,
                                          n_way=args.n_way, k_shot=args.k_shot,
                                          n_query=args.n_query, seed=seed * 1000 + 100 + t)

    # Phase 2: fine-tune sequencial
    for t in range(n_warmup, args.n_tasks):
        train_sampler = TaskEpisodeSampler(
            dataset, tasks[t].classes, tasks[t].train_indices_by_class,
            n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query,
            seed=seed * 1000 + 200 + t,
        )
        encoder.train()
        for step in range(args.finetune_episodes):
            ep = train_sampler.sample()
            loss, _ = proto_episode_loss(encoder, ep, n_classes=args.n_way, device=device)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Acc just_after pra task t
        acc_just_after[t] = evaluate_task(encoder, dataset, tasks[t], device,
                                          n_episodes=args.eval_episodes,
                                          n_way=args.n_way, k_shot=args.k_shot,
                                          n_query=args.n_query, seed=seed * 1000 + 100 + t)

        if (t + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [seed={seed}] após task {t+1}/{args.n_tasks}, just_after acc = {acc_just_after[t]*100:.1f}%, elapsed={elapsed:.1f}s")

    # Phase 3: final eval em todas 50 tasks
    print(f"  [seed={seed}] final eval em todas as {args.n_tasks} tasks...")
    acc_final = [0.0] * args.n_tasks
    for t in range(args.n_tasks):
        acc_final[t] = evaluate_task(encoder, dataset, tasks[t], device,
                                     n_episodes=args.eval_episodes,
                                     n_way=args.n_way, k_shot=args.k_shot,
                                     n_query=args.n_query, seed=seed * 1000 + 500 + t)

    # Métricas
    ACC = float(np.mean(acc_final))
    BWT = float(np.mean([acc_final[t] - acc_just_after[t] for t in range(args.n_tasks)]))
    elapsed = time.time() - t0
    print(f"  [seed={seed}] FINAL: ACC={ACC*100:.2f}%  BWT={BWT*100:+.2f} p.p.  ({elapsed:.1f}s)")
    return {
        "seed": seed,
        "ACC": ACC,
        "BWT": BWT,
        "acc_final": acc_final,
        "acc_just_after": acc_just_after,
        "elapsed": elapsed,
    }


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05):
    rng = np.random.default_rng(0)
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", type=int, default=5, help="quantos seeds rodar")
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--n-tasks", type=int, default=50)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=1)
    p.add_argument("--n-query", type=int, default=5)
    p.add_argument("--n-train-per-class", type=int, default=14)
    p.add_argument("--n-warmup-tasks", type=int, default=5)
    p.add_argument("--warmup-episodes", type=int, default=500)
    p.add_argument("--finetune-episodes", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    cfg = default_config()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Config: {args.n_tasks} tasks × {args.n_way}-way {args.k_shot}-shot, "
          f"warmup={args.warmup_episodes} eps, finetune={args.finetune_episodes} eps/task, "
          f"eval={args.eval_episodes} eps/task, {args.seeds} seeds")

    # Carrega dataset uma vez (sample dos tasks é dependente do seed mas dataset é o mesmo)
    print("\nCarregando background set...")
    dataset = load_background(cfg)

    sess_t0 = time.time()
    results = []
    for s in range(args.seeds):
        seed = args.seed_start + s
        r = run_one_seed(args, seed, dataset, device)
        results.append(r)

    total_time = time.time() - sess_t0
    print(f"\n\nTempo total: {total_time:.1f}s ({total_time/60:.1f} min) pra {args.seeds} seeds")

    # Agregação
    accs = np.array([r["ACC"] for r in results])
    bwts = np.array([r["BWT"] for r in results])
    acc_mean = accs.mean()
    bwt_mean = bwts.mean()
    acc_lo, acc_hi = bootstrap_ci(accs)
    bwt_lo, bwt_hi = bootstrap_ci(bwts)

    print(f"\n{'='*72}")
    print(f"BASELINE NAIVE — Split-Omniglot 50-tasks")
    print(f"{'='*72}")
    seeds_list = [r["seed"] for r in results]
    accs_str = [f"{r['ACC']*100:.2f}%" for r in results]
    bwts_str = [f"{r['BWT']*100:+.2f}" for r in results]
    print(f"  Seeds: {seeds_list}")
    print(f"  ACC final por seed: {accs_str}")
    print(f"  BWT por seed:       {bwts_str}")
    print(f"")
    print(f"  ACC média: {acc_mean*100:.2f}%  IC95% [{acc_lo*100:.2f}, {acc_hi*100:.2f}]")
    print(f"  BWT média: {bwt_mean*100:+.2f} p.p.  IC95% [{bwt_lo*100:+.2f}, {bwt_hi*100:+.2f}]")

    # Sanity check vs critério da sessão
    print(f"\n=== Sanity check vs critério ===")
    print(f"  Esperado: ACC 30-50%, BWT −30 a −50%")
    if 30 <= acc_mean * 100 <= 50:
        print(f"  ✓ ACC dentro do esperado")
    elif acc_mean * 100 > 50:
        print(f"  ⚠️ ACC ACIMA do esperado ({acc_mean*100:.1f}%) — possível bug (sequencial real?)")
    else:
        print(f"  ⚠️ ACC ABAIXO do esperado ({acc_mean*100:.1f}%) — mas ainda informativo")
    if -50 <= bwt_mean * 100 <= -30:
        print(f"  ✓ BWT dentro do esperado")
    elif bwt_mean * 100 > -30:
        print(f"  ⚠️ BWT MENOS forgetting que esperado ({bwt_mean*100:+.1f}) — verificar setup")
    else:
        print(f"  ⚠️ BWT MAIS forgetting que esperado ({bwt_mean*100:+.1f})")


if __name__ == "__main__":
    main()
