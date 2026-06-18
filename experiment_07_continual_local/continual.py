"""Marco 4 — continual learning com plasticidade local (SoftHebb sequencial vs backprop).

Treina a pilha de features SEQUENCIALMENTE nas T tarefas (Split-CIFAR-100), sem revisitar dados
antigos. Mede acc_matrix[t][i] = acc na tarefa i com a pilha após treinar até a tarefa t.
  ACC  = média_i acc_matrix[T-1][i]            (acurácia final média)
  BWT  = média_i (acc_matrix[T-1][i] - acc_matrix[i][i])   (forgetting; ~0 = não esquece)

Hipótese: SoftHebb (features genéricas não-sup) tem BWT ≈ 0; backprop e2e sequencial esquece.

Uso: python continual.py --method softhebb --tasks 5 --seed 0
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiment_06_plasticity"))
from softhebb_cifar import DeepSoftHebb, DeepBackpropCNN, TensorLRSGD, WeightNormDependentLR  # noqa
from split_cifar import load_tasks

FEAT_DIM = 24576


def train_stack_on_task(model, task, device, epochs=1):
    """Treina (incrementa) a pilha SoftHebb nos dados de UMA tarefa (sem labels, single-pass)."""
    for m in (model.conv1, model.conv2, model.conv3, model.bn1, model.bn2, model.bn3):
        m.train()
    opt = TensorLRSGD([
        {"params": model.conv1.parameters(), "lr": -0.08},
        {"params": model.conv2.parameters(), "lr": -0.005},
        {"params": model.conv3.parameters(), "lr": -0.01},
    ], lr=0)
    sched = WeightNormDependentLR(opt, power_lr=0.5)
    loader = DataLoader(TensorDataset(*task["train"]), batch_size=10, shuffle=True)
    for _ in range(epochs):
        for x, _y in loader:
            opt.zero_grad()
            with torch.no_grad():
                model(x)
            opt.step(); sched.step()


@torch.no_grad()
def _features(model, x, bs=500):
    model.eval()
    return torch.cat([model.features(x[k:k+bs]) for k in range(0, x.shape[0], bs)])


def probe_task(model, task, device, epochs=15):
    """Linear-probe nas classes de UMA tarefa, sobre features congeladas da pilha atual. Retorna acc."""
    trx, try_ = task["train"]; tex, tey = task["test"]
    f_tr = _features(model, trx); f_te = _features(model, tex)
    clf = nn.Linear(FEAT_DIM, task["n_classes"]).to(device)
    opt = optim.Adam(clf.parameters(), lr=1e-3); crit = nn.CrossEntropyLoss()
    n = f_tr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for k in range(0, n, 64):
            idx = perm[k:k+64]
            loss = crit(clf(f_tr[idx]), try_[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (clf(f_te).argmax(1) == tey).float().mean().item() * 100
    return acc


def softhebb_continual(tasks, device, unsup_epochs=1, probe_epochs=15):
    T = len(tasks)
    model = DeepSoftHebb().to(device)
    accm = np.full((T, T), np.nan)
    for t in range(T):
        train_stack_on_task(model, tasks[t], device, unsup_epochs)
        for i in range(t + 1):
            accm[t, i] = probe_task(model, tasks[i], device, probe_epochs)
    return accm


def metrics(accm):
    T = accm.shape[0]
    acc = np.nanmean(accm[T-1, :])
    bwt = np.mean([accm[T-1, i] - accm[i, i] for i in range(T-1)]) if T > 1 else 0.0
    return acc, bwt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["softhebb"], default="softhebb")  # backprop: proximo passo
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unsup-epochs", type=int, default=1)
    ap.add_argument("--probe-epochs", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    tasks = load_tasks(n_tasks=args.tasks, seed=args.seed, device=dev)
    accm = softhebb_continual(tasks, dev, args.unsup_epochs, args.probe_epochs)
    acc, bwt = metrics(accm)
    print(f"method={args.method} tasks={args.tasks} seed={args.seed} ACC={acc:.2f} BWT={bwt:+.2f}")
    print("acc_matrix (linha t = apos treinar ate tarefa t; col i = acc tarefa i):")
    for t in range(accm.shape[0]):
        print("  " + " ".join(f"{accm[t,i]:5.1f}" if not np.isnan(accm[t,i]) else "  .  " for i in range(accm.shape[1])))


if __name__ == "__main__":
    main()
