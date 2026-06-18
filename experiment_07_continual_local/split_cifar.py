"""Marco 4 — divide CIFAR-100 em T tarefas (subconjuntos disjuntos de classes) para continual learning.

Reaproveita cifar100.npz do Marco 3. Cada tarefa: train/test só das suas classes, labels remapeados
para 0..(classes_por_tarefa-1). Pré-carrega no device (GPU).
"""
from __future__ import annotations
import os
import numpy as np
import torch

DATA = os.path.join(os.path.dirname(__file__), "..", "experiment_06_plasticity", "data", "cifar100.npz")


def load_tasks(n_tasks=5, seed=0, device="cpu"):
    d = np.load(DATA)
    trX, trY, teX, teY = d["train_x"], d["train_y"], d["test_x"], d["test_y"]
    classes = np.arange(100)
    np.random.default_rng(seed).shuffle(classes)
    per = 100 // n_tasks
    assert 100 % n_tasks == 0, "n_tasks deve dividir 100"

    def to_tensor(X, Y, mask, remap):
        x = torch.tensor(X[mask], dtype=torch.float, device=device).div_(255).movedim(-1, 1).contiguous()
        y = torch.tensor([remap[int(c)] for c in Y[mask]], device=device)
        return x, y

    tasks = []
    for t in range(n_tasks):
        cls = classes[t * per:(t + 1) * per]
        remap = {int(c): i for i, c in enumerate(cls)}
        tasks.append({
            "classes": cls.tolist(),
            "n_classes": per,
            "train": to_tensor(trX, trY, np.isin(trY, cls), remap),
            "test": to_tensor(teX, teY, np.isin(teY, cls), remap),
        })
    return tasks


if __name__ == "__main__":
    tasks = load_tasks(n_tasks=5, seed=0)
    print(f"{len(tasks)} tarefas")
    for i, tk in enumerate(tasks):
        trx, tr_y = tk["train"]; tex, te_y = tk["test"]
        print(f"  task {i}: {tk['n_classes']} classes, train {tuple(trx.shape)} y[{int(tr_y.min())}..{int(tr_y.max())}], test {tuple(tex.shape)}")
