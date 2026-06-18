"""Marco 5 (escala) — SoftHebb em Tiny-ImageNet (200 classes, 32x32). Reusa o harness do Marco 3.

Mesma regra/arquitetura SoftHebb do Marco 3, só troca o dataset (Tiny-ImageNet) e o classificador
(10 -> 200 classes). Caracteriza até onde a plasticidade local escala em nº de classes.

Uso: python scale.py --mode softhebb --seed 0 --probe-epochs 30
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiment_06_plasticity"))
from softhebb_cifar import DeepSoftHebb, DeepBackpropCNN, train_unsup, train_probe, train_backprop, evaluate  # noqa

NPZ = os.path.join(os.path.dirname(__file__), "data", "tinyimagenet32.npz")
N_CLASSES = 200


class TinyNPZ(torch.utils.data.Dataset):
    def __init__(self, train=True, device="cpu"):
        d = np.load(NPZ)
        x = d["train_x"] if train else d["test_x"]
        y = d["train_y"] if train else d["test_y"]
        self.data = torch.tensor(x, dtype=torch.float, device=device).div_(255).movedim(-1, 1).contiguous()
        self.targets = torch.tensor(y, device=device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]


def make_softhebb(dev, competitive=True):
    m = DeepSoftHebb(competitive=competitive).to(dev)
    m.classifier = nn.Linear(24576, N_CLASSES).to(dev)
    m.classifier.weight.data = 0.11048543456039805 * torch.rand(N_CLASSES, 24576, device=dev)
    return m


def run(mode, seed, probe_epochs, dev):
    torch.manual_seed(seed)
    tr = TinyNPZ(True, dev); te = TinyNPZ(False, dev)
    unsup = DataLoader(tr, 10, shuffle=True)
    sup = DataLoader(tr, 64, shuffle=True)
    test = DataLoader(te, 1000)
    if mode == "backprop":
        m = DeepBackpropCNN().to(dev)
        m.net[-1] = nn.Linear(24576, N_CLASSES).to(dev)
        train_backprop(m, sup, dev, probe_epochs)
    else:
        m = make_softhebb(dev)
        if mode == "softhebb":
            train_unsup(m, unsup, dev)
        # mode == "random": pilha random congelada
        train_probe(m, sup, dev, probe_epochs)
    return evaluate(m, test, dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["softhebb", "random", "backprop"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    acc = run(args.mode, args.seed, args.probe_epochs, dev)
    line = f"tiny-imagenet mode={args.mode} seed={args.seed} probe_epochs={args.probe_epochs} acc={acc:.2f} (chance={100/N_CLASSES:.1f}%)"
    print(line)
    with open(os.path.join(os.path.dirname(__file__), "results_scale.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
