"""Marco 3/4 — fecho de eficiência: QUANTIFICA a contribuição ortogonal do SoftHebb.

O Marco 4 concluiu que a vantagem do SoftHebb é ortogonal (local/online/single-pass/sem-backprop =
EFICIÊNCIA), não capacidade exclusiva. Aqui medimos essa eficiência em software (prévia do Marco 6
em hardware): wall-clock de treino das FEATURES, SoftHebb (1 passada, sem backprop, sem labels) vs
backprop e2e (N épocas, com labels). CIFAR-10.
"""
from __future__ import annotations
import os, sys, time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from softhebb_cifar import DeepSoftHebb, DeepBackpropCNN, CIFAR10NPZ, train_unsup

BP_EPOCHS = 50  # backprop precisa de ~50 épocas (atinge 87%); SoftHebb = 1 passada


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = CIFAR10NPZ(train=True, device=dev)
    unsup = DataLoader(tr, batch_size=10, shuffle=True)
    sup = DataLoader(tr, batch_size=64, shuffle=True)
    crit = nn.CrossEntropyLoss()

    # --- SoftHebb: treino das features = 1 passada local, sem backprop ---
    torch.manual_seed(0)
    m = DeepSoftHebb().to(dev)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    t0 = time.time()
    train_unsup(m, unsup, dev, epochs=1)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    t_softhebb = time.time() - t0

    # --- Backprop: 1 época e2e (extrapola p/ BP_EPOCHS) ---
    torch.manual_seed(0)
    bp = DeepBackpropCNN().to(dev)
    opt = optim.Adam(bp.parameters(), lr=1e-3)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    t0 = time.time()
    bp.train()
    for x, y in sup:
        opt.zero_grad(); crit(bp(x), y).backward(); opt.step()
    torch.cuda.synchronize() if dev.type == "cuda" else None
    t_bp_epoch = time.time() - t0
    t_bp_full = t_bp_epoch * BP_EPOCHS

    ratio = t_bp_full / t_softhebb
    lines = [
        "Eficiência de treino das FEATURES (CIFAR-10, RTX 4070):",
        f"  SoftHebb  : {t_softhebb:6.1f}s  (1 passada local, SEM backprop, SEM labels)  -> features ~80%",
        f"  Backprop  : {t_bp_full:6.1f}s  ({BP_EPOCHS} épocas e2e = {t_bp_epoch:.1f}s/época, com labels) -> ~87%",
        f"  => SoftHebb treina features {ratio:.1f}x MAIS RAPIDO (e sem labels/backprop), a -7pp de acc.",
        "",
        "Interpretação: a contribuição do SoftHebb (Marco 4) e ORTOGONAL e agora QUANTIFICADA --",
        "a mesma resistencia a forgetting / features uteis, por uma fracao do custo de treino.",
    ]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(os.path.dirname(__file__), "results_efficiency.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
