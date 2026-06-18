"""
Marco 3 — solidificação: visualiza os filtros da 1a camada Hebbian (SoftHebb) vs random.
Se os filtros Hebbian forem detectores reais (Gabor/cor/borda) e os random forem ruido,
confirma VISUALMENTE que a plasticidade local aprendeu features genuinas (nao arquitetura).
Gera figs/fig_filters.png.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from softhebb_cifar import DeepSoftHebb, CIFAR10NPZ, train_unsup

FIGS = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(FIGS, exist_ok=True)


def grid(weight, ax, title):
    # weight: (96,3,5,5) -> grid 8x12 de patches RGB normalizados por filtro
    w = weight.detach().cpu().numpy()
    n = w.shape[0]
    rows, cols = 8, 12
    canvas = np.ones((rows * 6 - 1, cols * 6 - 1, 3))
    for i in range(min(n, rows * cols)):
        f = w[i].transpose(1, 2, 0)  # (5,5,3)
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        r, c = divmod(i, cols)
        canvas[r * 6:r * 6 + 5, c * 6:c * 6 + 5] = f
    ax.imshow(canvas)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    trainset = CIFAR10NPZ(train=True, device=dev)
    loader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)

    # random (sem treino)
    torch.manual_seed(0)
    m_rand = DeepSoftHebb().to(dev)
    w_rand = m_rand.conv1.weight.clone()

    # softhebb (treino Hebbian local)
    torch.manual_seed(0)
    m_hebb = DeepSoftHebb().to(dev)
    train_unsup(m_hebb, loader, dev, epochs=1)
    w_hebb = m_hebb.conv1.weight.clone()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    grid(w_rand, axes[0], "Pesos RANDOM (sem treino) -> ruido")
    grid(w_hebb, axes[1], "Aprendidos por Hebbian local (SoftHebb) -> features reais")
    fig.suptitle("Marco 3: filtros da 1a camada -- plasticidade local SEM backprop aprende features genuinas",
                 fontsize=13)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIGS, f"fig_filters.{ext}"), dpi=200, bbox_inches="tight")
    plt.close()

    # metrica quantitativa: diversidade dos filtros (filtros random sao i.i.d.; Hebbian devem ser estruturados)
    def offdiag_cos(w):
        f = w.view(w.shape[0], -1)
        f = f / (f.norm(dim=1, keepdim=True) + 1e-8)
        c = (f @ f.t()).abs()
        n = c.shape[0]
        return (c.sum() - n) / (n * (n - 1))  # cos medio off-diagonal
    print(f"filtros salvos em figs/fig_filters.png")
    print(f"cos off-diag medio: random={offdiag_cos(w_rand):.3f}  hebbian={offdiag_cos(w_hebb):.3f}")
    print(f"norma media: random={w_rand.view(96,-1).norm(dim=1).mean():.3f}  hebbian={w_hebb.view(96,-1).norm(dim=1).mean():.3f}")


if __name__ == "__main__":
    main()
