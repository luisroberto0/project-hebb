"""
Sanity check da Semana 1: reprodução simplificada de Diehl & Cook 2015.

Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition
using spike-timing-dependent plasticity. Frontiers in Computational Neuroscience.

Versão simplificada (CPU-friendly):
  - Codificação Poisson rate (T timesteps)
  - 1 camada FC-equivalente: ConvSTDPLayer com kernel=28 (cobre input completo)
    e N filtros (cada filtro vira um "feature detector" sem rótulo)
  - STDP unsupervised durante pretreino
  - Label assignment pós-hoc: cada filtro recebe o dígito que mais o ativa
  - Avaliação por voto majoritário dos filtros mais ativos

Critério de sucesso (definido em PLAN.md §Semana 1): ≥ 85% em test set.
Se não bater, trava o roadmap — significa que ou a stack ou o entendimento
de STDP estão errados.

Uso:
    python sanity_mnist.py                    # config padrão (1 epoch, 5k imgs)
    python sanity_mnist.py --epochs 3 --n-images 60000   # treino completo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import default_config, Config
from model import ConvSTDPLayer
from data import poisson_encode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--n-images", type=int, default=5000,
                   help="Subset do MNIST pra debug. Use 60000 pra treino completo.")
    p.add_argument("--n-filters", type=int, default=100,
                   help="Diehl & Cook usaram 100, 400, 1600, 6400. 100 é o mínimo decente.")
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default="./data")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Modelo de sanity check: 1 camada STDP cobrindo input inteiro
# ---------------------------------------------------------------------------
class SanityNet(torch.nn.Module):
    """
    Rede mínima estilo Diehl & Cook 2015.
    ConvSTDPLayer com kernel = tamanho da imagem (28×28) é equivalente a
    uma FC com 784 entradas e n_filters saídas, mantendo a regra STDP que já
    implementamos pra a versão convolucional.
    """

    def __init__(self, cfg: Config, n_filters: int):
        super().__init__()
        self.cfg = cfg
        self.n_filters = n_filters
        # padding=0 → output (1, 1) por imagem, ou seja, 1 valor por filtro.
        self.layer = ConvSTDPLayer(
            in_channels=1, out_channels=n_filters, kernel_size=28, cfg=cfg,
        )
        self.layer.padding = 0
        self.layer.conv.padding = (0, 0)

    def forward_image(self, image: torch.Tensor, train_stdp: bool) -> torch.Tensor:
        """
        image: (B, 1, 28, 28) em [0,1].
        Retorna spike_count: (B, n_filters) — total de spikes de cada filtro
        durante T timesteps.
        """
        cfg = self.cfg
        T = cfg.spike.timesteps
        device = image.device
        spikes_in = poisson_encode(image, T, cfg.spike.max_rate_hz, cfg.spike.dt_ms)  # (T, B, 1, 28, 28)

        B = image.shape[0]
        # Output de Conv2d com kernel=28, pad=0: shape (B, n_filters, 1, 1)
        mem = torch.zeros(B, self.n_filters, 1, 1, device=device)
        spike_count = torch.zeros(B, self.n_filters, 1, 1, device=device)

        if train_stdp:
            self.layer.apre = None; self.layer.apost = None

        for t in range(T):
            pre = spikes_in[t]
            spk, mem = self.layer(pre, mem)
            if train_stdp:
                self.layer.stdp_update(pre, spk)
            spike_count = spike_count + spk

        return spike_count.view(B, self.n_filters)


# ---------------------------------------------------------------------------
# Label assignment (Diehl & Cook §2.5)
# ---------------------------------------------------------------------------
def assign_labels(net: SanityNet, loader: DataLoader, device: torch.device, n_classes: int = 10) -> torch.Tensor:
    """
    Pra cada filtro, mede acumulação de spikes por classe sobre conjunto
    rotulado. Filtro recebe label da classe que mais o ativou em média.
    Retorna tensor (n_filters,) com label int de cada filtro.
    """
    net.eval()
    response = torch.zeros(net.n_filters, n_classes, device=device)
    counts = torch.zeros(n_classes, device=device)
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device); label = label.to(device)
            sc = net.forward_image(img, train_stdp=False)  # (B, F)
            for c in range(n_classes):
                mask = (label == c)
                if mask.any():
                    response[:, c] += sc[mask].sum(dim=0)
                    counts[c] += mask.sum()
    avg_response = response / counts.clamp(min=1)
    return avg_response.argmax(dim=1)


# ---------------------------------------------------------------------------
# Avaliação por voto
# ---------------------------------------------------------------------------
def evaluate(net: SanityNet, loader: DataLoader, filter_labels: torch.Tensor,
             device: torch.device, n_classes: int = 10) -> float:
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device); label = label.to(device)
            sc = net.forward_image(img, train_stdp=False)  # (B, F)
            # Pra cada classe, soma spikes dos filtros atribuídos a ela
            class_score = torch.zeros(img.shape[0], n_classes, device=device)
            for c in range(n_classes):
                mask = (filter_labels == c)
                if mask.any():
                    class_score[:, c] = sc[:, mask].sum(dim=1) / mask.sum()
            pred = class_score.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Fix Windows cp1252 encoding issue
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    args = parse_args()
    cfg = default_config()
    cfg.spike.timesteps = args.timesteps
    if args.device:
        cfg.device = args.device

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    print(f"Dispositivo: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
    ])
    train_ds = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)

    if args.n_images < len(train_ds):
        idx = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.seed))[:args.n_images].tolist()
        train_ds = Subset(train_ds, idx)

    # Hipótese 2: verificar distribuição de classes no subset
    full_ds = datasets.MNIST(root=args.data_root, train=True, download=False, transform=transform)
    labels_count = [0] * 10
    for i in range(len(train_ds)):
        if isinstance(train_ds, Subset):
            _, label = full_ds[train_ds.indices[i]]
        else:
            _, label = train_ds[i]
        labels_count[label] += 1
    print(f"Imagens de treino: {len(train_ds)}, teste: {len(test_ds)}")
    print(f"Distribuição de classes no subset: {labels_count}")

    # Loaders pequenos pra STDP que é online (batch=1 idealmente, mas vetorizamos batch)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    assign_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    net = SanityNet(cfg, n_filters=args.n_filters).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Parâmetros: {n_params}  (filtros: {args.n_filters})")

    # ----- Pretreino STDP -----
    print(f"\nPretreino STDP por {args.epochs} epoch(s)...")
    net.train()
    t0 = time.time()
    seen = 0
    for epoch in range(args.epochs):
        for step, (img, _) in enumerate(train_loader):
            img = img.to(device)
            with torch.no_grad():
                _ = net.forward_image(img, train_stdp=True)
            seen += img.shape[0]
            if step % 50 == 0:
                w = net.layer.conv.weight.data
                print(f"  epoch {epoch} step {step:4d}  "
                      f"w mean={w.mean():.3f} std={w.std():.3f} "
                      f"min={w.min():.3f} max={w.max():.3f}  seen={seen}")

    elapsed = time.time() - t0
    print(f"Pretreino concluído em {elapsed:.1f}s ({seen/elapsed:.1f} imgs/s)")

    # ----- Label assignment -----
    print("\nAtribuindo labels aos filtros (Diehl & Cook §2.5)...")
    filter_labels = assign_labels(net, assign_loader, device)
    counts = torch.bincount(filter_labels, minlength=10)
    print(f"Distribuição de labels nos filtros: {counts.tolist()}")
    if (counts == 0).any():
        print("⚠️  Algumas classes não têm nenhum filtro atribuído — STDP colapsou ou filtros são poucos.")

    # ----- Avaliação -----
    print("\nAvaliando em test set (10k imagens)...")
    acc = evaluate(net, test_loader, filter_labels, device)
    print(f"\nAcurácia teste: {acc*100:.2f}%")

    # ----- Critério de sucesso -----
    print("\n" + "=" * 60)
    if acc >= 0.85:
        print(f"✅ SANITY CHECK OK ({acc*100:.2f}% ≥ 85%)")
        print("   STDP funciona. Liberar Semana 2 (adaptação Omniglot).")
    elif acc >= 0.70:
        print(f"⚠️  SANITY CHECK PARCIAL ({acc*100:.2f}% entre 70-85%)")
        print("   STDP aprende algo mas abaixo de Diehl & Cook. Investigar:")
        print("   - hiperparâmetros (taus, A_pre, A_post, w_init)")
        print("   - número de filtros (Diehl & Cook usaram até 6400)")
        print("   - timesteps T (mais T = mais aprendizado por imagem)")
    else:
        print(f"❌ SANITY CHECK FALHOU ({acc*100:.2f}% < 70%)")
        print("   Possíveis causas:")
        print("   - bug na regra STDP (revise stdp_update em model.py)")
        print("   - codificação spike inadequada (Poisson rate vs temporal)")
        print("   - pretreino curto demais (--epochs ou --n-images baixos)")
        print("   - bug em label assignment ou evaluate")
        print("   Documente em experiment_01_oneshot/POSTMORTEM.md.")
    print("=" * 60)

    # Salva checkpoint pra inspeção/reuso
    ckpt_path = Path("checkpoints/sanity_mnist.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": net.state_dict(),
        "filter_labels": filter_labels.cpu(),
        "accuracy": acc,
        "config": {
            "n_filters": args.n_filters,
            "epochs": args.epochs,
            "n_images": args.n_images,
            "timesteps": args.timesteps,
        },
    }, ckpt_path)
    print(f"Checkpoint salvo em {ckpt_path}")


if __name__ == "__main__":
    main()
