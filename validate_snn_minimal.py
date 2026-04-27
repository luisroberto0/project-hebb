"""
SNN mínima end-to-end usando snnTorch sobre PyTorch + CUDA.

Treina uma rede LIF de 2 camadas em MNIST por 1 epoch só pra confirmar
que tudo funciona: dataset, GPU, surrogate gradient, backprop em spikes.

Não é um experimento sério — é validação de stack. Saída esperada:
loss caindo, acurácia >85% no fim do epoch único, GPU em uso.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import snntorch as snn
    from snntorch import surrogate
except ImportError as e:
    raise SystemExit(
        "snnTorch não instalado. Rode: pip install snntorch"
    ) from e

from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,)),
])

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

BATCH = 128
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

# ---------------------------------------------------------------------------
# Modelo: 784 -> 256 LIF -> 10 LIF
# ---------------------------------------------------------------------------
T = 25  # número de timesteps por amostra
BETA = 0.95  # fator de decay do potencial de membrana
spike_grad = surrogate.fast_sigmoid()


class LIFNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=spike_grad)
        self.fc2 = nn.Linear(256, 10)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=spike_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        x = x.view(x.size(0), -1)
        for _ in range(T):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
        # Saída = soma de spikes ao longo do tempo (rate code)
        return torch.stack(spk2_rec, dim=0).sum(dim=0)


model = LIFNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# ---------------------------------------------------------------------------
# Treino: 1 epoch
# ---------------------------------------------------------------------------
print("\nTreinando 1 epoch (validação rápida)...")
model.train()
t0 = time.time()
for step, (x, y) in enumerate(train_loader):
    x, y = x.to(DEVICE), y.to(DEVICE)
    out = model(x)
    loss = loss_fn(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"  step {step:4d}/{len(train_loader)}   loss={loss.item():.4f}")

elapsed = time.time() - t0
print(f"\nEpoch concluído em {elapsed:.1f}s ({len(train_loader)*BATCH/elapsed:.0f} samples/s)")

# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------
model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

acc = 100 * correct / total
print(f"Acurácia teste: {acc:.2f}%")
if acc < 80:
    print("⚠️  Acurácia baixa pra esse setup. Algo na stack pode estar errado.")
else:
    print("✅ Stack validada: SNN treina e generaliza.")
