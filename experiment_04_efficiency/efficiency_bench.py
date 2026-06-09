"""
Marco 2-B (#69) — benchmark de eficiência: MLP denso vs SNN-LIF vs SNN+k-WTA temporal.

Mede, em Fashion-MNIST:
  - acurácia (test split)
  - SynOps/amostra (Σ spikes pré-sinápticos × fan-out, sobre T) para as SNNs
  - MACs/amostra (analítico) para o denso
  - latência de inferência em CPU single-thread (a régua do CONTEXT: "rodar em CPU comum")

NÃO é o experimento final — é o harness + smoke pra primeiros números. Predição registrada
(experiment_04_efficiency/PLAN.md): SNN rate-coded com T~25 dá ~2-3× PIOR em SynOps
(overhead de timesteps domina); só k-WTA agressivo poderia chegar a 5×; latência CPU provável pior.

Uso:
    python efficiency_bench.py --epochs 3 --T 25 --k 32 --device cuda
    $env:HEBB_QUICK="1"  # smoke ainda mais rápido (subset)
"""
from __future__ import annotations
import argparse, os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import surrogate, spikegen


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
HID = 256
IN = 28 * 28
OUT = 10


class DenseMLP(nn.Module):
    """Baseline denso. MACs/amostra = IN*HID + HID*OUT (um forward)."""
    macs = IN * HID + HID * OUT

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN, HID), nn.ReLU(),
            nn.Linear(HID, OUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


class SNN(nn.Module):
    """SNN-LIF 2 camadas, rate (Poisson) input, surrogate-gradient.

    Se k is not None: k-WTA temporal na camada hidden — só os k neurônios de maior
    membrana podem disparar por timestep. Conta SynOps quando count=True.
    """

    def __init__(self, T: int = 25, beta: float = 0.95, k: int | None = None,
                 k_in: int | None = None) -> None:
        super().__init__()
        self.T, self.k, self.k_in = T, k, k_in
        sg = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(IN, HID)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=sg)
        self.fc2 = nn.Linear(HID, OUT)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=sg)

    def forward(self, x: torch.Tensor, count: bool = False):
        B = x.size(0)
        x = x.view(B, -1)
        spk_in = spikegen.rate(x, num_steps=self.T)  # (T, B, IN) binário
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        out_sum = torch.zeros(B, OUT, device=x.device)
        synops = torch.zeros((), device=x.device)
        for t in range(self.T):
            s_in = spk_in[t]
            if self.k_in is not None:
                # k-WTA temporal na ENTRADA: só os k_in pixels de maior intensidade
                # que dispararam neste timestep permanecem (ataca o gargalo fc1).
                idx_in = (s_in * x).topk(self.k_in, dim=1).indices
                mask_in = torch.zeros_like(s_in).scatter_(1, idx_in, 1.0)
                s_in = s_in * mask_in
            cur1 = self.fc1(s_in)
            spk1, mem1 = self.lif1(cur1, mem1)
            if self.k is not None:
                idx = mem1.topk(self.k, dim=1).indices
                mask = torch.zeros_like(spk1).scatter_(1, idx, 1.0)
                spk1 = spk1 * mask
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            out_sum = out_sum + spk2
            if count:
                # SynOps = (spikes pré-sinápticos) × (fan-out) por camada
                synops = synops + s_in.sum() * HID + spk1.sum() * OUT
        if count:
            return out_sum, (synops / B).item()
        return out_sum


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_loaders(batch: int, quick: bool):
    tf = transforms.Compose([transforms.ToTensor()])
    root = os.path.join(os.path.dirname(__file__), "data")
    train = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
    if quick:
        train = Subset(train, range(6000))
        test = Subset(test, range(2000))
    return (
        DataLoader(train, batch_size=batch, shuffle=True, num_workers=0),
        DataLoader(test, batch_size=batch, shuffle=False, num_workers=0),
    )


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def train(model, loader, device, epochs, is_snn):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def evaluate(model, loader, device, is_snn):
    model.eval()
    correct = total = 0
    synops_acc = 0.0
    nb = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_snn:
            out, so = model(x, count=True)
            synops_acc += so; nb += 1
        else:
            out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    acc = 100 * correct / total
    synops = synops_acc / nb if nb else None
    return acc, synops


@torch.no_grad()
def cpu_latency(model, is_snn, n=200):
    """Latência de inferência por amostra em CPU single-thread (ms)."""
    torch.set_num_threads(1)
    model = model.to("cpu").eval()
    x = torch.rand(1, 1, 28, 28)
    for _ in range(5):  # warmup
        model(x)
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    return (time.perf_counter() - t0) / n * 1000.0


def run_one(name, model, train_loader, test_loader, device, epochs, is_snn):
    train(model, train_loader, device, epochs, is_snn)
    acc, synops = evaluate(model, test_loader, device, is_snn)
    lat = cpu_latency(model, is_snn)
    cost = synops if is_snn else float(DenseMLP.macs)
    print(f"{name:22s} acc={acc:5.2f}%  cost/sample={cost:10.0f}  "
          f"({'SynOps' if is_snn else 'MACs'})  cpu_lat={lat:6.3f} ms")
    return dict(name=name, acc=acc, cost=cost, is_snn=is_snn, cpu_lat_ms=lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--T", type=int, default=25)
    ap.add_argument("--k", type=int, default=32, help="k-WTA temporal na hidden (de 256)")
    ap.add_argument("--k-in", type=int, default=None, help="k-WTA temporal na entrada (de 784); ataca o gargalo fc1")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    quick = os.environ.get("HEBB_QUICK") == "1"

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    print(f"device={dev}  T={args.T}  k={args.k}  epochs={args.epochs}  quick={quick}")
    train_loader, test_loader = get_loaders(args.batch, quick)

    results = []
    results.append(run_one("DenseMLP", DenseMLP().to(dev), train_loader, test_loader, dev, args.epochs, False))
    results.append(run_one("SNN-LIF (vanilla)", SNN(T=args.T).to(dev), train_loader, test_loader, dev, args.epochs, True))
    results.append(run_one(f"SNN+kWTA(k={args.k})", SNN(T=args.T, k=args.k).to(dev), train_loader, test_loader, dev, args.epochs, True))
    if args.k_in is not None:
        results.append(run_one(
            f"SNN+kWTAin(ki={args.k_in},k={args.k})",
            SNN(T=args.T, k=args.k, k_in=args.k_in).to(dev),
            train_loader, test_loader, dev, args.epochs, True))

    print("\n=== Efficiency vs dense baseline ===")
    dense = results[0]
    for r in results[1:]:
        syn_ratio = dense["cost"] / r["cost"]
        lat_ratio = r["cpu_lat_ms"] / dense["cpu_lat_ms"]
        verdict = "WORSE" if syn_ratio < 1 else f"{syn_ratio:.2f}x fewer ops"
        print(f"{r['name']:22s} SynOps vs MACs: {verdict:18s}  "
              f"CPU latency: {lat_ratio:5.2f}x dense  (acc {r['acc']:.2f} vs {dense['acc']:.2f})")


if __name__ == "__main__":
    main()
