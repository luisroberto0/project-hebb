"""
Marco 2-C (#72) — benchmark de raciocínio temporal em SHD:
baseline cego ao timing (MLP sobre histograma) vs SNN feedforward vs SNN recorrente.

Mede acurácia test. `SNN_rec − cego` = contribuição do timing; `rec − ff` = contribuição
da dinâmica recorrente. NÃO é o experimento final (5 seeds) — é harness + smoke.

Uso:
    python temporal_bench.py --epochs 3 --bins 100 --device cuda
"""
from __future__ import annotations
import argparse, os, sys, time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from shd_data import get_shd_loaders, bag_of_spikes, N_UNITS, N_CLASSES

import snntorch as snn
from snntorch import surrogate

HID = 256
BETA = 0.9


class BlindMLP(nn.Module):
    """Baseline CEGO ao timing: histograma de spikes por canal -> MLP. Ignora a ordem temporal."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_UNITS, HID), nn.ReLU(), nn.Linear(HID, N_CLASSES))

    def forward(self, x):  # x: (B, T, U)
        return self.net(bag_of_spikes(x))


class SNN_FF(nn.Module):
    """SNN feedforward LIF — usa a sequência temporal mas sem recorrência.

    gain escala a corrente de entrada (diag #72: corrente crua ~0.04 << threshold 1.0,
    hidden sub-ativada 3.7% -> gradiente fraco). BatchNorm1d na fc1 estabiliza a escala.
    """
    def __init__(self, gain=1.0):
        super().__init__()
        sg = surrogate.fast_sigmoid()
        self.gain = gain
        self.fc1 = nn.Linear(N_UNITS, HID); self.bn1 = nn.BatchNorm1d(HID, track_running_stats=False)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=sg)
        self.fc2 = nn.Linear(HID, N_CLASSES)  # readout integrador (logits contínuos)

    def forward(self, x):  # (B, T, U)
        m1 = self.lif1.init_leaky()
        out = 0
        for t in range(x.size(1)):
            s1, m1 = self.lif1(self.bn1(self.fc1(self.gain * x[:, t])), m1)
            out = out + self.fc2(s1)
        return out


class SNN_Rec(nn.Module):
    """SNN recorrente — recorrência all-to-all na hidden. Explora a DINÂMICA temporal."""
    def __init__(self, gain=1.0):
        super().__init__()
        sg = surrogate.fast_sigmoid()
        self.gain = gain
        self.fc1 = nn.Linear(N_UNITS, HID); self.bn1 = nn.BatchNorm1d(HID, track_running_stats=False)
        self.rec = nn.Linear(HID, HID, bias=False)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=sg)
        self.fc2 = nn.Linear(HID, N_CLASSES)  # readout integrador (logits contínuos)

    def forward(self, x):  # (B, T, U)
        B = x.size(0)
        m1 = self.lif1.init_leaky()
        s1 = torch.zeros(B, HID, device=x.device)
        out = 0
        for t in range(x.size(1)):
            cur1 = self.bn1(self.fc1(self.gain * x[:, t])) + self.rec(s1)
            s1, m1 = self.lif1(cur1, m1)
            out = out + self.fc2(s1)
        return out


def train(model, loader, device, epochs):
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
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100 * correct / total


def run(name, model, tr, te, device, epochs):
    t0 = time.time()
    train(model, tr, device, epochs)
    acc = evaluate(model, te, device)
    print(f"{name:18s} acc={acc:5.2f}%   ({time.time()-t0:.0f}s)")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--gain", type=float, default=1.0, help="escala da corrente de entrada (BN ja normaliza)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    print(f"device={dev}  bins={args.bins}  epochs={args.epochs}")
    print("carregando SHD (train+test)...")
    tr, te = get_shd_loaders(args.batch, args.bins)
    print(f"  train batches={len(tr)}  test batches={len(te)}")

    accs = {}
    accs["cego"] = run("BlindMLP (cego)", BlindMLP().to(dev), tr, te, dev, args.epochs)
    accs["ff"]   = run("SNN feedforward", SNN_FF(gain=args.gain).to(dev), tr, te, dev, args.epochs)
    accs["rec"]  = run("SNN recorrente", SNN_Rec(gain=args.gain).to(dev), tr, te, dev, args.epochs)

    print("\n=== margens (critério: rec-cego >=10pp E rec >=65%) ===")
    print(f"  timing (rec - cego):      {accs['rec']-accs['cego']:+.2f} p.p.")
    print(f"  recorrência (rec - ff):   {accs['rec']-accs['ff']:+.2f} p.p.")
    print(f"  SNN recorrente absoluto:  {accs['rec']:.2f}% (chance 5%)")


if __name__ == "__main__":
    main()
