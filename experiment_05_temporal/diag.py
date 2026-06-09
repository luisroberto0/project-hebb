"""Diagnóstico #72: SNN trava em ~13% (cego=48%). Instrumenta treino + taxas + logits."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch, torch.nn as nn
from shd_data import get_shd_loaders
from temporal_bench import SNN_FF, HID

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tr, te = get_shd_loaders(128, 100)
m = SNN_FF(gain=1.0).to(dev)

# taxa de disparo da hidden + magnitude dos logits (pós-init)
def probe(model, x):
    model.eval()
    with torch.no_grad():
        m1 = model.lif1.init_leaky(); rates = []; out = 0
        for t in range(x.size(1)):
            s1, m1 = model.lif1(model.bn1(model.fc1(model.gain * x[:, t])), m1)
            rates.append(s1.mean().item()); out = out + model.fc2(s1)
    return np.mean(rates), out

x, y = next(iter(te)); x, y = x.to(dev), y.to(dev)
r0, out0 = probe(m, x)
print(f"PRE-treino: hidden rate={r0:.3f}  logits abs.mean={out0.abs().mean():.2f} std={out0.std():.2f}")

# treina 2 epochs imprimindo loss
opt = torch.optim.Adam(m.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss()
for ep in range(2):
    m.train(); losses = []
    for xb, yb in tr:
        xb, yb = xb.to(dev), yb.to(dev)
        o = m(xb); loss = lf(o, yb)
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    print(f"  epoch {ep}: loss {np.mean(losses):.3f} (primeiro {losses[0]:.3f} -> ultimo {losses[-1]:.3f})")

r1, out1 = probe(m, x)
print(f"POS-treino: hidden rate={r1:.3f}  logits abs.mean={out1.abs().mean():.2f} std={out1.std():.2f}")
# grad norm da fc1 (o gradiente flui pela dinamica?)
o = m(x); loss = lf(o, y); m.zero_grad(); loss.backward()
gn = m.fc1.weight.grad.norm().item(); rn = m.fc1.weight.norm().item()
print(f"grad fc1 norm={gn:.5f}  weight norm={rn:.3f}  ratio={gn/rn:.5f}")
print(f"chance={100/20:.1f}%  cego~48%  | logits sao discriminativos? std deve crescer com treino")
