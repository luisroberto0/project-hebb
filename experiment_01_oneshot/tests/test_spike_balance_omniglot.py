"""
Mede empiricamente a razão pré:pós-spikes nas DUAS layers do
STDPHopfieldModel quando alimentado com Omniglot.

Diferente de test_spike_balance.py (que usa kernel=28 sobre MNIST), aqui
testamos a arquitetura conv real (kernel=5 + pool, 2 layers) que vai ser
usada na Semana 2.

Hipótese (a confirmar): o regime de spikes é radicalmente diferente do
MNIST kernel=28 — k=1 WTA por POSIÇÃO espacial gera muito mais pós-spikes
por imagem do que o WTA global do MNIST sanity. Isso pode mudar
fundamentalmente o balanço LTP/LTD e exigir recalibração.

Uso:
    cd experiment_01_oneshot && python tests/test_spike_balance_omniglot.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, Subset

from config import default_config
from data import load_background, encode
from model import STDPHopfieldModel


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    cfg = default_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Config arquitetura: layer1={cfg.arch.conv1_filters}f/k{cfg.arch.conv1_kernel}, "
          f"pool{cfg.arch.conv1_pool}, layer2={cfg.arch.conv2_filters}f/k{cfg.arch.conv2_kernel}, "
          f"pool{cfg.arch.conv2_pool}")
    print(f"Config STDP: A_pre={cfg.stdp.A_pre}, A_post={cfg.stdp.A_post}, "
          f"theta_plus={cfg.stdp.theta_plus}")

    print("\nCarregando Omniglot background...")
    dataset = load_background(cfg)
    idx = torch.randperm(len(dataset),
                         generator=torch.Generator().manual_seed(42))[:100].tolist()
    dataset = Subset(dataset, idx)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    model = STDPHopfieldModel(cfg).to(device)
    print(f"Parâmetros: {sum(p.numel() for p in model.parameters())}")

    pre1_total, post1_total = 0.0, 0.0
    pre2_total, post2_total = 0.0, 0.0
    n_imgs = 0
    weights1_history = []
    weights2_history = []

    print("\nMedindo razão pré:pós em 100 imagens (com STDP ativo, igual ao train.py)...")
    t0 = time.time()
    for img, _ in loader:
        img = img.squeeze(1).to(device)  # (B, H, W)
        B, H, W = img.shape
        T = cfg.spike.timesteps

        x = img.unsqueeze(1)
        spikes_in = encode(x, cfg)

        out1_shape, out2_shape, pool1_shape, pool2_shape = model._compute_shapes((H, W))
        mem1 = torch.zeros((B,) + out1_shape, device=device)
        mem2 = torch.zeros((B,) + out2_shape, device=device)

        # Reset traços (igual extract_features com train_stdp=True)
        model.layer1.apre = None; model.layer1.apost = None
        model.layer2.apre = None; model.layer2.apost = None

        pre1_count, post1_count = 0.0, 0.0
        pre2_count, post2_count = 0.0, 0.0

        with torch.no_grad():
            for t in range(T):
                pre1 = spikes_in[t]
                spk1, mem1 = model.layer1(pre1, mem1)
                model.layer1.stdp_update(pre1, spk1)

                pre1_count += pre1.sum().item()
                post1_count += spk1.sum().item()

                spk1_pool = model.pool1(spk1)
                spk2, mem2 = model.layer2(spk1_pool, mem2)
                model.layer2.stdp_update(spk1_pool, spk2)

                pre2_count += spk1_pool.sum().item()
                post2_count += spk2.sum().item()

        pre1_total += pre1_count
        post1_total += post1_count
        pre2_total += pre2_count
        post2_total += post2_count
        n_imgs += B

        weights1_history.append(model.layer1.conv.weight.data.mean().item())
        weights2_history.append(model.layer2.conv.weight.data.mean().item())

    elapsed = time.time() - t0

    avg_pre1 = pre1_total / n_imgs
    avg_post1 = post1_total / n_imgs
    R1 = avg_pre1 / max(avg_post1, 1e-9)

    avg_pre2 = pre2_total / n_imgs
    avg_post2 = post2_total / n_imgs
    R2 = avg_pre2 / max(avg_post2, 1e-9)

    print(f"\n=== Resultados ({n_imgs} imagens, {elapsed:.1f}s) ===\n")
    print(f"Layer 1 (input 1×28×28 → output {cfg.arch.conv1_filters}×28×28):")
    print(f"  Pré-spikes/imagem (Poisson da imagem):  {avg_pre1:>10.1f}")
    print(f"  Pós-spikes/imagem (após k=1 WTA):       {avg_post1:>10.1f}")
    print(f"  Razão R1 = pré/pós:                     {R1:>10.3f}")
    print()
    print(f"Layer 2 (input {cfg.arch.conv1_filters}×14×14 → output {cfg.arch.conv2_filters}×14×14):")
    print(f"  Pré-spikes/imagem (de spk1_pool):       {avg_pre2:>10.1f}")
    print(f"  Pós-spikes/imagem (após k=1 WTA):       {avg_post2:>10.1f}")
    print(f"  Razão R2 = pré/pós:                     {R2:>10.3f}")
    print()
    print(f"Pesos durante 100 imagens:")
    print(f"  Layer 1: início={weights1_history[0]:.4f}, fim={weights1_history[-1]:.4f}, "
          f"Δ={weights1_history[-1]-weights1_history[0]:+.4f}")
    print(f"  Layer 2: início={weights2_history[0]:.4f}, fim={weights2_history[-1]:.4f}, "
          f"Δ={weights2_history[-1]-weights2_history[0]:+.4f}")

    print(f"\n=== Comparação com regimes anteriores ===\n")
    print(f"  MNIST kernel=28 (Semana 1):    R = 10.1 (LTD-dominante, pesos morrem com paper)")
    print(f"  Paper Diehl&Cook 2015:         R ≈ 1   (calibrado pra esse regime)")
    print(f"  Omniglot kernel=5 layer1:      R = {R1:.2f}")
    print(f"  Omniglot kernel=5 layer2:      R = {R2:.2f}")

    print(f"\n=== Recomendações pra balancear LTP·post ≈ |LTD|·pre ===\n")
    print(f"Critério: A_pre/|A_post| ≈ R (mantém produto efetivo equilibrado)")
    print()
    print(f"Layer 1 (R={R1:.3f}):")
    print(f"  Sugestão (manter A_pre=0.01): A_post = {-0.01/R1:.6f}")
    print(f"  Sugestão (manter |A_post|=0.0105): A_pre = {0.0105*R1:.6f}")
    print()
    print(f"Layer 2 (R={R2:.3f}):")
    print(f"  Sugestão (manter A_pre=0.01): A_post = {-0.01/R2:.6f}")
    print(f"  Sugestão (manter |A_post|=0.0105): A_pre = {0.0105*R2:.6f}")
    print()
    print(f"NOTA: STDPConfig é compartilhado entre layers. Se R1 e R2 divergem muito,")
    print(f"pode ser necessário separar configs por layer no futuro. Por ora, calibrar")
    print(f"pelo regime mais crítico (layer 1, que recebe input bruto).")


if __name__ == "__main__":
    main()
