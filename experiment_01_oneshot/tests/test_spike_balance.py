"""
Mede empiricamente a razão pré-spikes : pós-spikes durante pretreino STDP
com k=1 WTA. Objetivo: calibrar A_pre/A_post pra que LTP e LTD se balanceiem.

Hipótese: razão R = pré/pós é grande (>10), e como STDP usa
  Δw_LTP ∝ A_pre × pos_spikes
  Δw_LTD ∝ A_post × pre_spikes
o net effect com A_pre=0.01, A_post=-0.0105 (magnitude similar) é dominado
por LTD por um fator ~R. Pra balancear, |A_post| precisa ser ~A_pre / R.

Uso:
    cd experiment_01_oneshot && python tests/test_spike_balance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import default_config
from sanity_mnist import SanityNet
from data import poisson_encode


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    cfg = default_config()
    cfg.spike.timesteps = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    idx = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(42))[:100].tolist()
    train_ds = Subset(train_ds, idx)
    loader = DataLoader(train_ds, batch_size=16, shuffle=False)

    net = SanityNet(cfg, n_filters=100).to(device)
    print(f"Filtros: {net.n_filters}, kernel: {net.layer.kernel_size}")
    print(f"v_thresh={cfg.lif.v_thresh}, max_rate={cfg.spike.max_rate_hz}Hz, T={cfg.spike.timesteps}")

    pre_total, post_total, n_imgs = 0.0, 0.0, 0
    weights_per_image = []

    print("\nMedindo razão pré:pós em 100 imagens com k=1 WTA...")
    t0 = time.time()
    for img, _ in loader:
        img = img.to(device)
        B = img.shape[0]
        T = cfg.spike.timesteps

        spikes_in = poisson_encode(img, T, cfg.spike.max_rate_hz, cfg.spike.dt_ms)  # (T, B, 1, 28, 28)
        mem = torch.zeros(B, net.n_filters, 1, 1, device=device)

        pre_count, post_count = 0.0, 0.0
        net.layer.apre = None
        net.layer.apost = None

        for t in range(T):
            pre = spikes_in[t]
            spk, mem = net.layer(pre, mem)
            net.layer.stdp_update(pre, spk)
            pre_count += pre.sum().item()
            post_count += spk.sum().item()

        pre_total += pre_count
        post_total += post_count
        n_imgs += B
        weights_per_image.append(net.layer.conv.weight.data.mean().item())

    elapsed = time.time() - t0
    avg_pre = pre_total / n_imgs
    avg_post = post_total / n_imgs
    ratio = avg_pre / max(avg_post, 1e-9)

    print(f"\nResultados ({n_imgs} imagens, {elapsed:.1f}s):")
    print(f"  Pré-spikes média/imagem:  {avg_pre:>10.1f}")
    print(f"  Pós-spikes média/imagem:  {avg_post:>10.1f}")
    print(f"  Razão R = pré/pós:        {ratio:>10.2f}")
    print(f"\nPesos durante o treino (mean):")
    print(f"  Início: {weights_per_image[0]:.4f}")
    print(f"  Meio:   {weights_per_image[len(weights_per_image)//2]:.4f}")
    print(f"  Fim:    {weights_per_image[-1]:.4f}")
    if weights_per_image[-1] < weights_per_image[0]:
        print(f"  DECRESCENDO: pesos perderam {(1 - weights_per_image[-1]/weights_per_image[0])*100:.1f}%")

    print(f"\nA_pre atual:  {default_config().stdp.A_pre:.6f}")
    print(f"A_post atual: {default_config().stdp.A_post:.6f}")
    print(f"Razão |A_post|/A_pre atual: {abs(default_config().stdp.A_post)/default_config().stdp.A_pre:.4f}")

    print(f"\n--- RECOMENDAÇÃO ---")
    print(f"Pra balancear LTP·post == LTD·pre, precisa A_pre/|A_post| = {ratio:.2f}")
    print(f"Sugestão 1 (manter A_pre=0.01): A_post = {-0.01/ratio:.6f}")
    print(f"Sugestão 2 (manter |A_post|=0.0105): A_pre = {0.0105*ratio:.6f}")
    print(f"Sugestão 3 (split do meio): A_pre = {0.005*ratio**0.5:.6f}, A_post = {-0.005/ratio**0.5:.6f}")


if __name__ == "__main__":
    main()
