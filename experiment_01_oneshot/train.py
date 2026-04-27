"""
Pretreino STDP não-supervisionado no split background do Omniglot.

Loop:
  - Encoda imagem em spikes Poisson (T timesteps)
  - Forward por T timesteps com stdp_update ativo a cada timestep
  - Reseta traços entre imagens (cada amostra é episódio independente)
  - Loga estatísticas dos pesos pra TensorBoard

Uso:
    python train.py                                  # treino completo (24k imgs, 1 epoch)
    python train.py --n-images 500 --epochs 1        # debug rápido
    python train.py --epochs 3 --log-dir logs/run1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from config import default_config
from data import load_background
from model import STDPHopfieldModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-images", type=int, default=None,
                   help="Subset pra debug (ex.: 500). Default: usar tudo.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default="checkpoints/stdp_model.pt")
    p.add_argument("--log-dir", type=str, default="logs/train")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.n_images is not None:
        cfg.train.n_pretrain_images = args.n_images
    if args.device is not None:
        cfg.device = args.device

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    print(f"Dispositivo: {device}")
    torch.manual_seed(args.seed)

    # TensorBoard opcional — fail soft se não instalado
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)
        print(f"Logs TensorBoard em {args.log_dir}")
    except ImportError:
        print("(tensorboard não disponível — seguindo sem logging)")

    print("Carregando Omniglot background...")
    dataset = load_background(cfg)
    if cfg.train.n_pretrain_images < len(dataset):
        idx = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(args.seed)
        )[: cfg.train.n_pretrain_images].tolist()
        dataset = Subset(dataset, idx)
    print(f"  {len(dataset)} imagens no pretreino")

    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    model = STDPHopfieldModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros do modelo: {n_params}")

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nIniciando pretreino STDP por {cfg.train.epochs} epoch(s)...")
    t0 = time.time()
    seen = 0
    global_step = 0

    for epoch in range(cfg.train.epochs):
        for step, (img, _) in enumerate(loader):
            img = img.squeeze(1).to(device)  # (B, H, W) — Omniglot vem (1, H, W)
            with torch.no_grad():
                _ = model.extract_features(img, train_stdp=True)
                model.layer1.clip_weights()
                model.layer2.clip_weights()

            seen += img.shape[0]
            global_step += 1

            if step % cfg.train.log_every == 0:
                w1 = model.layer1.conv.weight.data
                w2 = model.layer2.conv.weight.data
                stats = {
                    "w1/mean": w1.mean().item(),
                    "w1/std": w1.std().item(),
                    "w2/mean": w2.mean().item(),
                    "w2/std": w2.std().item(),
                }
                elapsed = time.time() - t0
                print(f"  epoch {epoch} step {step:4d}  "
                      f"w1=μ{stats['w1/mean']:.3f}/σ{stats['w1/std']:.3f}  "
                      f"w2=μ{stats['w2/mean']:.3f}/σ{stats['w2/std']:.3f}  "
                      f"seen={seen}  elapsed={elapsed:.1f}s")
                if writer:
                    for k, v in stats.items():
                        writer.add_scalar(k, v, global_step)
                    writer.add_scalar("speed/imgs_per_sec", seen / elapsed, global_step)

            if step % cfg.train.save_every == 0 and step > 0:
                torch.save(model.state_dict(), args.checkpoint)

    torch.save(model.state_dict(), args.checkpoint)
    if writer:
        # Histogramas finais dos pesos
        writer.add_histogram("w1_final", model.layer1.conv.weight.data, 0)
        writer.add_histogram("w2_final", model.layer2.conv.weight.data, 0)
        writer.close()

    elapsed = time.time() - t0
    print(f"\nCheckpoint salvo em {args.checkpoint}")
    print(f"Tempo total: {elapsed:.1f}s ({seen} imagens, {seen/elapsed:.1f} imgs/s)")


if __name__ == "__main__":
    main()
