"""
Sessão #53 — Treina e salva encoders ProtoNet baseline e C3 (k-WTA k=16) em Omniglot.

Necessário porque c3_protonet_sparse.py (sessão #20) não salvou checkpoints.
Marco 2-A precisa dos encoders treinados em Omniglot pra eval cross-domain em CUB-200.

Não modifica c3_protonet_sparse.py — apenas importa as classes e o loop de treino,
salvando state_dict ao final.

Reproduz exatamente o setup da sessão #20:
- 5000 episodes 5w1s 5q
- Adam lr=1e-3
- seed=42
- Encoder ProtoNet (CNN-4)
- C3b: ProtoNet + k-WTA k=16 (75% sparsity)

Uso:
    python train_encoders.py --device cuda
    # Outputs:
    #   experiment_01_oneshot/checkpoints/protonet_omniglot_seed42.pt
    #   experiment_01_oneshot/checkpoints/c3_kwta_k16_seed42.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Imports do experiment_01 (não modifica scripts originais)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment_01_oneshot"))
from config import default_config
from data import load_background, EpisodeSampler
from c3_protonet_sparse import ProtoEncoder, ProtoEncoderSparse, proto_episode_loss


def train_one_encoder(name: str, encoder, args, cfg, device, save_path: Path):
    print(f"\n{'='*72}\n[{name}] Training (5w1s, {args.train_episodes} eps, seed={args.seed})\n{'='*72}")

    train_ds = load_background(cfg)
    sampler = EpisodeSampler(train_ds, n_way=5, k_shot=1, n_query=5, seed=args.seed)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    encoder.train()

    t0 = time.time()
    log_every = max(1, args.train_episodes // 10)
    for step in range(args.train_episodes):
        ep = sampler.sample()
        loss, acc = proto_episode_loss(encoder, ep, n_classes=5, device=device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"  [{name}] step {step+1:5d}/{args.train_episodes}  loss={loss.item():.3f}  acc={acc*100:.1f}%  ({elapsed:.1f}s)")

    train_time = time.time() - t0
    print(f"  [{name}] training done in {train_time:.1f}s")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": encoder.state_dict(),
        "name": name,
        "seed": args.seed,
        "train_episodes": args.train_episodes,
        "n_way": 5,
        "k_shot": 1,
        "n_query": 5,
        "train_time_seconds": train_time,
        "session": "#53 (replicating #20 setup)",
    }, save_path)
    print(f"  [{name}] saved to {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="One or more seeds; trains both encoders per seed")
    p.add_argument("--train-episodes", type=int, default=5000)
    p.add_argument("--ckpt-dir", default="experiment_01_oneshot/checkpoints")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip seeds whose checkpoints already exist")
    args = p.parse_args()
    # Backward-compat single-arg internal access
    args.seed = args.seeds[0]

    cfg = default_config()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}, seeds={args.seeds}")

    repo_root = Path(__file__).resolve().parent.parent
    ckpt_dir = repo_root / args.ckpt_dir

    for seed in args.seeds:
        args.seed = seed
        proto_path = ckpt_dir / f"protonet_omniglot_seed{seed}.pt"
        c3_path = ckpt_dir / f"c3_kwta_k16_seed{seed}.pt"

        if args.skip_existing and proto_path.exists() and c3_path.exists():
            print(f"\n[seed={seed}] both checkpoints exist, skipping (--skip-existing)")
            continue

        # ProtoNet baseline (sem k-WTA)
        torch.manual_seed(seed)
        enc_proto = ProtoEncoder().to(device)
        train_one_encoder(
            "ProtoNet baseline", enc_proto, args, cfg, device, proto_path,
        )

        # C3b: ProtoNet + k-WTA k=16 (75% sparsity)
        torch.manual_seed(seed)
        enc_c3 = ProtoEncoderSparse(k=16).to(device)
        train_one_encoder(
            "C3b k=16 (75% sparse)", enc_c3, args, cfg, device, c3_path,
        )

        print(f"\n=== seed={seed} done ===")
        print(f"  {proto_path}")
        print(f"  {c3_path}")


if __name__ == "__main__":
    main()
