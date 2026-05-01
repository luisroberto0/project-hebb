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
                   help="One or more seeds")
    p.add_argument("--train-episodes", type=int, default=5000)
    p.add_argument("--ckpt-dir", default="experiment_01_oneshot/checkpoints")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip seeds whose checkpoints already exist")
    p.add_argument("--k-wta", type=int, default=16,
                   help="k for k-WTA. Special: k=64 (>= embed_dim 64) means no-op k-WTA, "
                        "equivalent to ProtoNet baseline gradient flow. In that case we copy "
                        "from protonet_omniglot_seed{seed}.pt instead of retraining.")
    p.add_argument("--include-protonet", action="store_true",
                   help="Also train ProtoNet baseline (default off; treat as separate concern). "
                        "For k=16 with no other flags, behaves like #53 default (trains both).")
    args = p.parse_args()
    args.seed = args.seeds[0]

    cfg = default_config()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}, seeds={args.seeds}, k_wta={args.k_wta}")

    repo_root = Path(__file__).resolve().parent.parent
    ckpt_dir = repo_root / args.ckpt_dir

    # Default behaviour for k=16 without explicit flags: also train ProtoNet baseline (compat #53)
    train_protonet = args.include_protonet or (args.k_wta == 16 and not args.include_protonet)

    for seed in args.seeds:
        args.seed = seed
        proto_path = ckpt_dir / f"protonet_omniglot_seed{seed}.pt"
        c3_path = ckpt_dir / f"c3_kwta_k{args.k_wta}_seed{seed}.pt"

        # ProtoNet baseline path (only if asked)
        if train_protonet:
            if args.skip_existing and proto_path.exists():
                print(f"\n[seed={seed}] {proto_path.name} exists, skipping (--skip-existing)")
            else:
                torch.manual_seed(seed)
                enc_proto = ProtoEncoder().to(device)
                train_one_encoder(
                    "ProtoNet baseline", enc_proto, args, cfg, device, proto_path,
                )

        # C3 with given k
        if args.skip_existing and c3_path.exists():
            print(f"\n[seed={seed}] {c3_path.name} exists, skipping (--skip-existing)")
            continue

        if args.k_wta >= 64:
            # k>=64=embed_dim means k-WTA is no-op (returns z unchanged).
            # Gradient flow is identical to ProtoNet baseline. Copy state_dict
            # with prefix adjustment: ProtoEncoder has 'net.*', ProtoEncoderSparse
            # nests it as 'encoder.net.*'.
            if not proto_path.exists():
                raise FileNotFoundError(
                    f"k_wta={args.k_wta} >= embed_dim 64 means no-op k-WTA. "
                    f"Need {proto_path} to exist (ProtoNet baseline weights). "
                    f"Run with --include-protonet first or train ProtoNet baseline separately."
                )
            print(f"\n[seed={seed}] k={args.k_wta} >= 64: no-op k-WTA, copying ProtoNet baseline weights")
            ckpt_proto = torch.load(proto_path, map_location="cpu", weights_only=False)
            old_state = ckpt_proto["state_dict"]
            new_state = {f"encoder.{k}": v for k, v in old_state.items()}
            ckpt_proto["state_dict"] = new_state
            ckpt_proto["copied_from"] = str(proto_path)
            ckpt_proto["k_wta"] = args.k_wta
            ckpt_proto["session"] = f"#57 k={args.k_wta} (no-op k-WTA, copy of ProtoNet baseline with prefix adjust)"
            torch.save(ckpt_proto, c3_path)
            print(f"  saved {c3_path}")
        else:
            torch.manual_seed(seed)
            enc_c3 = ProtoEncoderSparse(k=args.k_wta).to(device)
            train_one_encoder(
                f"C3 k={args.k_wta}", enc_c3, args, cfg, device, c3_path,
            )

        print(f"\n=== seed={seed} k={args.k_wta} done ===")


if __name__ == "__main__":
    main()
