"""
Sessão #53 — Smoke test pipeline cross-domain CUB-200.

Objetivo: validar que o pipeline cross-domain fecha. Não é teste estatístico —
é teste de que CUB carrega, dataloader produz shape correto, encoder C3 frozen
faz forward sem erro, distâncias query→prototype são numéricas, accuracy está
no range válido.

NÃO é teste do método C3. ACC de 1 episódio é qualitativo.

Uso:
    python smoke_test.py --device cuda
    python smoke_test.py --device cuda --encoder protonet  # baseline ProtoNet em vez de C3
    python smoke_test.py --device cuda --use-kwta          # aplica k-WTA k=16 no embedding
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment_01_oneshot"))

from cub_data import CUBDataset
from episodes import CUBEpisodeSampler, proto_episode_eval
from c3_protonet_sparse import ProtoEncoder, ProtoEncoderSparse


SANITY_BOUNDS = {
    "acc_min": 0.0,
    "acc_max": 1.0,
    "embed_dim_expected": 64,  # CNN-4 com 28x28 input → 1x1 spatial × 64 channels
}


def load_encoder(encoder_kind: str, ckpt_path: Path, use_kwta: bool, k_wta: int,
                  device: torch.device):
    """Loads encoder with frozen weights from checkpoint."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    if encoder_kind == "c3":
        encoder = ProtoEncoderSparse(k=k_wta).to(device)
    elif encoder_kind == "protonet":
        encoder = ProtoEncoder().to(device)
        if use_kwta:
            # Wrap in ProtoEncoderSparse for k-WTA — but state_dict matches plain ProtoEncoder.
            # To keep things simple: load into ProtoEncoder, then wrap.
            encoder.load_state_dict(state)
            wrapped = ProtoEncoderSparse(k=k_wta).to(device)
            wrapped.encoder.load_state_dict(state)
            encoder = wrapped
            state = None  # already loaded
    else:
        raise ValueError(f"unknown encoder_kind: {encoder_kind}")

    if state is not None:
        encoder.load_state_dict(state)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, ckpt


def run_sanity_checks(diagnostics: dict, acc: float, n_way: int) -> tuple[bool, list[str]]:
    """Returns (all_pass, list_of_failures)."""
    failures = []

    if not (SANITY_BOUNDS["acc_min"] <= acc <= SANITY_BOUNDS["acc_max"]):
        failures.append(f"acc={acc:.4f} out of [0, 1]")

    sup_d = diagnostics["sup_emb_shape"][-1]
    qry_d = diagnostics["qry_emb_shape"][-1]
    proto_d = diagnostics["protos_shape"][-1]
    if not (sup_d == qry_d == proto_d == SANITY_BOUNDS["embed_dim_expected"]):
        failures.append(
            f"embed dim mismatch: sup={sup_d} qry={qry_d} proto={proto_d} "
            f"expected={SANITY_BOUNDS['embed_dim_expected']}"
        )

    if diagnostics["preds_min"] < 0 or diagnostics["preds_max"] >= n_way:
        failures.append(
            f"preds out of [0, {n_way-1}]: min={diagnostics['preds_min']} max={diagnostics['preds_max']}"
        )

    if diagnostics["dists_mean"] == 0.0 or diagnostics["dists_std"] == 0.0:
        failures.append(
            f"degenerate distances: mean={diagnostics['dists_mean']} "
            f"std={diagnostics['dists_std']}"
        )

    if diagnostics["sup_emb_norm_mean"] == 0.0:
        failures.append("sup embedding norm is zero (encoder may be silent)")

    return len(failures) == 0, failures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--encoder", choices=("c3", "protonet"), default="c3")
    p.add_argument("--ckpt", default=None,
                   help="Path to checkpoint. Default depends on --encoder.")
    p.add_argument("--use-kwta", action="store_true",
                   help="Apply k-WTA k=16 over embeddings (only matters with --encoder protonet)")
    p.add_argument("--k-wta", type=int, default=16)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=1)
    p.add_argument("--n-query", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[smoke #53] Device: {device}")

    repo_root = Path(__file__).resolve().parent.parent
    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
    elif args.encoder == "c3":
        ckpt_path = repo_root / "experiment_01_oneshot" / "checkpoints" / f"c3_kwta_k16_seed{args.seed}.pt"
    else:
        ckpt_path = repo_root / "experiment_01_oneshot" / "checkpoints" / f"protonet_omniglot_seed{args.seed}.pt"

    print(f"[smoke #53] Loading encoder='{args.encoder}' from {ckpt_path.name} (frozen)")
    t0 = time.time()
    encoder, ckpt_meta = load_encoder(args.encoder, ckpt_path, args.use_kwta, args.k_wta, device)
    n_params = sum(p.numel() for p in encoder.parameters())
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"[smoke #53] encoder params total={n_params}, trainable={n_trainable} (must be 0)")
    if n_trainable != 0:
        raise RuntimeError(f"Encoder has {n_trainable} trainable params; should be frozen")

    print(f"[smoke #53] Loading CUB-200 test split (28x28 grayscale)...")
    dataset = CUBDataset(split="test", verbose=True)

    print(f"[smoke #53] Sampling 1 episode {args.n_way}w{args.k_shot}s{args.n_query}q seed={args.seed}")
    sampler = CUBEpisodeSampler(
        dataset, n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query, seed=args.seed,
    )
    episode = sampler.sample()
    print(f"[smoke #53] support shape={tuple(episode.support.shape)} "
          f"query shape={tuple(episode.query.shape)}")
    print(f"[smoke #53] support classes (mapped): "
          f"{episode.support_labels.unique().tolist()}")

    acc, diag = proto_episode_eval(encoder, episode, device)
    elapsed = time.time() - t0

    print(f"\n[smoke #53] CUB {args.n_way}w{args.k_shot}s 1 episode: ACC={acc*100:.2f}% "
          f"(encoder={args.encoder} frozen, input shape=(1,28,28) grayscale, t={elapsed:.1f}s)")
    print(f"[smoke #53] diagnostics:")
    for k, v in diag.items():
        print(f"    {k}: {v}")

    all_pass, failures = run_sanity_checks(diag, acc, args.n_way)
    print(f"\n[smoke #53] sanity checks: {'ALL PASS' if all_pass else 'FAILURES'}")
    for f in failures:
        print(f"    [FAIL] {f}")

    chance = 1.0 / args.n_way
    print(f"\n[smoke #53] interpretation:")
    print(f"    chance = {chance*100:.1f}%")
    print(f"    1-episode ACC is qualitative — not statistical evidence either way")
    print(f"    pipeline functional if all sanity checks pass and ACC is finite")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
