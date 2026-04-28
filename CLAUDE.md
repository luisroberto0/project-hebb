# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Project Hebb** is neuroscience-inspired AI research focused on building systems with capabilities LLMs lack: continual learning without catastrophic forgetting, genuine one-shot learning, radical efficiency (consumer CPU/GPU), and temporal reasoning via spiking neural networks (SNNs).

**Core principle:** This is research, not production code. 90% is conceptual iteration, 10% is experimentation. **Never optimize code before validating the idea.** Document failures more carefully than successes.

**Current phase:** Experiment 01 — One-shot learning on Omniglot using STDP (Spike-Timing-Dependent Plasticity) + Modern Hopfield Memory, without backpropagation.

## Essential Reading Order

1. **`CONTEXT.md`** — Mission, principles, architecture vision (read this first always)
2. **`PLAN.md`** — Current phase, active experiments, architectural decisions, iteration notes
3. **`experiment_01_oneshot/PLAN.md`** — Research question, hypothesis, 6-week roadmap
4. **`experiment_01_oneshot/WEEKLY-N.md`** — Weekly progress, failures, next steps

**Rule:** Always read `CONTEXT.md` + `PLAN.md` before any code changes. If they conflict, `CONTEXT.md` wins.

## Environment Setup

**Target hardware:** RTX 4070 Laptop GPU, CUDA 12.4, Windows 11

```powershell
# 1. Create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install PyTorch with CUDA first (critical order)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install remaining dependencies (note: causalnex may fail on Python 3.13+, skip if not needed)
pip install scipy matplotlib seaborn pandas scikit-learn tqdm einops snntorch brian2 brian2tools

# 4. Verify CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**Expected output:** `CUDA: True NVIDIA GeForce RTX 4070 Laptop GPU`

## Running Experiments

### Quick sanity check (MNIST STDP validation)

```bash
cd experiment_01_oneshot
python sanity_mnist.py --device cuda --epochs 1 --n-images 5000 --n-filters 100
```

**Success criteria:** ≥85% accuracy. If <85%, check `experiment_01_oneshot/WEEKLY-1.md` for diagnostic protocol.

### Full experiment pipeline (6 weeks automated)

```powershell
# Quick mode (debug, ~10 min)
$env:HEBB_QUICK="1"
pwsh -File experiment_01_oneshot/run_all.ps1

# Full mode (production, ~hours)
pwsh -File experiment_01_oneshot/run_all.ps1
```

**Outputs:** `logs/run_<timestamp>/`, `RESULTS.md`, `checkpoints/*.pt`

### Individual phases

```bash
cd experiment_01_oneshot

# Pretrain STDP on Omniglot background set
python train.py --n-images 24000 --epochs 1

# Evaluate N-way K-shot
python evaluate.py --checkpoint checkpoints/stdp_model.pt --ways 5 --shots 1 --episodes 1000

# Run baselines
python baselines.py --baseline pixel_knn --ways 5 --shots 1 --episodes 1000
python baselines.py --baseline proto_net --ways 5 --shots 1 --episodes 1000
```

## Architecture

**Two-system design** (cortex + hippocampus analogy):

```
Image → Poisson Encoding → Conv-STDP Layer 1 (LIF neurons)
                         → Conv-STDP Layer 2 (LIF neurons)
                         → Flatten + L2-norm
                         → Modern Hopfield Memory → Classification
```

**Key components:**

- **`config.py`** — Central hyperparameters (STDP, LIF, architecture). Edit here, not scattered across files.
- **`model.py`** — `ConvSTDPLayer` (vectorized STDP via PyTorch `F.unfold` + `einsum`), `HopfieldMemory`, full `STDPHopfieldModel`.
- **`data.py`** — Omniglot dataset + N-way K-shot episode sampler + Poisson spike encoding.
- **`train.py`** — Unsupervised STDP pretraining loop (no labels, no backprop).
- **`evaluate.py`** — Few-shot evaluation with episodic memory.
- **`baselines.py`** — Pixel kNN, Prototypical Networks for comparison.

## Critical Implementation Details

### STDP is NOT trained via backprop

**Weights have `requires_grad=False`.** Updates happen in `ConvSTDPLayer.stdp_update()` using local spike timing:

```python
# LTP (pre before post): Δw += A_pre * post_spikes · apre_trace
# LTD (post before pre): Δw += A_post * pre_spikes · apost_trace
```

Vectorization uses `F.unfold` to extract receptive field patches, then `einsum` to contract with spike traces. **Do not add `.backward()` calls.**

### Lateral inhibition (currently broken, see WEEKLY-1.md)

Current implementation applies weak weight penalties post-STDP. Diehl & Cook 2015 uses **membrane inhibition during LIF dynamics** (winner-take-all). Known issue documented in `experiment_01_oneshot/WEEKLY-1.md` Hypothesis 1.

### Configuration hierarchy

All hyperparameters live in `config.py` as dataclasses:
- `STDPConfig` — tau_pre/post, A_pre/post, w_min/max, lateral_inhibition
- `LIFConfig` — tau_mem, v_thresh, v_reset, refractory
- `ArchitectureConfig` — layer sizes, kernels, pooling
- `MemoryConfig` — Hopfield beta, distance metric

**Always edit `config.py`, never hardcode values.**

### Spike encoding

`data.py:poisson_encode()` converts images to spike trains:
- Input: `(B, C, H, W)` intensity [0,1]
- Output: `(T, B, C, H, W)` binary spikes
- Higher intensity → higher spike rate over T timesteps

## Development Workflow

### After making changes

1. **Run sanity check first** (cheap, fast feedback):
   ```bash
   python experiment_01_oneshot/sanity_mnist.py --n-images 1000
   ```

2. **If sanity passes, run quick Omniglot test**:
   ```bash
   $env:HEBB_QUICK="1"
   pwsh -File experiment_01_oneshot/run_all.ps1
   ```

3. **Document in appropriate WEEKLY-N.md** before committing.

### When experiments fail

1. **Do NOT immediately change code.** Diagnose first:
   - Features sparse? (% silent neurons)
   - Weights converged? (norm stable over time)
   - Filter visualizations make sense? (Gabor-like edges)

2. **Write hypotheses in WEEKLY-N.md** with:
   - Observed evidence
   - Probability ranking
   - Minimal experiment to test each

3. **Run cheapest test first** (e.g., check class distribution before retraining).

4. **Update `PLAN.md` § Notas de iteração** with date, observation, cause, decision.

### Committing

Follow existing commit style (see `git log`):
```
chore(experiment_01): brief description

Detailed multi-line explanation:
- What was tested
- Key metrics (time, accuracy, etc.)
- Problems found
- Next steps

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Always commit after completing a week/phase**, even (especially) if it failed.

## Known Issues & Current Status

**Status as of 2026-04-27:**
- Environment: ✅ Python 3.13 venv, PyTorch 2.6.0+cu124, RTX 4070 working
- Week 1 sanity check: ❌ FAILED — all 100 filters collapsed to class 0
- Root cause (likely): Lateral inhibition factor `1e-4` too weak + wrong implementation (should inhibit membrane, not weights)
- Next steps: Test class distribution → fix inhibition → iterate to ≥85%

**See `experiment_01_oneshot/WEEKLY-1.md` for full diagnostic.**

## Papers & References

Essential context (cited throughout code):
- **Diehl & Cook (2015)** — STDP unsupervised digit recognition (our baseline)
- **Ramsauer et al. (2020)** — Modern Hopfield Networks (episodic memory)
- **Lake et al. (2015)** — Omniglot benchmark, human-level concept learning
- **Snell et al. (2017)** — Prototypical Networks (baseline to beat)

## Architecture Decisions (Immutable)

Documented in `PLAN.md` § Decisões arquiteturais:

1. **STDP in PyTorch (not Brian2)** — GPU-first via vectorized `einsum`, tight integration. Cost to revert: ~1 week.
2. **Python scaffold (not Julia)** — Mature ecosystem (snntorch, Brian2). Port to Julia in Phase 2 if model validates. Cost: ~3-4 weeks.

**Do not reverse these without updating `PLAN.md` with full justification.**

## Dos and Don'ts

**DO:**
- Read `CONTEXT.md` + `PLAN.md` before coding
- Run sanity checks before full experiments
- Document failures in WEEKLY-N.md with hypotheses ranked by probability
- Use GPU (`--device cuda`) for all experiments
- Edit hyperparameters in `config.py`, not inline

**DON'T:**
- Add `.backward()` calls to STDP layers (no backprop!)
- Optimize code before validating the scientific idea
- Skip documentation when experiments fail
- Hardcode hyperparameters in scripts
- Create new experiments before completing Experiment 01

## File Reference

```
project-hebb/
├── CONTEXT.md              # Mission, principles (READ FIRST)
├── PLAN.md                 # Current phase, decisions, iteration log
├── requirements.txt        # Python dependencies
├── experiment_01_oneshot/
│   ├── PLAN.md            # 6-week roadmap, hypothesis, success criteria
│   ├── WEEKLY-{0-6}.md   # Weekly progress, diagnostics, decisions
│   ├── config.py          # Central hyperparameters (edit here)
│   ├── model.py           # ConvSTDPLayer, HopfieldMemory, full model
│   ├── data.py            # Omniglot + episode sampler + Poisson encoding
│   ├── train.py           # STDP pretraining (unsupervised)
│   ├── evaluate.py        # N-way K-shot evaluation
│   ├── baselines.py       # Pixel kNN, Prototypical Networks
│   ├── sanity_mnist.py    # Week 1 validation (Diehl & Cook reproduction)
│   ├── analysis.py        # Generate RESULTS.md from logs
│   ├── run_all.ps1        # Automated 6-week pipeline
│   └── utils/
│       └── visualize.py   # Filter/weight visualization tools
└── checkpoints/           # Saved models (.pt files, in .gitignore)
```
