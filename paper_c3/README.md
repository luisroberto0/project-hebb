# Paper C3 — Workshop submission

## Overview

Paper sobre o resultado C3 do `experiment_01_oneshot/` (sessão #20): k-WTA esparso aplicado ao embedding final de Prototypical Networks preserva alta acurácia em few-shot Omniglot — empirical study sobre quanto sparsity ProtoNet tolera.

## Title (tentative)

**k-WTA Sparsity Preserves Prototypical Network Performance in Few-Shot Learning**

## Target venue

NeurIPS 2026 Workshop on Bio-Plausible Learning (submissão típica setembro 2026).

Backups se workshop específico não existir em 2026:
- ICML AI for Science Workshop
- NESY (Neuro-Symbolic) Workshop
- CCN (Cognitive Computational Neuroscience)

## Author

Luis Roberto — [LinkedIn](https://www.linkedin.com/in/luisroberto0/)

## Status

| Sessão | Conteúdo | Status |
|---|---|---|
| #31 (esta) | Outline + Introduction + Background + refs.bib inicial | em progresso |
| #32 | Methods + Experiments | TODO |
| #33 | Discussion + revisão geral | TODO |
| #34 | Figures refinement + bibliography final + LaTeX conversion | TODO |
| #35 | Peer review interno + revisão final | TODO |

## Files

- `outline.md` — estrutura detalhada de cada seção
- `intro.md` — Section 1: Introduction
- `background.md` — Section 2: Background
- `methods.md` — Section 3: Method (TODO #32)
- `experiments.md` — Section 4: Experiments (TODO #32)
- `discussion.md` — Section 5: Discussion (TODO #33)
- `conclusion.md` — Section 6: Conclusion (TODO #33)
- `appendix.md` — Apêndice opcional sobre Marco 1 (TODO #33)
- `refs.bib` — Bibliography (BibTeX)
- `figs/` — Figuras (TODO #34, gerar do `experiment_01_oneshot/c3_protonet_sparse.py` outputs)

## Honest scoping

C3 é **incrementalismo empírico defensável**, não breakthrough mecanístico. Paper precisa refletir isso pra evitar overclaim que reviewer rejeita:

- ✅ Claim aceitável: "demonstramos empiricamente que k-WTA esparso preserva ProtoNet"
- ❌ Claim a evitar: "novel architecture", "breakthrough", "first to combine X with Y"

Ver `STRATEGY.md` "Plano paper C3 — Workshop NeurIPS Bio-Plausible Learning" para framing completo e cronograma estimado.

## Source experiments

Tudo já está rodado, sem necessidade de re-rodar:

- `experiment_01_oneshot/c3_protonet_sparse.py` — script principal (sessão #20)
- `experiment_01_oneshot/baselines.py` — ProtoNet baseline (94.55%)
- `experiment_01_oneshot/WEEKLY-2.md` sessão #20 — resultados completos com tabelas

Reproduzível via:
```bash
cd experiment_01_oneshot
python c3_protonet_sparse.py --device cuda --train-episodes 5000 --eval-eps 1000 --seed 42
```
