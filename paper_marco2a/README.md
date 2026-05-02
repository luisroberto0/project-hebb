# Paper Marco 2-A — Cross-domain k-WTA characterization

## Overview

Paper sobre Marco 2-A do `experiment_03_crossdomain/` (sessões #52-#57): caracterização empírica de k-WTA cross-domain Omniglot→CUB-200, demonstrando que o efeito de k-WTA documentado in-domain (paper C3) **colapsa em transfer extremo**.

Paper de exploração honesta com 3 achados centrais:

1. **k-WTA effect collapse:** spread Omniglot in-domain entre k=8 e k=64 = 3.78 p.p.; spread CUB cross-domain = 0.52 p.p. (ruído).
2. **Anti-transfer characterization:** encoder bio-inspirado treinado em fonte distante é estatisticamente indistinguível de random encoder cross-domain (delta +0.18 p.p., ICs sobrepostos).
3. **Bottleneck decomposition:** treino na target (+12.22 p.p.) e resolução adequada (+15.53 p.p.) são gargalos comparáveis pra cross-domain few-shot extreme.

## Title (tentative)

**"When Sparsity Stops Mattering: k-WTA Effect Collapse Under Extreme Domain Shift"**

Alternativas em `outline.md`. Decisão final pós-draft completo.

## Target venue (a definir)

Decisão fica pra Luis após draft completo. Candidatos prováveis:
- NeurIPS Bio-Plausible Learning Workshop (~setembro 2026)
- ICLR Workshop on Mathematical and Empirical Understanding of Foundation Models
- LinkedIn-only (consistente com decisão paper C3 pós-#36)

## Author

**Luis Roberto Pinho da Silva Junior** — Independent research
[LinkedIn: luisroberto0](https://www.linkedin.com/in/luisroberto0/)

## Status

| Sessão | Conteúdo | Status |
|---|---|---|
| #58 | Outline + Intro + Background drafted | em curso |
| #59 | Methods + Experiments | pendente |
| #60 | Figures + tabelas | pendente |
| #61 | Discussion | pendente |
| #62 | Conclusion + abstract + refinement | pendente |
| #63 | LaTeX consolidação + bibliography final | pendente |
| #64-65 | Peer review interno + revisão | pendente |
| #66 | Admin obrigatória — decide próximo passo | pendente |

## Files

- `outline.md` — estrutura detalhada com checkboxes ✅
- `intro.md` — Section 1: Introduction (✅ draft #58)
- `background.md` — Section 2: Background (✅ draft #58)
- `methods.md` — Section 3: Method (placeholder, draft #59)
- `experiments.md` — Section 4: Experiments (placeholder, draft #59-#60)
- `discussion.md` — Section 5: Discussion (placeholder, draft #61)
- `conclusion.md` — Section 6: Conclusion (placeholder, draft #62)
- `refs.bib` — Bibliography (BibTeX, ~18 entradas)
- `main.tex` — LaTeX consolidado (a criar #63)
- `figs/` — figuras 300 DPI (a criar #60)

## Source data

Tudo está em `experiment_03_crossdomain/`:

- `WEEKLY-1.md` — 7 condições caracterizadas com IC95% bootstrap (sessões #52-#57)
- `PAPERS.md` — lit review de 5 papers core (Triantafillou 2020, Tseng 2020, Chen 2019, Phoo & Hariharan 2021, Wah 2011)
- Scripts reproduzíveis: `train_encoders.py`, `train_cub_protonet.py`, `eval_crossdomain.py`, `eval_pixel_knn.py`, `eval_random_encoder.py`
- Checkpoints em `experiment_01_oneshot/checkpoints/` (gitignored, reproduzível via scripts)

## Honest scoping

Este paper **NÃO claim**:
- Método novo cross-domain
- Biological realism
- Breakthrough mecanístico
- Generalization a outros pares source/target

Este paper **claim**:
- Caracterização empírica rigorosa de quando k-WTA importa e quando não
- Achado mecanístico negativo defensável (anti-transfer em extreme task differences)
- Decomposição quantitativa de gargalos cross-domain

Limitações documentadas em §5: single source dataset (Omniglot), single target dataset (CUB-200), CNN-4 architecture (não testou ResNet, ViT), 28×28 grayscale como input principal.

## Tom

Honesto, direto. Sem "novel", "breakthrough", "first to". Linguagem técnica precisa, em inglês, consistente com paper C3 main.tex.

Workshop-scope, não conference. Achado negativo bem caracterizado vale mais que claim positivo inflado.

## Cronograma estimado pós-#58

| Sessão | Conteúdo | Tempo est. |
|---|---|---|
| #59 | Methods + Experiments draft | 2-3 h |
| #60 | Figures + tabelas | 1-2 h |
| #61 | Discussion | 2-3 h |
| #62 | Conclusion + abstract + refinement | 1-2 h |
| #63 | LaTeX consolidação | 1-2 h |
| #64-65 | Peer review interno | 1-2 h cada |
| **Total** | **6-7 sessões pós-#58** | **8-15 h** |

Dentro do limite hard de 15 sessões Marco 2-A (#52-#66) — sobram 8 sessões pós-#58, mais que suficiente.
