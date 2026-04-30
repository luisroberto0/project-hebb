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
| #31 | Outline + Introduction + Background + refs.bib inicial | ✅ done |
| #33 | Methods + Experiments | ✅ done (#32 do PLAN foi admin de README cleanup) |
| #34 | Discussion + Conclusion + decisão sobre apêndice Marco 1 | ✅ done |
| #35 | Slim revision + Abstract + Figures + LaTeX conversion + bib final | TODO |
| #36 | Peer review interno + revisão final | TODO |

**Decisão pós-#34 sobre apêndice Marco 1:** `appendix.md` criado mas **NÃO incluído no main paper draft**. Razão: Marco 1 explora continual learning sem replay; main paper foca sparsity em few-shot — narrativas distintas, conexão "ambos do mesmo projeto" é justificativa fraca pra apêndice em workshop paper de 6-8 páginas. Conteúdo permanece como supplementary material disponível mediante solicitação.

**Word count global pós-#34:** 4632 palavras (target 3500-4500). Slim de ~130 palavras necessário em #35; foco em background (857 → ~700) e methods (916 → ~750).

## Files

- `outline.md` — estrutura detalhada de cada seção (✅ atualizada com checkboxes)
- `intro.md` — Section 1: Introduction (✅ draft)
- `background.md` — Section 2: Background (✅ draft)
- `methods.md` — Section 3: Method (✅ draft)
- `experiments.md` — Section 4: Experiments (✅ draft)
- `discussion.md` — Section 5: Discussion (✅ draft, ~964 palavras)
- `conclusion.md` — Section 6: Conclusion (✅ draft, ~288 palavras)
- `appendix.md` — Apêndice Marco 1 (✅ draft ~500 palavras, **NÃO incluir no main paper** — supplementary material)
- `refs.bib` — Bibliography (BibTeX, ~14 entradas, expandir conforme texto evolui)
- `figs/` — Figuras (TODO #35, gerar do `experiment_01_oneshot/c3_protonet_sparse.py` outputs)

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
