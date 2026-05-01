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

**Luis Roberto Pinho da Silva Junior** — Independent research
[LinkedIn: luisroberto0](https://www.linkedin.com/in/luisroberto0/)

## Status

| Sessão | Conteúdo | Status |
|---|---|---|
| #31 | Outline + Introduction + Background + refs.bib inicial | ✅ done |
| #33 | Methods + Experiments | ✅ done |
| #34 | Discussion + Conclusion + decisão sobre apêndice Marco 1 | ✅ done |
| #35 | Slim revision + Abstract + Figures + LaTeX conversion | ✅ done |
| #36 | LinkedIn post draft + LaTeX compilation status (decisão: NÃO submeter NeurIPS) | ✅ done |
| #36b | LaTeX fixes pós-revisão Luis: cite Adam, Figure 2, header polish | ✅ done |

## Pós-#36b — Correções LaTeX aplicadas

Luis revisou o PDF compilado da sessão #36 (via Overleaf) e identificou 3 issues. Todos corrigidos nesta sessão:

1. **`refs.bib`** — adicionada entrada `kingma2015adam` (citação de Adam estava renderizando como `(?)` em Section 3.3 por entrada faltante).
2. **`main.tex`** — Figure 2 (`figs/fig2_validation.pdf`, gerada na #35) inserida em Section 4.4 logo após o parágrafo do gap +55.50 p.p.; estava órfã no diretório.
3. **`main.tex`** — header `\affil[1]` trocou `\texttt` por `\url` (resolve vírgula órfã na renderização, semanticamente correto pra URLs).

**Conteúdo do paper intocado** (apenas fixes técnicos de LaTeX). PDF requer recompilação manual via Overleaf — passos detalhados em `latex_status.md`.

## Final state — pós-#36

**Decisão estratégica final:** **NÃO submeter pra NeurIPS Bio-Plausible Workshop.** Em vez disso, postar no LinkedIn em PT como anúncio + repo + PDF anexado. Razão registrada em `STRATEGY.md` "Decisão pós-#35: LinkedIn em vez de NeurIPS" — tempo limitado de side project não absorve rebuttals/revisões/registration; LinkedIn alcança parte do que peer-review faria sem overhead institucional.

**Paper draft preservado integralmente.** Conteúdo permanece publicável depois se Luis decidir submeter pra workshop futuro.

**Status dos componentes:**

| Componente | Status |
|---|---|
| 6 seções drafted + slim (3789 palavras dentro do alvo) | ✅ |
| Abstract (158 palavras) | ✅ |
| Figuras (2x PNG+PDF, 300 DPI) | ✅ |
| Bibliography (`refs.bib`, 14 entradas) | ✅ |
| `main.tex` consolidado (~530 linhas, compilável) | ✅ |
| Apêndice Marco 1 (supplementary, não no main) | ✅ |
| **PDF compilado** | ⏳ pendente — Luis roda Overleaf (10 min, ver `latex_status.md`) |
| LinkedIn post (longo + curto) | ✅ drafts em `linkedin_post.md` e `linkedin_post_short.md` |
| Plano de publicação | ✅ documentado |

**Próximo passo (Luis, sem cadência fixa):**
1. Compilar `main.tex` via Overleaf (~10 min, instruções em `latex_status.md`)
2. Renomear PDF pra `Project_Hebb_C3_DeepDive.pdf`, adicionar ao repo
3. Revisar `linkedin_post.md` (longo) ou `linkedin_post_short.md` (curto), ajustar tom pra rede dele
4. Postar no LinkedIn anexando `figs/fig1_sparsity_curve.png` + link pro repo

**Decisão pós-#34 sobre apêndice Marco 1:** `appendix.md` criado mas **NÃO incluído no main paper draft**. Razão: Marco 1 explora continual learning sem replay; main paper foca sparsity em few-shot — narrativas distintas. Conteúdo permanece como supplementary material disponível mediante solicitação.

**Word count global pós-#35:** 3789 palavras (target 3500-4500) ✅ DENTRO DO ALVO. Todas seções individuais também no range. Slim de #35 mais agressivo que planejado em background (−331) e methods (−413), mas resultado coerente.

**LaTeX status pós-#35:** `main.tex` escrito (compilável), 2 figuras geradas (`figs/fig1_sparsity_curve.{png,pdf}`, `figs/fig2_validation.{png,pdf}`), `refs.bib` com 14 entradas. **Não compilado localmente** (pdflatex não instalado). Ver `latex_status.md` pra opções de compilação.

## Files

- `outline.md` — estrutura detalhada com checkboxes ✅
- `linkedin_post.md` — Post LinkedIn versão longa PT-BR (✅ draft, ~1900 chars)
- `linkedin_post_short.md` — Post LinkedIn versão curta PT-BR (✅ draft, ~750 chars, fallback)
- `abstract.md` — Abstract (✅ draft, 158 palavras)
- `intro.md` — Section 1: Introduction (✅ draft, 647 palavras)
- `background.md` — Section 2: Background (✅ slim, 526 palavras)
- `methods.md` — Section 3: Method (✅ slim, 503 palavras)
- `experiments.md` — Section 4: Experiments (✅ draft, 969 palavras)
- `discussion.md` — Section 5: Discussion (✅ draft, 964 palavras)
- `conclusion.md` — Section 6: Conclusion (✅ slim, 180 palavras)
- `appendix.md` — Apêndice Marco 1 (✅ draft 500 palavras, **NÃO incluir no main paper** — supplementary)
- `refs.bib` — Bibliography (BibTeX, 14 entradas)
- `main.tex` — LaTeX consolidado (✅ draft, ~6-8 páginas estimadas, **não compilado localmente**)
- `latex_status.md` — Status de compilação + opções (TeX Live, Overleaf, Docker)
- `generate_figures.py` — Script reusable pra gerar figuras
- `figs/fig1_sparsity_curve.{png,pdf}` — Figura 1 (✅ gerada, 300 DPI)
- `figs/fig2_validation.{png,pdf}` — Figura 2 (✅ gerada, 300 DPI)
- `figs/README.md` — Descrição das figuras + como regenerar

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
