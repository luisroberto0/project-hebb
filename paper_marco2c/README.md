# Paper Marco 2-C — Disentangling timing from architecture in recurrent SNNs (SHD)

## Overview

Paper sobre o Marco 2-C (`experiment_05_temporal/`, sessões #71–#77b): caracterização honesta de **quanto** uma SNN recorrente explora o **timing** dos spikes em Spiking Heidelberg Digits (SHD), com a contribuição do timing **separada** do confound arquitetural, e o limite de generalização medido.

Paper de caracterização rigorosa — o **primeiro achado positivo** do Project Hebb, deliberadamente *não* inflado.

## Title (tentative)

**"Disentangling Timing from Architecture in Recurrent Spiking Networks: A Characterization on Spiking Heidelberg Digits"**

Alternativas:
- "How Much Does Spike Timing Help? A Controlled Characterization on SHD"
- "Recoverable Timing Information in Recurrent SNNs, and Its Limits"

Decisão final pós-draft (Luis decide com material concreto).

## Contribuição central (honesta)

1. **Decomposição timing vs. arquitetura:** a vantagem bruta de uma SNN recorrente sobre um baseline cego ao timing (+19.7 p.p. em SHD) é decomposta, via sweep de resolução temporal, em **timing genuíno (+10.18 p.p.)** e **confound arquitetural (LIF+BN vs MLP, +9.5 p.p.)**. A maioria dos trabalhos reporta a vantagem bruta; nós a controlamos.
2. **Múltiplas evidências do timing:** latency coding (a SNN extrai 50.68% só do onset); a acurácia cresce monotonicamente com a resolução temporal.
3. **k-WTA temporal tolerante a 75% (−1.50 p.p.)** — paralelo quase exato ao k-WTA *espacial* in-domain do paper C3 (−1.45 p.p.). A esparsidade é compatível in-domain em ambos os eixos.
4. **Limite de generalização medido:** o achado é forte no SHD mas **fraco e estável** em SSC (+4–5 p.p., dataset-específico, não subtreino).

## Honest scoping

**NÃO claim:** método novo, superioridade sobre redes convencionais (que fazem >90% em SHD/SSC), caminho pós-LLM, ou treino sem backprop (usa surrogate-gradient = BPTT).

**Claim:** caracterização rigorosa e controlada de quanto o timing agrega numa SNN recorrente, com o confound arquitetural isolado e o limite de generalização medido — incluindo o paralelo transversal com a tolerância à esparsidade espacial do C3.

## Target venue (a definir)

Decisão de Luis pós-draft. Candidatos: NeurIPS/ICLR workshop bio-plausible/neuromorphic, ou LinkedIn em PT (consistente com paper C3). Workshop-scope.

## Fontes (tudo reproduzível)

- `experiment_05_temporal/WEEKLY-1.md` — 6 frentes caracterizadas (#72–#77b)
- `experiment_05_temporal/PAPERS.md` — lit review (Cramer/Zenke 2020, Maass 1997, Neftci 2019, Xiao 2017)
- Scripts: `temporal_bench.py`, `sweep_temporal.py`, `sweep_bins.py`, `sweep_latency.py`, `sweep_kwta.py`, `sweep_ssc_full.py`
- `SYNTHESIS.md` (raiz) — contexto da jornada inteira

## Status

| Sessão | Conteúdo | Status |
|---|---|---|
| kickoff | README + outline + abstract + intro | ✅ |
| kickoff | Background + Method | ✅ |
| kickoff | Experiments (5 subseções + 4 tabelas) | ✅ |
| kickoff | Discussion + Conclusion | ✅ |
| kickoff | refs.bib (5 entradas) | ✅ |
| kickoff | Figuras (barras SHD, curva bins, curva k-WTA — 300 DPI) | ✅ |
| kickoff | LaTeX consolidação (`main.tex`) + validação estrutural | ✅ |
| próxima | Peer review interno + decisão de venue (Luis) | pendente |

**Draft COMPLETO** (6 seções + abstract + 3 figuras + `main.tex`, ~3300 palavras). Validação estrutural OK (5 cites no .bib, refs/labels resolvem, 14/14 ambientes, 3 figuras). Compila via Overleaf (upload `main.tex` + `refs.bib` + `figs/`) ou `pdflatex+bibtex` local. Falta: peer review + decisão de venue.

## Tom

Honesto, direto, sem hype. Inglês técnico, consistente com `paper_c3/main.tex` e `paper_marco2a/main.tex`. Achado positivo modesto e bem-caracterizado vale mais que claim inflado.
