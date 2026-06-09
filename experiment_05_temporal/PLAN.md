# Experimento 05 — Raciocínio temporal: timing em SHD (Marco 2-C)

> **Status:** kickoff/scoping em sessão #71 (2026-06-09). Eixo B (benchmark neuromórfico real, SHD). Spec pendente de revisão do Luis antes do código (#72).
> **Limite hard:** ~10 sessões (#71-#80). Admin obrigatória no fim. Cancelável se evidência clara antes.

---

## Posicionamento na missão pós-LLM

CONTEXT.md §1 linha 14: **raciocínio temporal via timing de spikes (event-driven, esparso)** — a 4ª e **última** das capacidades pós-LLM, a única ainda não atacada. STRATEGY.md (#26) registrou-a como a mais "especulativa, niche, poucos benchmarks padrão". Mas é também o **domínio nativo da SNN**: ao contrário da eficiência em CPU (Marco 2-B, ❌) ou da transferência cross-domain (Marco 2-A, ❌), aqui o spiking tem **vantagem estrutural** — informação codificada no tempo é o que a dinâmica LIF/recorrente processa por construção.

Estado das 4 capacidades ao iniciar este marco: one-shot inédito ❌ (2-A), continual ❌ (Marco 1), eficiência radical ❌ (2-B), **raciocínio temporal = esta**. Os 3 marcos anteriores produziram achados negativos quando o mecanismo bio-inspirado foi isolado. Marco 2-C testa a hipótese onde a SNN tem a melhor chance estrutural de **genuinamente agregar** — possível primeiro marco positivo.

---

## Pergunta científica

> **"Uma SNN recorrente (LIF) classifica Spiking Heidelberg Digits (SHD) com acurácia substancialmente acima de um baseline cego ao timing (histograma de spikes por canal → MLP)? A diferença SNN − baseline-cego isola a contribuição do timing — e a SNN recorrente bate a feedforward, isolando que é a dinâmica temporal recorrente, não só o spiking?"**

### Por que o baseline cego é a referência certa

O baseline cego soma os spikes de cada canal ao longo do tempo (vetor 700-dim) e classifica com MLP. Preserva **qual** canal/frequência disparou (o espectro médio), **destrói quando** (o timing). Logo, `SNN − cego` mede a contribuição do **timing especificamente**, controlando pela informação espectral que ambos têm.

**Nuance honesta:** o espectro médio é bem discriminativo para fala — o baseline cego pode ser mais forte que o ingênuo (~50–60%?), apertando a margem. Isso torna o critério de ≥10 p.p. não-trivial e o experimento genuinamente informativo.

---

## Predição provisional (registrada ANTES do experimento)

Diferente dos 3 marcos anteriores, aqui há chance real de positivo (domínio nativo da SNN; SHD desenhado para exigir timing).

| Modelo | ACC SHD esperado (literatura/estimativa) |
|---|---|
| Baseline cego (histograma → MLP) | 45–60% (espectro médio é informativo; timing perdido) |
| SNN feedforward (LIF) | 48–66% (Cramer 2020) |
| SNN recorrente (LIF recurrent) | 70–83% (Cramer 2020, Zenke) |
| chance (20 classes) | 5% |

**Predição central:** SNN recorrente atinge ≥65% (provável); a **margem sobre o cego é a incógnita real** — pode ser ≥10 p.p. (timing agrega, Sucesso) ou menor (espectro médio já carrega muito). Predição: **provável Sucesso ou Mediano**; a recorrente deve bater a feedforward (isolando a dinâmica temporal). Se o timing NÃO agregar nem aqui (margem < 10 p.p.), seria o achado negativo mais forte do projeto — bio-inspiração não agrega nem no seu domínio nativo.

---

## Dataset: SHD (Spiking Heidelberg Digits)

Cramer, Stradmann, Schemmel & Zenke 2020 ("The Heidelberg spiking data sets..."). ~10.420 amostras de dígitos falados 0–9 em inglês e alemão (**20 classes**), convertidos em spikes por um modelo de cóclea artificial (**700 canais** de input × ~1 s). Splits train/test oficiais (test = falantes held-out). Formato HDF5; acesso via `tonic` (biblioteca padrão de datasets neuromórficos) ou download direto do Zenke lab. **#72 começa com download + dataloader + smoke (1 batch).**

---

## Setup experimental

| Modelo | Definição |
|---|---|
| **Baseline cego ao timing** | Σ spikes por canal sobre o tempo (700-dim) → MLP (256 hidden). Sem ordem temporal. Referência. |
| **SNN feedforward** | LIF 2 camadas (700→256→20), rate/temporal input, surrogate-gradient (BPTT). |
| **SNN recorrente** | LIF recorrente (recorrência na camada hidden), surrogate-gradient. O modelo que explora o timing. |

Stack: PyTorch + snntorch (+ `tonic` para SHD). Treino na 4070. **Risco de VRAM:** SNN recorrente em sequências de ~1 s (centenas de timesteps) é VRAM-hungry via BPTT — mitigar com truncated BPTT, binning temporal (ex: 100 bins) e batch menor.

---

## Métrica primária

- **ACC test split SHD**, 5 seeds, IC95% bootstrap (consistente com marcos anteriores).
- **Métrica-chave: `ACC(SNN recorrente) − ACC(baseline cego)`** = contribuição do timing.
- Secundária: `ACC(recorrente) − ACC(feedforward)` = contribuição da dinâmica recorrente; curva de acc vs nº de bins temporais (granularidade do timing).

---

## Critério literal de fechamento

Test split, 5 seeds, IC95% bootstrap. Casos exaustivos:

| Resultado | Decisão |
|---|---|
| SNN recorrente − cego **≥ 10 p.p.** **E** SNN recorrente **≥ 65%** | **Sucesso** → raciocínio temporal AGREGA (primeiro marco positivo) |
| exatamente um dos dois (margem ≥10pp mas SNN <65%; ou SNN ≥65% mas margem <10pp) | **Mediano** → admin #80 decide (timing agrega mas modelo fraco / modelo bom mas timing não é o diferencial) |
| margem < 10 p.p. **E** SNN < 65% | **Falha** → timing não agrega nem no domínio nativo; Marco 2-C encerrado |

Critério não-negociável. Régua sujeita a refino numérico após lit review (#71b/#72, Cramer 2020 reporta números concretos), mas a estrutura (margem sobre cego + piso de acc) é fixa.

---

## Plano de sessões #71-#80 (orientativo, sem cadência fixa)

| # | Tipo | Output |
|---|---|---|
| **71 (esta)** | Admin + scoping | Design aprovado + `PLAN.md` + STRATEGY/CONTEXT |
| 72 | Lit review + code | `PAPERS.md` (Cramer 2020, Maass 1997, Zenke SuperSpike, Neftci surrogate); download SHD + dataloader + smoke |
| 73 | Code | Baseline cego (histograma → MLP) + harness de avaliação |
| 74 | Code | SNN feedforward + binning temporal; primeira acc |
| 75-76 | Code | SNN recorrente (truncated BPTT, mitigação VRAM); acc + margem sobre cego |
| 77-78 | Analysis | Multi-seed 5 seeds + IC95%; curva acc vs nº bins; critério literal? |
| 79 | Writing | Draft de caracterização (positivo se Sucesso, negativo defensável se Falha) |
| **80** | **Admin obrigatória** | Critério? Decide: publicar (1º positivo) / Marco 2-C.2 / encerrar projeto (4 capacidades exploradas) |

---

## Restrições

- **Não modifica** os marcos anteriores (paper_c3, paper_marco2a, experiment_01-04 congelados).
- **Surrogate-gradient é backprop** — reconhecido: este marco testa se a *dinâmica temporal* (recorrência LIF) explora o timing, não treino-sem-backprop. A comparação recorrente-vs-feedforward isola a dinâmica.
- **Honestidade do baseline:** o cego deve ser forte (MLP bem treinado sobre o histograma), não um espantalho — senão a margem é inflada.
- **Sem cadência fixa.** Cancelável se #75-#76 já mostrarem evidência inequívoca.

---

## Riscos e mitigação

| Risco | Mitigação |
|---|---|
| SHD indisponível / formato | `tonic` (padrão); fallback download HDF5 direto Zenke lab. Validar nº amostras/classes em #72. |
| VRAM estoura (BPTT recorrente, seq longa) | Binning temporal (100 bins), truncated BPTT, batch menor, `empty_cache`. |
| Baseline cego fraco infla a margem | Treinar o MLP cego com o mesmo esforço da SNN; reportar o cego com rigor. |
| Achado "esperado por construção" (SHD é temporal) | A nuance espectro-médio aperta a margem; reportar a magnitude exata + recorrente-vs-feedforward isola o mecanismo. |

---

## Referências (lit review #72)

- **Cramer, Stradmann, Schemmel & Zenke 2020** — SHD/SSC datasets + baselines (recorrente vs feedforward)
- **Maass 1997** — Networks of Spiking Neurons (computação temporal com spikes)
- **Zenke & Ganguli 2018** — SuperSpike (treino temporal de SNN)
- **Neftci, Mostafa & Zenke 2019** — surrogate gradient (método de treino)
