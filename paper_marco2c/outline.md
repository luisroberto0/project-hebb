# Paper Marco 2-C — Outline

> Status: criado no kickoff. Word-count target ~3000-4000 (paralelo aos papers C3/2-A).

---

## Estrutura

| Seção | Word target | Status |
|---|---|---|
| Abstract | 150-180 | ✅ kickoff |
| 1. Introduction | 500-700 | ✅ kickoff |
| 2. Background | 500-700 | ⏳ |
| 3. Method | 500-700 | ⏳ |
| 4. Experiments | 900-1100 | ⏳ |
| 5. Discussion | 600-800 | ⏳ |
| 6. Conclusion | 150-200 | ⏳ |

---

## 1. Introduction
- SNNs recorrentes "exploram timing" — mas a vantagem sobre baseline não-temporal conflaciona *timing genuíno* com *diferença arquitetural*.
- Gap: poucos trabalhos *controlam* esse confound.
- O que fazemos: caracterizar SNN recorrente em SHD, separar timing de arquitetura, medir o limite de generalização.
- Contribuições (4): decomposição timing/arq; múltiplas evidências do timing; k-WTA temporal tolerante (paralelo C3); limite de generalização.
- Honest scoping: sem superioridade vs convencional, sem método novo, usa backprop.

## 2. Background
- SNNs, surrogate gradient (Neftci 2019), LIF recorrente.
- SHD/SSC (Cramer/Zenke 2020) — benchmarks neuromórficos temporais.
- k-WTA e sparse coding (Maass 2000) + referência ao paper C3 (k-WTA espacial in-domain).
- Por que "timing-blind baseline" (histograma por canal) é o controle certo.

## 3. Method
- Arquitetura: BlindMLP (cego), SNN feedforward, SNN recorrente (LIF, BN sem running-stats, readout integrador).
- Detalhe crítico de treino: BatchNorm com `track_running_stats=False` (BN running-stats misturava os 100 timesteps; destravou 13%→71%). Reportar como lição reproduzível.
- Encodings: rate (contagem por bin) e latency (time-to-first-spike).
- k-WTA temporal (≤k spikes/timestep).
- Avaliação: SHD test, 5 seeds, IC95% bootstrap.

## 4. Experiments
- **4.1** Resultado principal (SHD, 5 seeds): cego 51.56 / ff 61.02 / rec 71.27; timing +19.71, recorrência +10.26.
- **4.2** Decomposição (sweep de bins): timing genuíno +10.18 (mesma SNN, bins 1→100); arquitetura +9.51 (bins=1 vs cego). Curva acc vs resolução temporal.
- **4.3** Latency coding: rec extrai 50.68% só do onset; rate (69.10) > latency (50.68) por +18 p.p.
- **4.4** k-WTA temporal: tolerante até 75% (−1.50 p.p.); paralelo C3 (−1.45 p.p.); colapsa em >96%. Curva acc vs k.
- **4.5** Generalização SSC: fraca, positiva, estável (+4–5 p.p.); não subtreino (platôou); magnitude dataset-específica.
- Tabelas + figuras (curva de bins, curva de k-WTA, barras das condições).

## 5. Discussion
- Por que separar timing de arquitetura importa (a vantagem bruta superestima o timing).
- O paralelo k-WTA espacial↔temporal (a esparsidade é compatível in-domain em qualquer eixo).
- Por que generaliza fraco no SSC (mais classes, baseline cego relativamente mais forte, limite de preprocessing).
- Limitações: 1 par de datasets, surrogate-gradient (backprop), sem comparação com transformers/RNN convencionais, CPU/GPU (não neuromórfico).
- O que NÃO concluímos.

## 6. Conclusion
- O timing carrega informação recuperável, quantificada com controles (+10.18 p.p. genuíno); modesto e dataset-específico. Contribuição: a decomposição honesta + o paralelo k-WTA.

---

## Figuras (a gerar)
- Fig 1: barras das condições SHD (cego/ff/rec) + chance.
- Fig 2: curva acc vs resolução temporal (bins) — mostra timing genuíno.
- Fig 3: curva acc vs k (k-WTA temporal) — mostra tolerância 75% + colapso.
