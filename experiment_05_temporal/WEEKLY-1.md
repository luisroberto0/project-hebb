# WEEKLY-1 — Marco 2-C (raciocínio temporal: SHD)

> Sessões #71-#72. **Smoke POSITIVO** — primeiro sinal positivo do projeto. Confirmação formal (5 seeds) em curso.

---

## #71 (2026-06-09) — kickoff/scoping

Marco 2-C reaberto (Luis). Eixo B (SHD). Spec em `PLAN.md`. Critério: SNN recorrente − cego ≥ 10 p.p. **E** SNN recorrente ≥ 65% (5 seeds, IC95%). Predição: provável Sucesso/Mediano (domínio nativo da SNN).

## #72 (2026-06-09) — dados + harness + debugging + smoke

**Setup de dados.** `tonic` não builda (expelliarmus/C++ falha no Py3.13/Windows) → `h5py` 3.16 instalado, download direto dos HDF5 (Zenke lab): `shd_test.h5` (79 MB, 2264 amostras), `shd_train.h5` (268 MB, 8156 amostras), 20 classes balanceadas. `shd_data.py`: binning temporal (100 bins × 700 canais), `bag_of_spikes` (histograma por canal, o baseline cego). Validado: baseline cego tem 546/700 canais ativos — confirma a nuance (espectro médio é rico).

**Harness (`temporal_bench.py`):** BlindMLP (cego), SNN_FF (LIF feedforward), SNN_Rec (LIF recorrente all-to-all), readout integrador.

### Debugging metódico (CLAUDE.md: diagnosticar antes de mudar)

Smoke inicial: cego 44%, mas **SNN_FF 13% / SNN_Rec 7% (~chance)**. As SNNs ficavam *abaixo* do cego (que deriva do mesmo histograma) → mal treinadas, não achado científico.

| Hipótese | Teste | Resultado |
|---|---|---|
| H1: saturação da entrada (corrente >> threshold) | diag: taxa de disparo da hidden | **Refutada** — corrente 0.042 << threshold 1.0; hidden só 3.7% (sub-ativada, não saturada) |
| H2: corrente baixa + readout fraco | + BatchNorm na fc1 + readout integrador | **Parcial** — ainda ~13% test |
| H3: gap train/test (loss treina mas test trava) | diag instrumentado | **Confirmada raiz** — loss 12.7→0.92, hidden 0.32, grad flui (ratio 1.14), logits std 18: **treino funciona, test em 13%** |
| H3 raiz: BatchNorm running-stats ruins por timestep | `track_running_stats=False` (batch stats no eval) | **RESOLVIDO** |

A BN acumulava running mean/var misturando os 100 timesteps; no eval aplicava stats ruins. `track_running_stats=False` (usa batch stats também no eval) é o fix padrão de BN em SNN sem BNTT completo.

### Smoke (1 seed, 8 epochs) — POSITIVO

| Modelo | acc |
|---|---|
| BlindMLP (cego) | 48.67% |
| SNN feedforward | 62.46% |
| **SNN recorrente** | **69.88%** |

- **timing (rec − cego): +21.20 p.p.** (critério ≥10) ✓✓
- **recorrência (rec − ff): +7.42 p.p.** (a dinâmica recorrente agrega)
- **SNN recorrente: 69.88%** (critério ≥65) ✓ — consistente com a literatura (Cramer ~71–83%)

→ **Ambas as condições do critério satisfeitas no smoke → SUCESSO preliminar.** Primeiro sinal **positivo** do projeto: o timing carrega informação real (o áudio falado é discriminado pela ordem dos eventos, não só pelo espectro médio). O baseline cego forte (48.67%) mostra que muito está no espectro, mas o timing ainda agrega +21 p.p.

### Ressalvas honestas

- **Smoke (1 seed, 8 epochs).** Confirmação formal (5 seeds, IC95%, 15 epochs) **em curso** (`sweep_temporal.py`).
- **Esperado-por-construção parcial:** SHD foi desenhado para exigir timing, então um positivo aqui é menos surpreendente que um negativo seria. O valor é a *magnitude* quantitativa (+21 p.p. timing, +7 p.p. recorrência) e o fato de ser o 1º marco onde o mecanismo neuro-inspirado genuinamente agrega.
- surrogate-gradient é backprop — mede se a *dinâmica temporal* (recorrência LIF) explora o timing, não treino-sem-backprop.

## #72b — sweep formal (5 seeds, IC95% bootstrap, SHD completo, 15 epochs) CONFIRMA: SUCESSO

| Modelo | acc (5 seeds) | CI95% | seeds |
|---|---|---|---|
| BlindMLP (cego) | 51.56% ±0.77 | [50.78, 52.09] | 51.9/51.8/50.1/52.3/51.7 |
| SNN feedforward | 61.02% ±0.85 | [60.27, 61.77] | 62.1/60.1/61.1/61.8/60.0 |
| **SNN recorrente** | **71.27% ±0.45** | **[70.91, 71.70]** | 71.2/71.3/72.0/70.6/71.2 |

- **timing (rec − cego): +19.71 p.p.** (critério ≥10) ✓✓ — os ICs das três condições **não se sobrepõem** (cego [50.78,52.09] / ff [60.27,61.77] / rec [70.91,71.70]); diferenças estatisticamente robustas.
- **recorrência (rec − ff): +10.26 p.p.** — a dinâmica recorrente agrega claramente, além do spiking feedforward.
- **SNN recorrente: 71.27%** (critério ≥65) ✓ — dentro da faixa da literatura (Cramer ~71–83%).

→ **VEREDICTO: SUCESSO.** Critério literal atingido com 5 seeds e ICs apertados. **Primeiro marco POSITIVO do Project Hebb** após 3 achados negativos (continual #30, cross-domain #66, eficiência #70).

### Interpretação honesta (o que o Sucesso significa e o que não significa)

- **Significa:** o timing dos spikes carrega +19.7 p.p. de informação sobre o histograma (espectro médio); a recorrência LIF agrega +10.3 p.p. sobre feedforward. O mecanismo neuro-inspirado (dinâmica temporal) genuinamente agrega — quantificado e robusto.
- **NÃO significa "raciocínio" abstrato:** o critério mede *classificação que explora timing*, não inferência temporal abstrata. Chamar de "raciocínio temporal" é a moldura do CONTEXT §1; o achado concreto é "classificação SNN explora estrutura temporal do áudio".
- **Esperado-por-construção parcial:** SHD foi desenhado para exigir timing. O valor é a *magnitude* (+19.7 / +10.3 p.p.) e o contraste com os 3 marcos negativos — não que "a SNN venceu" (esperado num benchmark temporal).
- surrogate-gradient é backprop: mede a *dinâmica temporal* (recorrência), não treino-sem-backprop.

### Decisão pendente (Luis, admin)

Critério atingido (Sucesso). Caminhos (decisão de rumo do Luis): publicar o 1º positivo (paper/post) / estender (mais bins, latency coding, SSC) / fechar o marco como Sucesso e consolidar o projeto (4 capacidades exploradas: 3 ❌ + 1 ✅). Confirmação formal completa; o que falta é a moldura do achado.
