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

### #73 — extensão: sweep de resolução temporal (a vantagem é mesmo o timing?)

Teste mecanístico da ressalva "SHD é temporal por construção". SNN recorrente, 3 seeds, 8 epochs, variando o nº de bins. Baseline cego (invariante a bins, soma total): 49.41%.

| bins | SNN-rec acc | margem sobre cego |
|---|---|---|
| 1 | 58.92% ±0.70 | +9.51 |
| 4 | 63.06% | +13.65 |
| 8 | 63.49% | +14.08 |
| 16 | 67.29% | +17.87 |
| 32 | 68.45% | +19.04 |
| 64 | 68.54% | +19.13 |
| 100 | 69.10% ±0.81 | +19.68 |

**A acurácia cresce monotonicamente com a resolução temporal** (58.92 → 69.10), saturando ~32–100 bins. Mais resolução temporal ⇒ mais acurácia: o timing carrega informação.

**Nuance honesta (predição parcialmente refutada — e isso é valioso):** a predição era bins=1 ≈ cego (margem ~0). Em vez disso, **bins=1 mantém +9.51 p.p.** sobre o cego. Causa: em bins=1 a SNN ainda usa LIF + BatchNorm (vs ReLU do cego) — esse +9.5 p.p. é **arquitetural, não timing**.

**Decomposição limpa (a extensão valeu por isto):**
- **Timing puro** (mesma SNN recorrente, bins 1→100, só muda a resolução): **+10.18 p.p.** — controla a arquitetura, isola o timing genuíno.
- Confound arquitetural (SNN-1step vs MLP, bins=1 vs cego): +9.51 p.p.

Ou seja, dos ~+19.7 p.p. da vantagem total da SNN recorrente sobre o cego (#72), **~+10 p.p. são timing genuíno e ~+9.5 p.p. são arquitetura** — a margem "timing" do #72 estava inflada pelo confound arquitetural. O critério literal (margem total ≥10 p.p., rec ≥65%) **permanece atingido**, mas a atribuição fica honesta: o timing genuinamente agrega ~+10 p.p., quantificado e controlado. Achado mais forte e mais defensável que o #72 isolado.

## #74 — extensão: latency coding (time-to-first-spike) — a SNN explora timing fino?

rate = contagem por bin; latency = 1 spike por canal, no bin do 1º spike (onset). 3 seeds, 8 epochs.

| coding | cego | SNN-rec | margem |
|---|---|---|---|
| rate | 49.41% | 69.10% | +19.68 p.p. |
| latency | 23.51% | 50.68% | +27.16 p.p. |

1. **A SNN explora o timing do onset:** com APENAS 1 spike por canal (a latência pura), a SNN-rec atinge **50.68%** — muito acima de chance (5%) e do cego-latency (23.51%, mesmos canais sem timing). Prova que a SNN extrai informação da **latência do 1º spike**, não só de quais canais ativam.
2. **Mas a contagem completa (rate) agrega mais:** SNN-rate 69.10% > SNN-latency 50.68% por **+18.42 p.p.** A dinâmica de spikes completa carrega informação além do onset.

**Nuance honesta:** o cego-latency (23.51%) é baseline fraco — sob latency o `bag_of_spikes` vira presença binária quase-saturada (546/700 canais), pouco discriminativa; a margem +27 está inflada. Número honesto = absoluto: **50.68% de padrões de latência pura**, bem acima de chance.

## #75 — extensão: k-WTA temporal — a esparsificação preserva o timing?

k-WTA por timestep na hidden da SNN recorrente (≤k spikes ativos/timestep, de 256). Fecha a narrativa de k-WTA do projeto. 3 seeds, 8 epochs. Baseline cego 49.41%.

| k (de 256) | sparsity | SNN-rec acc | vs denso |
|---|---|---|---|
| denso | 0% | 69.10% | — |
| 128 | 50% | 69.60% | +0.50 |
| 64 | 75% | 67.59% | **−1.50** |
| 32 | 87.5% | 62.71% | −6.39 |
| 16 | 93.8% | 55.54% | −13.56 |
| 8 | 96.9% | 48.98% | −20.11 (≈ cego) |
| 4 | 98.4% | 42.64% | −26.46 |

**Achado — paralelo quase exato com o C3 in-domain:** o k-WTA temporal é tolerante até **75% de sparsity** (k=64: −1.50 p.p.) — praticamente idêntico ao k-WTA espacial in-domain do C3 (75% = −1.45 p.p.). A esparsidade é tolerada in-domain em **ambos os eixos** (espaço e tempo) com o mesmo custo. Degradação acelera além de 87.5% (como no C3). **Colapso em esparsidade extrema:** k≤8 (>96%) → SNN converge para o baseline cego (48.98% ≈ 49.41%); o timing exige uma capacidade mínima de spikes/timestep.

**Narrativa de k-WTA fechada (3 papers):** tolerante in-domain espacial (C3, −1.45 p.p. @75%) ≈ tolerante in-domain temporal (2-C, −1.50 p.p. @75%); **colapsa** cross-domain (Marco 2-A, effect collapse) e em esparsidade temporal extrema (2-C, k≤8). A esparsidade k-WTA é compatível in-domain em qualquer eixo, mas frágil sob shift de domínio ou compressão extrema.

## #76 — extensão: generalização para SSC (Spiking Speech Commands)

SSC: dataset irmão (Cramer/Zenke), 35 classes (~75k), mais difícil. Subset balanceado (75k estouraria RAM). chance 2.86%.

| run | cego | SNN-ff | SNN-rec | timing (rec−cego) |
|---|---|---|---|---|
| 8 ep, 400/classe | 19.32% | 6.83% | 14.05% | −5.27 p.p. |
| 25 ep, 500/classe | 21.72% | 9.82% | 22.42% | +0.70 p.p. |

**Resultado: INCONCLUSIVO / generalização frágil no regime testado.** Com mais treino a SNN-rec subiu (14% → 22.4%) e ultrapassou o cego, e a margem cresceu (−5.27 → +0.70 p.p.) — mas fica **muito abaixo** dos +19.7 p.p. do SHD.

**Caveat de subtreino (decisivo):** as SNNs estão muito abaixo da literatura SSC (recorrente ~50–70% com dataset completo); ff em ~10% confirma subtreino. A margem **cresce com epochs** (−5.27 → +0.70), logo o regime testado (subset 17.5k de 75k, epochs limitados) é **insuficiente** para concluir. Resposta definitiva exigiria SSC completo até convergência — caro (BPTT recorrente, 35 classes), fora do escopo desta passada.

**Honestidade:** o Sucesso do Marco 2-C está estabelecido no **SHD** (#72, robusto, 5 seeds); a generalização para SSC é **inconclusiva** — a vantagem de timing não se reproduz com subset/epochs limitados, embora a tendência (margem crescente) seja ambígua. Isto **endurece a ressalva** sem invalidar o achado: o positivo é demonstrado em 1 benchmark (SHD); robustez a um benchmark temporal mais difícil fica em aberto.

## Síntese — caracterização do timing (Marco 2-C, #72–#76)

| Frente | Achado |
|---|---|
| #72 Sucesso (formal, 5 seeds, SHD) | timing total **+19.7 p.p.**, SNN-rec **71.27%** |
| #73 sweep de bins | timing genuíno **+10.18 p.p.** (controlando arquitetura) |
| #74 latency coding | SNN extrai **50.68%** só do onset timing |
| #75 k-WTA temporal | esparsidade tolerante até 75% (**−1.50 p.p.**, paralelo ao C3); colapsa em >96% |
| #76 generalização SSC | **inconclusiva** — margem +0.70 p.p. (vs +19.7 SHD), mas SNNs subtreinadas/subset |

**Veredicto honesto:** Marco 2-C = **Sucesso no SHD, bem-caracterizado** (timing genuíno ~+10 p.p. controlado, robusto à esparsificação, paralelo ao C3). É o 1º achado positivo do projeto. A generalização para um benchmark temporal mais difícil (SSC) **não foi estabelecida** nesta passada — o achado é forte mas não confirmado como universal. Honestidade metodológica: reporto o limite, não inflo o positivo.

## Decisão pendente (Luis, admin)

Marco 2-C = **Sucesso bem-caracterizado**. Caminhos (decisão de rumo do Luis): mais extensões (k-WTA temporal, SSC) / publicar (1º positivo + 3 negativos = narrativa completa) / fechar e consolidar o projeto (4 capacidades: 3 ❌ + 1 ✅).
