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

### Próximo

Sweep formal 5 seeds (rodando). Se confirmar (timing ≥10 p.p. E rec ≥65% com IC95%), **Marco 2-C = Sucesso** — decisão de #80 (publicar 1º positivo / estender). Se a margem encolher com seeds, reavaliar.
