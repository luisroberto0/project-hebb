# WEEKLY-1 — Marco 2-B (eficiência radical: inferência event-driven)

> Sessões #67-#69. Smoke + primeira caracterização. **Veredicto preliminar: FALHA do critério literal** (conforme predição #67). Confirmação formal (treino completo, 5 seeds) pendente, mas o padrão é estrutural.

---

## #67 (2026-06-08) — kickoff/scoping

Marco 2-B reaberto (Luis via /goal "Eficiência radical"). Eixo A (inferência event-driven). Spec em `PLAN.md`: SNN-LIF + k-WTA temporal vs MLP denso em Fashion-MNIST, métrica dupla SynOps + latência CPU. Critério: acc −2 p.p. **E** SynOps ≥5× menores **E** latência CPU ≤ denso. Predição registrada: Sucesso falha, Mediano provável.

## #69 (2026-06-08) — harness + smoke

`efficiency_bench.py` implementado: MLP denso, SNN-LIF (rate/Poisson, surrogate-gradient), k-WTA temporal na **hidden** (k) e na **entrada** (k_in), contador de SynOps (Σ spikes pré × fan-out, sobre T), latência CPU single-thread (`torch.set_num_threads(1)`, warmup). **Smoke:** Fashion-MNIST quick (6k/2k), 2-3 epochs. NÃO são números finais — são tendência.

### Resultados (3 regimes)

**T=25, k=32 (default):**

| Modelo | acc | custo/amostra | latência CPU |
|---|---|---|---|
| DenseMLP | 80.5% | 203,264 MACs | 0.022 ms (1×) |
| SNN-LIF vanilla | 39.8% | 1,452,652 SynOps | 6.61 ms (**297× pior**) |
| SNN+kWTA hidden (k=32) | 49.7% | 1,448,407 SynOps | 7.08 ms (319× pior) |

**T=10, k=32, k_in=64:**

| Modelo | acc | SynOps | latência CPU |
|---|---|---|---|
| DenseMLP | 80.5% | 203,264 MACs | 1× |
| SNN-LIF vanilla | 77.0% | 580,976 (2.9× pior) | 148× pior |
| SNN+kWTA hidden (k=32) | 60.4% | 579,479 (não move) | 158× pior |
| SNN+kWTA **entrada** (ki=64) | 72.4% | 165,831 (**1.23× menos**) | 175× pior |

**T=5, k=32, k_in=32 (extremo esparso):**

| Modelo | acc | SynOps | latência CPU |
|---|---|---|---|
| DenseMLP | 81.5% | 203,264 MACs | 1× |
| SNN-LIF vanilla | 76.8% | 291,814 (1.4× pior) | 67× pior |
| SNN+kWTA hidden (k=32) | 59.9% | 289,584 | 74× pior |
| SNN+kWTA **entrada** (ki=32) | 63.8% | **42,239 (4.81× menos)** | 83× pior |

### Achados centrais (já decisivos no smoke)

1. **k-WTA na hidden NÃO move o SynOps** (1.452M→1.448M; 581k→579k; 292k→290k). O custo é dominado pela **primeira camada** (input-spikes × 256); k-WTA na hidden só afeta a fc2 (pequena). **Corolário de design:** a variável "k-WTA temporal" só ataca a eficiência se aplicada à **entrada** (gargalo fc1), não à hidden. Isso é específico de uma rede rasa — numa rede profunda haveria mais camadas internas pra esparsificar.

2. **Trade-off fundamental SynOps↔acurácia.** Para chegar perto dos 5× de SynOps, é preciso esparsidade extrema (T=5, k_in=32 = 96% dos pixels-spike descartados), que custa **−17.8 p.p. de acurácia** (63.8% vs 81.5%). Não há regime testado com 5× SynOps **e** acc dentro de −2 p.p. simultaneamente. A curva de Pareto acc↔SynOps é íngreme.

3. **A latência CPU é o veredicto estrutural.** Em TODOS os regimes (T=5 a 25, com/sem k-WTA), a SNN é **60× a 300× mais lenta** que o denso em CPU single-thread. A esparsidade de SynOps **nunca** vira velocidade, porque o runtime denso (PyTorch) faz T matmuls completos e não pula neurônios silenciosos. A régua literal do CONTEXT ("rodar em CPU comum") **nunca** é satisfeita neste setup.

### Veredicto preliminar vs critério literal

| Componente do critério | Resultado |
|---|---|
| acc dentro de −2 p.p. | atingível (SNN vanilla T=10: −3.5 p.p., perto) |
| SynOps ≥5× menores | atingível só no extremo (4.81×), MAS custa −17.8 p.p. |
| acc −2pp **E** SynOps ≥5× (simultâneos) | **NÃO** — trade-off impede |
| latência CPU ≤ denso | **NÃO** — sempre 60–300× pior |

→ **FALHA** do critério literal (predição #67 confirmada). A eficiência neuromórfica é, na melhor das hipóteses, marginal e cara em SynOps, e **não se realiza** em CPU von Neumann com runtime denso.

## #69b — teste decisivo: inferência event-driven (sparse) refutada

O caminho que restaria pra um Sucesso era a **inferência event-driven real** (runtime que pula neurônios silenciosos, computando só `W[:, ativos].sum`). `latency_probe.py` mede isso em CPU single-thread (pesos random + esparsidade controlada; latência independe do treino; validado que event-driven == denso no output):

| Config | DenseMLP | SNN denso (T loop) | **SNN event-driven (sparse)** |
|---|---|---|---|
| T=10, k_in=32 | 0.014 ms (1×) | 0.65 ms (46×) | **1.19 ms (84× — PIOR que o denso runtime)** |
| T=5, k_in=16 | 0.016 ms (1×) | 0.34 ms (22×) | **0.57 ms (36×)** |

**Achado contraintuitivo e decisivo:** a inferência event-driven é **mais lenta que o próprio runtime denso** da SNN, e ~84× mais lenta que o MLP. Causa: o overhead de indexação esparsa (`nonzero`, `index_select` de colunas, loop Python sobre T) supera o custo de um único matmul BLAS denso (vetorizado, cache-friendly) para matrizes pequenas. A economia de SynOps **não vira velocidade** — o co-design de runtime via sparse PyTorch em CPU **piora**, não resolve.

## #70 — sweep formal (5 seeds, IC95% bootstrap, treino completo) CONFIRMA

Fashion-MNIST completo, 5 epochs, 5 seeds (`sweep_formal.py`). Confirma o smoke com rigor estatístico:

| Config | acc (5 seeds) | SynOps vs denso | latência CPU | critério |
|---|---|---|---|---|
| DenseMLP | **87.16% ±0.33** CI[86.86, 87.41] | 1× (203k MACs) | 0.020 ms | ref |
| SNN vanilla T=25 | 84.82% ±0.72 CI[84.18, 85.42] | 0.14× (**7× pior**) | 6.54 ms (327×) | FALHA |
| SNN vanilla T=10 | 83.86% ±0.64 CI[83.32, 84.45] | 0.35× (2.9× pior) | 2.66 ms (133×) | FALHA |
| SNN kWTAin T=10 ki=64 | 79.72% ±0.42 CI[79.34, 80.05] | 1.23× menos | 3.25 ms (162×) | FALHA |
| SNN kWTAin T=5 ki=32 | 72.56% ±0.29 CI[72.32, 72.80] | **4.79× menos** | 1.64 ms (82×) | FALHA |

**Análise formal:**
- Com treino completo, o denso chega a **87.16%**. Nenhuma SNN fica dentro de −2 p.p.: a melhor (T=25) é **−2.34 p.p.**, já fora da tolerância — e custa 7× mais SynOps.
- O melhor SynOps (4.79×, quase os 5×) custa **−14.6 p.p.** de acurácia (72.56%). O trade-off acc↔SynOps é confirmado, com IC95% apertados (±0.3–0.7, não é ruído).
- Latência CPU **80–327× pior** em todas. Sempre.
- **As 4 configs falham os 3 componentes do critério simultaneamente.** Não existe config única com acc −2pp **E** SynOps ≥5× **E** latência ≤ denso. O smoke não era artefato de subtreino — o padrão sobrevive ao treino completo e multi-seed.

## Veredicto FINAL (não mais preliminar)

| Componente do critério | Resultado |
|---|---|
| acc −2 p.p. **E** SynOps ≥5× (simultâneos) | **NÃO** — trade-off Pareto íngreme (5× SynOps custa −18 p.p.) |
| latência CPU ≤ denso | **NÃO** — 20–300× pior (denso); **84× pior** mesmo com runtime event-driven |

→ **FALHA decisiva** do critério literal (predição #67 confirmada e fortalecida). **Eficiência radical via SNN não se realiza em CPU von Neumann comum** — nem com runtime denso, nem com sparse/event-driven. A vantagem de SynOps é real apenas como contagem teórica; sua materialização exige **silício neuromórfico dedicado** (Loihi/TrueNorth), onde a esparsidade event-driven é nativa em hardware, não emulada sobre BLAS denso.

**Robustez:** confirmado formalmente em **#70** (5 seeds, IC95% bootstrap, treino completo Fashion-MNIST) — o veredicto não depende mais de smoke. ICs apertados; o trade-off acc↔SynOps e a penalidade de latência CPU sobrevivem ao treino completo e multi-seed.

## No espírito do projeto

Achado negativo **completo e à prova de objeção** — alto valor pela disciplina do projeto ("falhas bem documentadas valem mais que sucessos superficiais"). A tese pós-LLM de "eficiência radical em CPU comum" via SNN **não se sustenta em hardware von Neumann**: a eficiência neuromórfica é uma propriedade de *co-design hardware-algoritmo*, não do algoritmo isolado num framework de propósito geral.

## Fechamento (#70)

Luis (#69 admin) optou por **confirmar formalmente e depois fechar**. Confirmação feita (#70). **Marco 2-B encerrado** — eixo A (eficiência radical via SNN em CPU) respondido com **Falha decisiva, estatisticamente confirmada**. Achado negativo arquivado como documentado/reproduzível (`efficiency_bench.py`, `latency_probe.py`, `sweep_formal.py`, `PAPERS.md`). Registro: STRATEGY.md "Fechamento Marco 2-B — sessão #70", CONTEXT.md §1.5 (🟡→❌).

**Próximo passo — EM ABERTO (decisão do Luis):** Marco 2-B.2 (energia neuromórfica estimada), eixo B/C (treino sem backprop / dados event-based), Marco 2-C (raciocínio temporal — a 4ª e última capacidade não atacada), ou encerrar o projeto. Project Hebb volta a manutenção.
