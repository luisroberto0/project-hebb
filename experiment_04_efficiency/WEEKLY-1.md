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

### Caveats e próximos passos

- **Smoke, não final:** 2-3 epochs, quick subset, 1 seed. Treino completo subiria a acc de todos uniformemente, mas (a) o trade-off SynOps↔acc é estrutural, (b) a latência CPU é determinística pela arquitetura (T matmuls), não pelo treino. O padrão não deve inverter.
- **#70-#73 (se Luis quiser confirmar):** sweep formal (T∈{5,10,25,50} × k_in × 5 seeds, IC95%), curva de Pareto completa, energia estimada (SynOps×E_AC vs MACs×E_MAC).
- **O caminho que o achado aponta para um Sucesso:** inferência **event-driven real** (sparse runtime que pula neurônios silenciosos), i.e. co-design de runtime. Testar `torch.sparse` ou kernel custom seria o teste decisivo da tese "a vantagem de SynOps se realiza com o runtime certo". Predição: ainda perde em CPU pra rede pequena (overhead de formato esparso > economia), mas é a prova à prova de objeção.

**No espírito do projeto:** achado negativo bem caracterizado. A eficiência radical via SNN rate-coded em hardware von Neumann comum **exige co-design de hardware/runtime event-driven** — não é uma propriedade do algoritmo isolado.
