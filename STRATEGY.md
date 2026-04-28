# Estratégia da pesquisa STDP+Hopfield

## Decisão tomada em 2026-04-27 (sessão #7, encerramento)

Combinar tuning + opção de pivot:

1. **Próximas 2-3 sessões:** testar H_theta_omn e H_norm (custo baixo, alto ROI)
2. **Se H_theta_omn e H_norm não destravarem:** considerar pivot pra abordagem adjacente (Hopfield puro, meta-learning bio, ou prototypical com features esparsas)
3. **Brian2 (validação contra paper original) fica como último recurso** — alto custo (~1 semana), só justificável se pivot também falhar

## Critério pra cada decisão futura

- Sessão = 60-90 min, max 2x/semana
- Após cada sessão: atualizar PLAN.md "Notas de iteração"
- Se 3 sessões consecutivas sem progresso (sinal acima de chance), revisar STRATEGY.md

---

## Pós-Sessão #9 (2026-04-28): Sinal real, mecanismo a investigar

### O que foi confirmado

`tau_theta_ms=1e4` produz **primeiro sinal acima de chance do projeto**:

- 5w1s = 35.98% IC95% [35.17, 36.79], z≈1.3 (+15.98 p.p. acima de chance)
- 20w1s = 9.80% IC95% [9.58, 10.01], z≈1.4 (+4.80 p.p. acima de chance)

**Robusto em 3 verificações ortogonais:**
- V1 (eval seed=100, mesmo ckpt): 36.06% — não é ruído de seed de eval
- V2 (retrain seed=43, eval seed=42): 35.96% — não é combinação específica de seeds
- V3 (20w1s, escala dificuldade): mantém z≈1.4 — sinal escala

Decisão arquitetural fixada em PLAN.md: `tau_theta_ms=1e4`.

### O que NÃO foi confirmado: mecanismo

Pesos saturam em μ=0.999 σ=0.001 (filtros teoricamente indistinguíveis), theta range é ~0.1 entre filtros, ainda assim sinal aparece. **Não sabemos por quê.**

Conjectura inicial "theta diferenciada carrega sinal" foi **refutada** pela V2: seed=43 produziu theta range ainda mais apertado [20.78, 20.86] que seed=42 [20.55, 20.68], com acc idêntica.

Hipóteses vivas sobre o mecanismo:

- **H_init_bias:** initialization bias preservado na saturação (qual filtro satura primeiro / ordem temporal codifica algo discriminativo, mesmo que valores finais sejam quase iguais)
- **H_lif_dynamics:** spike timing residual via dinâmica LIF (input × peso saturado + reset + ruído gera spike trains discriminativos mesmo com pesos ~iguais)
- **H_pooling_amp:** σ=0.001 amplificado pela não-linearidade do segundo layer + pooling
- **H_artifact:** sinal é artefato de eval/dataset (improvável dado V1/V3, mas precisa ser descartado)

### Próxima sessão: dedicada exclusivamente a investigar mecanismo

**Não escalar, não tunar, não modificar config até mecanismo ser identificado.** A regra é honestidade metodológica: se o sinal vem de artefato, "amplificá-lo" é desperdício.

3 testes de ablação, todos no mesmo checkpoint Iter 1 (`stdp_model_iter1_seed42.pt`):

| Ablação | Como | O que testa | Predição se sinal real |
|---|---|---|---|
| **A1 — `_proj` zerada** | substituir projeção final (embedding_dim=64) por identidade ou zeros antes do Hopfield | sinal vem do feature space pós-conv ou da projeção? | acc cai pra chance se proj é o canal; mantém ~36% se conv carrega |
| **A2 — zero conv** | substituir `layer1.conv.weight` e `layer2.conv.weight` por zeros, manter theta treinada | pesos importam? | acc cai pra chance (zero spikes propagam) |
| **A3 — random conv** | substituir pesos treinados por re-initialization aleatória (`w_init_low/high`), manter theta treinada | pesos *treinados* importam, ou só a presença de qualquer estrutura conv? | acc cai pra chance se pesos treinados carregam sinal; mantém ~36% se theta sozinha basta |

### Critério de decisão pós-ablação

- **Sinal sobrevive aos 3 testes (acc cai apropriadamente em A2; cai ou se mantém conforme predição em A1/A3, mas o resultado é interpretável e consistente):** sinal é real, mecanismo identificado (ou parcialmente identificado), decisão `tau_theta=1e4` se consolida, próxima sessão escala/tuna.

- **Algum teste de ablação atinge ~36% sem o componente ablacionado:** "descoberta" é artefato (a parte ablacionada não era necessária pra produzir o sinal — sinal está em outro canal não-conv, possivelmente eval ou pipeline). Reverte decisão `tau_theta=1e4`, restaura `tau_theta=1e7`, marca H_tau_theta como "sinal medido mas mecanismo artefatual" em vez de descartada-sucesso. Próxima sessão pivota pra H_norm.

- **Resultado intermediário/ambíguo:** documenta padrão observado, **não fixa decisão**, agenda mais ablações antes de seguir.
