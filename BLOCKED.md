# BLOCKED — Experimento 01 Semana 1

**Data atualizada:** 2026-04-27 (sessão #4 — homeostasis implementada)
**Status:** Bloqueado, com mais um mecanismo validado mas não suficiente
**Melhor resultado:** 17.76% acurácia (config baseline) vs meta 85%

---

## Resumo do progresso

| Sessão | Configuração | Distribuição labels | Acurácia | Notas |
|--------|--------------|---------------------|----------|-------|
| #1 | Decay pesos (1e-4) | [100,0,...,0] | ~10% | Colapso (LTD inativa) |
| #2 | k=1 WTA (sem homeostasis) | [24,23,11,9,3,5,7,13,1,4] | **17.76%** | Melhor consolidado |
| #2 | k=5 WTA | [27,23,4,7,18,2,3,14,0,2] | 10.89% | k=5 pior que k=1 |
| #3 | Config A (200/10k) | [36,40,6,11,18,19,17,43,2,8] | 9.94% | Escala piorou |
| #3 | R=10 (A_post=-0.00157) | [88,1,1,0,0,1,4,3,0,2] | 11.51% | Pesos crescem mas filtros colapsam |
| #3 | R=3 (A_post=-0.00333) | [94,2,0,0,0,0,0,3,1,0] | 11.36% | Mesmo padrão |
| #4 | homeostasis theta_plus=0.05 | [100,0,...,0] | 9.80% | Theta saturou em 267 |
| #4 | homeostasis theta_plus=0.0005 | [36,10,11,7,7,7,6,9,3,4] | 16.39% | Distribuição uniforme, acc igual |

---

## Hipóteses descartadas (testadas empiricamente)

### ~~H_assignment: Bug em assign_labels ou evaluate~~

**Status:** ✗ DESCARTADO em `tests/test_assignment.py`. Três casos sintéticos passam.

### ~~H_balance: Rebalancear LTP/LTD sozinho resolve colapso~~

**Status:** ✗ DESCARTADO. Trade-off estrutural irrecuperável sem homeostasis.

### ~~H_dose: Escalar filtros/dados melhora acurácia~~

**Status:** ✗ DESCARTADO. Config A piorou (17.76% → 9.94%).

### ~~H_homeostasis (sozinha): Adaptive threshold resolve colapso e destrava acurácia~~

**Status:** ✗ DESCARTADO PARCIALMENTE. Implementada (sessão #4), mecanicamente eficaz: theta com variância (mean=2.48, std=1.20), distribuição mais uniforme [36,10,11,7,7,7,6,9,3,4]. Mas acurácia continua ~17%. Razão: homeostasis força filtros a vencerem menos → LTP/LTD imbalance re-emerge por filtro.

---

## Hipóteses VIVAS

### H_combo: Homeostasis + LTP/LTD calibrado simultaneamente [MAIS PROVÁVEL agora]

**Evidência:**
- Sessão #3 isolou que LTP/LTD desbalanceia por R=10 (com k=1 WTA, sem homeostasis)
- Sessão #4 mostrou que homeostasis funciona mas re-expõe LTP/LTD imbalance individualmente
- **Conjectura:** se aplicarmos AMBAS correções juntas (homeostasis ativa + A_post reduzido), os dois problemas se resolvem
- Sessão #4 NÃO testou isso — só rodou homeostasis com hiperparâmetros default

**Experimento mínimo:**
1. Manter homeostasis (theta_plus=0.0005, tau_theta=1e7)
2. Rebalancear: A_pre=0.01, A_post=-0.001 (razão R=10, similar a iteração da sessão #3)
3. Rodar baseline 100/5k. Critério: distribuição uniforme (já validado) + pesos crescentes + acurácia > 30%.

**Custo:** ~5 min GPU. Trivial. Não foi feito por restrição da spec da sessão #4 (max 2 iter de homeostasis).

### H_arch: Kernel=28 (FC equivalente) é caso degenerado pra k-WTA

**Evidência:**
- Output spatial (1,1) → k-WTA compete sobre única posição
- Diehl & Cook usam FC genuíno; nossa "Conv-com-kernel-completo" pode distorcer dinâmica
- Em conv real (kernel=5 + pool), k-WTA permite filtros diferentes por região

**Experimento:** Pular Semana 1, ir direto pra Semana 2 (Omniglot conv real). Aceitar que sanity MNIST com kernel=28 é caso patológico.

### H_paper_replicability: Reimplementar e validar contra Brian2

**Evidência:**
- Diehl & Cook fornecem código de referência em Brian2
- Detalhes sutis podem ter sido perdidos no port pra PyTorch (refractory exata, ordem de operações)

**Custo:** ~1 semana (decisão arquitetural não-trivial).

---

## Decisão necessária (próxima sessão)

**Opção A — H_combo (RECOMENDADA por menor custo):**
~5 min. Combina dois ajustes já implementados (homeostasis + LTP/LTD ratio). Se resolver, fecha Semana 1. Se não, hipóteses A_combo descartada e descemos pra B/C.

**Opção B — H_arch (pular pra Omniglot conv):**
Aceitar que MNIST com kernel=28 é caso degenerado. Mover pra Semana 2 com nota no protocolo. Custo zero, mas pula validação.

**Opção C — H_paper_replicability (Brian2):**
~1 semana. Elimina ambiguidade mas custo alto.

**Opção D — Reduzir escopo Semana 1:**
Aceitar 17.76% como "sanity passou parcialmente" e mover. Equivalente a B na prática.

---

## Estado atual do código

- `experiment_01_oneshot/model.py`: k=1 WTA + adaptive threshold homeostático ativo
- `experiment_01_oneshot/config.py`: theta_plus=0.0005, tau_theta=1e7, A_pre=0.01, A_post=-0.0105
- `experiment_01_oneshot/sanity_mnist.py`: print de diagnóstico do theta
- `experiment_01_oneshot/tests/test_assignment.py`: ✓ 3/3 passam
- `experiment_01_oneshot/tests/test_spike_balance.py`: instrumentação de razão pré:pós

**Branch:** main, sincronizado com origin/main após esta sessão.
