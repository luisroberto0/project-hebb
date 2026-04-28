# BLOCKED — Experimento 01 Semana 1

**Data atualizada:** 2026-04-27 (sessão #5 — H_combo descartada)
**Status:** Bloqueado, espaço de hipóteses HP-paramétricas exaurido
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
| #5 | H_combo (homeostasis + A_post=-0.001) | [87,1,1,2,1,0,0,6,2,0] | 13.76% | Pior que componentes isolados |

---

## Hipóteses descartadas (testadas empiricamente)

### ~~H_assignment: Bug em assign_labels ou evaluate~~

**Status:** ✗ DESCARTADO em `tests/test_assignment.py`. Três casos sintéticos passam.

### ~~H_balance: Rebalancear LTP/LTD sozinho resolve colapso~~

**Status:** ✗ DESCARTADO. Trade-off estrutural irrecuperável sem homeostasis.

### ~~H_dose: Escalar filtros/dados melhora acurácia~~

**Status:** ✗ DESCARTADO. Config A piorou (17.76% → 9.94%).

### ~~H_homeostasis (sozinha): Adaptive threshold resolve colapso e destrava acurácia~~

**Status:** ✗ DESCARTADO PARCIALMENTE. Mecanicamente eficaz (distribuição [36,10,11,7,7,7,6,9,3,4]) mas acurácia continua 16.39%.

### ~~H_combo: Homeostasis + LTP/LTD calibrado simultaneamente~~

**Status:** ✗ DESCARTADO empiricamente em sessão #5. Testado com theta_plus=0.0005 + A_post=-0.001 (R=10): acurácia 13.76%, PIOR que ambos componentes isolados. Padrão observado: **quanto mais LTP relativo a LTD, mais colapso** (não menos). Theta std subiu 4× (1.20 → 5.33), 16% dos filtros nunca dispararam, distribuição colapsou [87,1,...]. Homeostasis não compensa rich-get-richer no regime esparso PyTorch independente do tuning.

---

## Hipóteses VIVAS (apenas 2 restantes)

### H_paper_replicability: Reimplementar e validar contra Brian2 [MAIS HONESTA]

**Evidência crescente da relevância:**
- 5 sessões de tuning hiperparamétrico (k-WTA, A_pre/A_post, theta_plus, combos) não destravaram nada acima de 17.76%
- Espaço de hiperparâmetros parece exaurido — qualquer ajuste adicional é especulação cega
- Diehl & Cook 2015 publicaram com 85%+; nossa implementação PyTorch tem alguma divergência arquitetural ou algorítmica não-óbvia
- Brian2 é a referência canônica do paper, com ordem de operações e dinâmicas exatas

**O que validar contra Brian2:**
1. Roda Diehl & Cook 2015 puro em Brian2, atinge ~85% como o paper reporta
2. Compara passo-a-passo com nossa implementação PyTorch
3. Identifica divergência (provavelmente: refractory exata, ordem de update de traces vs spikes, ou algo na inhibitória)
4. Porta correção de volta pra PyTorch

**Custo:** ~1 semana (decisão arquitetural). Mas é o único caminho que elimina ambiguidade.

### H_arch: Pular Semana 1 (kernel=28 é caso degenerado, ir direto pra Omniglot conv)

**Evidência:**
- Output spatial (1,1) torna k-WTA degenerado (compete sobre única posição)
- Diehl & Cook usam FC genuíno; nosso "Conv-com-kernel-completo" pode introduzir distorções não-óbvias
- Em conv real (kernel=5 + pool), k-WTA pelo menos permite filtros diferentes por região (mais perto da arquitetura prevista pra Omniglot)

**Risco:** se Omniglot conv também não funcionar, voltamos a estar bloqueados sem ter validado a Semana 1. Mas pelo menos estaríamos atacando o problema real do experimento (Omniglot, não MNIST sanity).

**Custo:** zero (só usar `train.py` que já existe, com a infraestrutura validada).

---

## Decisão necessária (próxima sessão)

**Opção C — H_paper_replicability (Brian2):**
Caminho rigoroso. Custo alto (~1 semana). Elimina ambiguidade definitivamente.

**Opção B — H_arch (pular pra Omniglot):**
Custo zero. Aceita 17.76% MNIST como "caso patológico de kernel completo". Move pra arquitetura conv real prevista no PLAN.md.

**Opção D — Reduzir escopo Semana 1:**
Documenta MNIST com kernel=28 como anti-padrão e move. Equivalente a B na prática.

**Opções A, H_combo, H_homeostasis, H_balance, H_dose, H_assignment:** todas descartadas empiricamente.

---

## Estado atual do código

- `experiment_01_oneshot/model.py`: k=1 WTA + adaptive threshold homeostático ativo
- `experiment_01_oneshot/config.py`: theta_plus=0.0005, tau_theta=1e7, **A_pre=0.01, A_post=-0.0105** (restaurado paper original como melhor estado conhecido)
- `experiment_01_oneshot/sanity_mnist.py`: print de diagnóstico do theta + UTF-8 fix + distribuição classes
- `experiment_01_oneshot/tests/test_assignment.py`: ✓ 3/3 passam
- `experiment_01_oneshot/tests/test_spike_balance.py`: instrumentação de razão pré:pós

**Branch:** main, sincronizado com origin/main após esta sessão.
