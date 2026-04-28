# BLOCKED — Experimento 01 Semana 1

**Data atualizada:** 2026-04-27 (sessão #3 — auditoria + rebalance LTP/LTD)
**Status:** Bloqueado, mas com causa raiz identificada e validada
**Melhor resultado:** 17.76% acurácia (config baseline) vs meta 85%

---

## Resumo do progresso

| Iteração | Configuração | Distribuição labels | Acurácia | Notas |
|----------|--------------|---------------------|----------|-------|
| Original | Decay pesos (1e-4) | [100,0,...,0] | ~10% | Colapso (LTD inativa) |
| Sessão 2: k=1 WTA | A_pre=.01, A_post=-.0105 | [24,23,11,9,3,5,7,13,1,4] | **17.76%** | **Melhor estado** |
| Sessão 2: k=5 WTA | mesmos valores | [27,23,4,7,18,2,3,14,0,2] | 10.89% | k=5 pior que k=1 |
| Sessão 2: sem WTA | mesmos valores | [100,0,...,0] | 9.80% | Confirma WTA necessário |
| Sessão 3: Config A | 200 filtros, 10k imgs | [36,40,6,11,18,19,17,43,2,8] | 9.94% | Escala piorou (regressão) |
| Sessão 3: R=10 | A_post=-0.00157 | [88,1,1,0,0,1,4,3,0,2] | 11.51% | Pesos crescem mas filtros colapsam |
| Sessão 3: R=3 | A_post=-0.00333 | [94,2,0,0,0,0,0,3,1,0] | 11.36% | Mesmo padrão |

---

## Hipóteses descartadas (testadas empiricamente nesta sessão)

### ~~H_assignment: Bug em assign_labels ou evaluate~~

**Status:** ✗ DESCARTADO em `tests/test_assignment.py`. Três casos sintéticos passam com 100% (perfeito), 10% (random), 100% (sinal fraco). Pipeline de classificação está correto.

### ~~H_balance: Rebalancear LTP/LTD resolve colapso~~

**Status:** ✗ DESCARTADO. Razão pré:pós medida = 10.1 (`tests/test_spike_balance.py`). Tentativas com R=10 e R=3 mostraram trade-off estrutural:
- LTP < LTD → pesos morrem (acurácia 17.76% mas instável)
- LTP > LTD → 1 filtro vence sempre (acurácia ~11%, chance)

Não existe ratio que resolva ambos os problemas simultaneamente, porque a causa raiz está em outro lugar.

### ~~H_dose: Escalar filtros/dados melhora acurácia~~

**Status:** ✗ DESCARTADO. Config A (200 filtros, 10k imgs) PIOROU vs baseline (17.76% → 9.94%). Mais filtros agrava o problema porque cada um vence menos vezes em k=1 WTA.

---

## Hipóteses VIVAS

### H_homeostasis: Falta adaptive threshold homeostático [MAIS PROVÁVEL]

**Evidência:**
- Diehl & Cook 2015 §2.3 implementa thresholds adaptativos: cada filtro tem `θ_i` que cresce a cada spike e decai com tempo
- Função: forçar todos os filtros a disparar aproximadamente igualmente ao longo do treino
- Sem isso, k-WTA + STDP é instável independente do ratio LTP/LTD (validado nas tentativas R=10 e R=3)

**Mecanismo proposto** (a implementar em `model.py:ConvSTDPLayer`):
```python
# Adicionar atributo theta: (C_out,) inicializado em 0
# Modificar threshold efetivo: v_thresh_eff = v_thresh + theta[c]
# Após cada timestep:
#   theta += spike_count_per_filter * theta_plus  # cresce com spikes
#   theta *= exp(-dt / tau_theta)                 # decai com tempo
```

**Custo:** ~2h implementação + teste.

### H_arch: Kernel=28 (FC equivalente) é ruim pra k-WTA

**Evidência:**
- Com kernel=28, output spatial é (1,1) → k-WTA compete sobre **uma única posição**
- Diehl & Cook usam arquitetura full-connected legítima (sem conv) — talvez nossa adaptação convolucional com kernel completo distorça a dinâmica
- Em arquitetura conv real (kernel=5 com pooling), k-WTA permite filtros diferentes para regiões diferentes

**Custo:** Já é a arquitetura prevista pra Omniglot (Semana 2). Pode ser que Semana 1 deva ser pulada e ir direto pra setup conv real.

### H_paper_replicability: Reimplementar tudo em Brian2

**Evidência:**
- Diehl & Cook fornecem código de referência em Brian2/Brian
- Pode haver detalhes sutis (ordem de operações, refractory exata, etc.) que o port pra PyTorch puro perdeu
- Brian2 é mais lento mas executa o paper exatamente como descrito

**Custo:** ~1 semana (decisão arquitetural não-trivial documentada em PLAN.md).

---

## Decisão necessária (próxima sessão)

**Opção A — Implementar adaptive threshold (H_homeostasis):**
~2h. Mais prometedora baseado no diagnóstico. Continua no caminho atual.

**Opção B — Pular Semana 1, ir direto pra arquitetura Omniglot (H_arch):**
Aceitar que MNIST simplificado com kernel=28 é caso degenerado. Implementar conv real (kernel=5 + pool) e testar diretamente em Omniglot — onde a tese principal vive de qualquer jeito.

**Opção C — Validar contra Brian2 (H_paper_replicability):**
Reimplementar Diehl & Cook em Brian2 puro, validar acurácia ~85%, depois portar componentes verificados de volta pra PyTorch. Custo alto mas elimina ambiguidade.

**Opção D — Abandonar Semana 1 sanity, declarar escopo reduzido:**
Anotar que k=1 WTA convolucional global é caso patológico que não reproduz Diehl & Cook, mover pra Semana 2 com ressalva no protocolo.

---

## Estado atual do código

- `experiment_01_oneshot/model.py`: k=1 WTA ativo, STDP sem decay artificial
- `experiment_01_oneshot/config.py`: A_pre=0.01, A_post=-0.0105 (paper original, melhor estado)
- `experiment_01_oneshot/sanity_mnist.py`: pseudocódigo de assignment/evaluate documentado, UTF-8 OK
- `experiment_01_oneshot/tests/test_assignment.py`: ✓ 3/3 passam
- `experiment_01_oneshot/tests/test_spike_balance.py`: instrumentação de razão pré:pós

**Branch:** main, sincronizado com origin/main após esta sessão.
