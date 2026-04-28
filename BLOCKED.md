# BLOCKED — Experimento 01 Semana 1

**Data:** 2026-04-27
**Status:** Bloqueado após 3 iterações de correção
**Melhor resultado:** 17.76% acurácia (k=1 WTA) vs meta 85%

---

## Resumo do bloqueio

Sanity check Diehl & Cook 2015 em MNIST falhou após 3 iterações:

| Iteração | Configuração | Dist labels | Acurácia | Observação |
|----------|--------------|-------------|----------|------------|
| Original | Decay pesos (1e-4) | [100,0,0,0,0,0,0,0,0,0] | ~10% | Colapso total |
| 1 | k=1 WTA | [24,23,11,9,3,5,7,13,1,4] | **17.76%** | Todas classes, acurácia baixa |
| 2 | k=5 WTA | [27,23,4,7,18,2,3,14,0,2] | 10.89% | Pior que k=1 |
| 3 | Sem WTA | [100,0,0,0,0,0,0,0,0,0] | 9.80% | Baseline, colapso |

**Melhor:** k=1 WTA distribui filtros entre todas as classes, mas acurácia ainda muito abaixo da meta.

---

## Hipóteses do bloqueio

### H1: Número de filtros insuficiente [MAIS PROVÁVEL]

**Evidência:**
- Diehl & Cook 2015 usaram 400-6400 filtros
- Testamos apenas 100 filtros
- Distribuição [24,23,11,9,3,5,7,13,1,4] mostra desequilíbrio: classes 4 e 8 têm apenas 3 e 1 filtros

**Experimento:** Re-rodar com 400 filtros, mesmo config (5k imgs, 1 epoch, k=1 WTA).
**Custo:** ~5 min GPU.

---

### H2: Pretreino curto demais

**Evidência:**
- Diehl & Cook usaram 60k imgs × 3 epochs
- Testamos 5k imgs × 1 epoch (12× menos dados)
- Pesos k=1 WTA estão decrescendo (0.149→0.115) em vez de crescer

**Experimento:** Re-rodar com 10k imgs, 3 epochs, k=1 WTA, 100 filtros.
**Custo:** ~15 min GPU.

---

### H3: Implementação de WTA diverge do paper

**Evidência:**
- Diehl & Cook implementam inibição via **condutância sináptica** (subtração contínua de membrana)
- Nossa implementação usa **masking binário** de spikes (hard WTA)
- Masking pode ser excessivamente agressivo

**Experimento:** Implementar soft WTA:
```python
# Em vez de: spikes_out = spikes_raw * wta_mask
# Fazer: mem = mem - alpha * (1 - wta_mask) * mem.max(dim=1, keepdim=True)[0]
```

**Custo:** ~2h implementação + teste.

---

### H4: Codificação Poisson inadequada

**Evidência:**
- Max rate 100Hz pode ser baixo para MNIST (poucos spikes por timestep)
- Diehl & Cook não especificam max rate no paper

**Experimento:** Aumentar `max_rate_hz` de 100 para 200 em `config.py:SpikeConfig`.
**Custo:** ~5 min GPU.

---

## Experimentos recomendados (ordem de prioridade)

1. **H1 + H2 combinados:** 400 filtros, 10k imgs, 3 epochs, k=1 WTA
   **Justificativa:** Testa as duas hipóteses mais prováveis de uma vez. Custo: ~30 min GPU.

2. **H4:** Aumentar max_rate para 200Hz
   **Justificativa:** Mudança trivial, rápida validação. Custo: ~5 min.

3. **H3:** Implementar soft WTA via condutância
   **Justificativa:** Mais fiel ao paper mas mais complexo. Custo: ~2h.

---

## Decisão necessária

**Aprovar uma das opções:**

- [ ] **Opção A:** Escalar filtros + dados (H1+H2)
- [ ] **Opção B:** Testar max_rate primeiro (H4), depois escalar
- [ ] **Opção C:** Reimplementar inibição lateral via condutância (H3)
- [ ] **Opção D:** Abandonar Diehl & Cook, tentar outra abordagem

**Aguardando input humano.**

---

## Código modificado (commitado)

- `experiment_01_oneshot/model.py`: Implementado k-WTA na dinâmica LIF, removido decay de pesos
- `experiment_01_oneshot/sanity_mnist.py`: Corrigido UTF-8, adicionado print distribuição de classes
- `PLAN.md`: Registrada decisão arquitetural k-WTA
- `experiment_01_oneshot/WEEKLY-1.md`: Documentadas 3 iterações + conclusão

**Branch:** main
**Último commit:** (pendente)
