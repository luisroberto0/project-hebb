# Semana 1 — Sanity Check MNIST com STDP

**Data:** 2026-04-27
**Meta:** Reproduzir Diehl & Cook 2015 em MNIST (≥85% acurácia)
**Status:** ❌ **FALHOU** — STDP colapsou, todos os filtros aprenderam apenas classe 0

---

## Setup

- **Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU, CUDA 12.4
- **Stack:** PyTorch 2.6.0+cu124, Python 3.13, Windows 11
- **Ambiente:** venv local (`C:\Users\pinho\Projects\project-hebb\.venv`)

## Hiperparâmetros

Conforme `config.py` defaults:

```python
# STDP
tau_pre_ms = tau_post_ms = 20.0
A_pre = 0.01, A_post = -0.0105
w_min = 0.0, w_max = 1.0
w_init ~ U(0.0, 0.3)
lateral_inhibition = 0.5

# LIF
tau_mem_ms = 20.0
v_thresh = 1.0, v_reset = 0.0
refractory_ms = 5.0

# Treino
epochs = 1
n_images = 5000 (subset do MNIST train)
n_filters = 100
timesteps = 100
batch_size = 16
```

## Execução

```bash
python experiment_01_oneshot/sanity_mnist.py --device cuda --epochs 1 --n-images 5000 --n-filters 100
```

**Tempo:** 74.2s pretreino (67.4 imgs/s)
**Parâmetros:** 78,400 (100 filtros × 784 pesos/filtro)

## Resultados

### Convergência de pesos

| Step | w_mean | w_std | w_min | w_max |
|------|--------|-------|-------|-------|
| 0    | 0.278  | 0.241 | 0.000 | 1.000 |
| 50   | 0.356  | 0.277 | 0.000 | 1.000 |
| 100  | 0.386  | 0.278 | 0.000 | 1.000 |
| 150  | 0.376  | 0.271 | 0.000 | 1.000 |
| 200  | 0.387  | 0.277 | 0.000 | 1.000 |
| 250  | 0.378  | 0.270 | 0.000 | 1.000 |
| 300  | 0.382  | 0.265 | 0.000 | 1.000 |

Pesos convergiram (mean: 0.278→0.382, std estável ~0.27).

### Atribuição de labels aos filtros

```
Distribuição de labels nos filtros: [100, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

**PROBLEMA CRÍTICO:** TODOS os 100 filtros foram atribuídos à classe 0.

### Acurácia

Script falhou com `UnicodeEncodeError` (emojis ⚠️ não suportados no Windows cp1252) antes de computar acurácia final. No entanto, com 100% dos filtros para classe 0, a acurácia esperada seria ~10% (chance).

---

## Diagnóstico

### 1. Features STDP são esparsas?

**NÃO.** Todos os 100 filtros colapsaram para o mesmo padrão (classe 0). Esperávamos distribuição razoavelmente uniforme de labels (~10 filtros/classe), mas tivemos 100% colapso.

### 2. Pesos convergiram?

**SIM.** Mean e std estabilizaram após ~150 steps. Pesos estão no intervalo [0, 1] conforme esperado. Porém, convergiram para um **padrão não-discriminativo**.

### 3. Distribuição visual dos filtros

Não gerada (checkpoint salvo em `checkpoints/sanity_mnist.pt` mas não inspecionado visualmente ainda).

---

## Hipóteses de causa (ordem de probabilidade)

### Hipótese 1: Inibição lateral ineficaz [MAIS PROVÁVEL]

**Evidência:**
`model.py:118-124` implementa inibição lateral com fator `1e-4`, que é minúsculo:

```python
self.conv.weight.data -= 1e-4 * inh * (spike_per_filter > spike_per_filter.median()).float().view(C_out, 1, 1, 1)
```

**Problema conceitual:**
Inibição lateral em Diehl & Cook 2015 é implementada via **spikes inibitórios na membrana** (winner-take-all instantâneo durante a dinâmica LIF), NÃO como ajuste de pesos pós-STDP. A implementação atual:
- Aplica penalidade insignificante (1e-4) nos pesos após cada timestep
- Só afeta filtros acima da mediana de ativação
- Magnitude é proporcional à média de spikes (pequena por natureza esparsa)

**Consequência:**
Sem competição real entre filtros → todos aprendem o padrão mais frequente/forte → colapso para classe dominante.

**Experimento mínimo:**
Modificar `model.py:124`, trocar `1e-4` por `1e-2` (100× mais forte). Re-rodar sanity check. Esperado: distribuição de labels menos colapsada.

---

### Hipótese 2: Distribuição de classes desbalanceada no subset

**Raciocínio:**
O subset de 5k imagens foi sampleado com `torch.randperm`, sem garantia de balanceamento. Classe 0 pode estar super-representada → STDP converge para padrão mais frequente.

**Experimento mínimo:**
Adicionar em `sanity_mnist.py` após linha 179:
```python
labels_count = [sum(1 for _, label in train_ds if label == c) for c in range(10)]
print(f"Distribuição de classes no subset: {labels_count}")
```
Re-rodar e verificar se classe 0 tem ~50% das amostras.

---

### Hipótese 3: Hiperparâmetros STDP inadequados

**Raciocínio:**
`A_pre = 0.01` e `A_post = -0.0105` são valores de Diehl & Cook, mas podem ser insuficientes para causar divergência entre filtros quando combinados com inibição lateral fraca.

**Experimento mínimo:**
Aumentar `A_pre` de 0.01 para 0.05 (5× mais LTP), manter `A_post = -0.0105`. Re-rodar. Esperado: pesos divergem mais rapidamente, maior variação entre filtros.

---

## Iterações de correção (2026-04-27 sessão 2)

### Hipótese 2 testada e descartada

Adicionado print da distribuição de classes no subset:
```
Distribuição de classes no subset: [499, 554, 517, 521, 464, 475, 469, 510, 531, 460]
```

**Conclusão:** Distribuição balanceada (~500/classe). Hipótese 2 DESCARTADA.

---

### Iteração 1: k=1 WTA na dinâmica LIF

**Modificação:** Implementado k-WTA (k=1) em `model.py:ConvSTDPLayer.forward`:
```python
max_filter_idx = mem.argmax(dim=1, keepdim=True)
wta_mask = torch.zeros_like(mem).scatter_(1, max_filter_idx, 1.0)
spikes_out = spikes_raw * wta_mask
```

Removido decay de pesos pós-STDP em `stdp_update`.

**Resultado:**
- Distribuição: `[24, 23, 11, 9, 3, 5, 7, 13, 1, 4]` ← **TODAS classes representadas!**
- Acurácia: **17.76%** (> chance 10%, < meta 70%)
- Pesos: 0.149→0.115 (decrescendo vs 0.278→0.382 antes)
- Tempo: 63.2s (79.1 imgs/s)

**Análise:** k=1 WTA funciona para distribuir filtros, mas gera esparsidade excessiva → poucos spikes → STDP insuficiente.

---

### Iteração 2: k=5 WTA

**Modificação:** Aumentado k de 1 para 5 (top-5 winners por posição espacial).

**Resultado:**
- Distribuição: `[27, 23, 4, 7, 18, 2, 3, 14, 0, 2]` ← classe 8 tem 0 filtros
- Acurácia: **10.89%** (pior que k=1)
- Tempo: 63.5s

**Análise:** k=5 permitiu mais spikes mas filtros colapsaram MAIS. Não é apenas problema de esparsidade.

---

### Iteração 3 (baseline): SEM WTA

**Modificação:** Removido completamente k-WTA para baseline.

**Resultado:**
- Distribuição: `[100, 0, 0, 0, 0, 0, 0, 0, 0, 0]` ← colapso total (igual original)
- Acurácia: **9.80%**
- Pesos: 0.281→0.384 (crescendo, igual original)

**Análise:** Sem inibição lateral = colapso completo. **k-WTA é NECESSÁRIO.**

---

## Conclusão das iterações

### Melhor configuração testada: k=1 WTA

| Métrica | Valor |
|---------|-------|
| Distribuição | [24, 23, 11, 9, 3, 5, 7, 13, 1, 4] |
| Acurácia | 17.76% |
| Status | ❌ < 70% (meta mínima) |

### Problema identificado

k=1 WTA resolve o colapso de filtros mas não atinge acurácia mínima. **Possíveis causas:**

1. **Número de filtros insuficiente.** Diehl & Cook 2015 usaram 400-6400 filtros, não 100.
2. **Epochs insuficientes.** Paper usa 60k imagens × 3 epochs, testamos 5k × 1 epoch.
3. **Implementação de WTA diverge do paper.** Diehl & Cook usam inibição via condutância sináptica, não masking de spikes.
4. **Codificação Poisson inadequada.** Max rate 100Hz pode ser baixo.

### Releitura do "bloqueio" (correção 2026-04-27)

Usuário corrigiu interpretação prematura: 17.76% com 1/500 do compute do paper original (100 vs 400 filtros, 5k vs 60k imgs, 1 vs 3 epochs, 100 vs 350 timesteps) é convergência incipiente, não falha. Decisão: rodar curva de escala (3 pontos) com k=1 WTA fixo antes de declarar bug.

---

## Curva de escala (k=1 WTA fixo)

### Config A: 200 filtros, 10k imgs, 1 epoch, 100 timesteps

**Comando:**
```bash
python sanity_mnist.py --device cuda --epochs 1 --n-images 10000 --n-filters 200 --seed 42
```

**Resultado:**
- Distribuição classes subset: `[993, 1074, 1016, 1042, 982, 914, 995, 1011, 1005, 968]` (balanceado)
- Distribuição labels: `[36, 40, 6, 11, 18, 19, 17, 43, 2, 8]` (classe 8 quase zerada)
- Acurácia: **9.94%** (chance)
- Tempo: 127.1s (78.7 imgs/s)
- Pesos: 0.149 → 0.114 (decrescendo monotonicamente)

**Esperado:** 25-40%. **Obtido:** chance.

### ANÁLISE CRÍTICA — escala PIOROU resultado

| Config | Filtros | Imgs | Acurácia | Δ vs anterior |
|--------|---------|------|----------|---------------|
| Original WTA | 100 | 5k | 17.76% | baseline |
| Config A | 200 | 10k | 9.94% | **−7.82pp ⬇** |

**Hipótese de causa: LTD dominando LTP sob k=1 WTA.**

Evidência:
1. Pesos **decrescem monotonicamente** (0.149→0.114) durante todo o treino
2. Sem WTA, pesos cresciam (0.281→0.384) — sinal claro de assimetria entre LTP/LTD
3. Com k=1 WTA: pré-spikes Poisson são **frequentes** (max_rate=100Hz × 100 timesteps × 784 pixels), pós-spikes são **raros** (só 1 vencedor por timestep)
4. STDP rule: LTP precisa de coincidência pré-pós; LTD acontece sempre que tem post-spike (mesmo sem pré recente)
5. Com mais filtros (200 vs 100), a probabilidade de cada filtro vencer cai pela metade → ainda menos LTP por filtro → LTD domina ainda mais → pesos morrem mais rápido

**Consequência:** Escalar filtros sem ajustar A_pre/A_post **prejudica** o aprendizado em vez de ajudar.

### Decisão: pausar curva de escala, atacar causa raiz

Config B (400 filtros, 30k imgs) muito provavelmente vai piorar ainda mais (mesma dinâmica, mais filtros). Vale ajustar **balanço LTP/LTD** antes de continuar escalando.

**Hipóteses concorrentes a testar (cheapest-first):**

1. **H4: Aumentar A_pre.** Razão: compensar dominância de LTD em regime esparso. Custo: 1 char no config.py.
2. **H5: Bug em label assignment** (sugerido por Luis). Razão: distribuição de filtros parece OK mas acurácia é chance — pode ser que o assignment esteja zerando a informação. Custo: ler código + adicionar prints.
3. **H6: Bug em evaluate** (mesmo raciocínio). Custo: idem.

Aguardando decisão humana.

---

## Referências

- Diehl, P. U., & Cook, M. (2015). *Unsupervised learning of digit recognition using spike-timing-dependent plasticity*. Frontiers in Computational Neuroscience.
- Song, S., Miller, K. D., & Abbott, L. F. (2000). *Competitive Hebbian learning through spike-timing-dependent synaptic plasticity*. Nature Neuroscience.
