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

## Decisão

**Não mudar código ainda.** Próximos passos:

1. ✅ **Documentar diagnóstico** (este arquivo)
2. ⏭️ **Testar Hipótese 2 primeiro** (verificar distribuição de classes - mais barato)
3. ⏭️ **Testar Hipótese 1** (consertar inibição lateral - mais impacto esperado)
4. ⏭️ **Visualizar filtros** do checkpoint atual (`checkpoints/sanity_mnist.pt`) pra confirmar colapso visual

Aguardando OK para prosseguir.

---

## Referências

- Diehl, P. U., & Cook, M. (2015). *Unsupervised learning of digit recognition using spike-timing-dependent plasticity*. Frontiers in Computational Neuroscience.
- Song, S., Miller, K. D., & Abbott, L. F. (2000). *Competitive Hebbian learning through spike-timing-dependent synaptic plasticity*. Nature Neuroscience.
