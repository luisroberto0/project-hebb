# Semana 2 — Adaptação pra Omniglot (conv real)

**Iniciada:** 2026-04-27 (sessão #7)
**Pré-condições:** Semana 1 fechada como caso patológico (`WEEKLY-1.md`), infra Omniglot validada via inspeção (`WEEKLY-2-NEXT.md`).

---

## Sessão #7 (smoke test + calibração)

### Etapa 0 — Smoke test do pipeline Omniglot ✅

**Objetivo:** confirmar que o pipeline executa end-to-end na arquitetura conv real (kernel=5 + pool, 2 layers).

**Configuração quick:** 500 imgs / 1 epoch / pretreino + baselines + eval.

**Resultados:**

| Componente | Status | Saída |
|------------|--------|-------|
| `validate_environment.py` | ✅ | PyTorch 2.6.0+cu124, RTX 4070 (8.9), 2933 GFLOPS, todas libs core OK |
| `evaluate.py` (random, no ckpt) | ✅ | 20.92% (chance ≈20%, z≈0.2) — pipeline fecha |
| `baselines.py pixel_knn` | ✅ | 45.76% ± 10.47% — pixel kNN funciona |
| `baselines.py proto_net` (500 train eps) | ✅ | **85.88% ± 10.95%** — SOTA baseline forte estabelecido |
| `train.py` (500 imgs, 1 epoch) | ⚠️ | Executou (17.3s, 28.9 imgs/s) MAS w1 colapsou (μ=0.000) |
| `evaluate.py` (com ckpt colapsado) | ✅ pipeline / ❌ resultado | 20.00% exato (predição constante, IC=[20,20], z=inf) |

**Conclusão Etapa 0:** Pipeline integra perfeitamente. Bug crítico de pretreino exposto: layer 1 colapsa em segundos.

---

### Bug fix: lambda não-picklável em `data.py`

`build_transforms` usava `transforms.Lambda(lambda x: 1.0 - x)` que não é picklável no Windows multiprocessing (spawn). DataLoader com `num_workers=2` falhava com `AttributeError: Can't get local object 'build_transforms.<locals>.<lambda>'`.

**Fix:** substituído por função module-level `_invert_intensity`. Mudança mínima, semântica idêntica.

---

### Bug ativo (a resolver na Etapa 1): layer 1 colapsa no pretreino

Pretreino de 500 imgs, 1 epoch:
```
epoch 0 step    0  w1=μ0.000/σ0.000  w2=μ0.609/σ0.417  seen=8  elapsed=4.6s
```

Layer 1 weights vão pra zero já no primeiro batch (8 imagens). Layer 2 cresce para μ=0.609 — mas alimentada por entrada zero, não significa nada (pesos saindo da inicialização interagindo com noise).

Eval com este checkpoint: 20.00% exato com IC zero — confirma que o modelo virou função constante (zero spikes propagam zero ativação).

**Hipótese mecânica (a confirmar na Etapa 1):**
Em conv real, regime de spikes é radicalmente diferente do MNIST:
- **MNIST kernel=28:** 1 winner global por timestep × 100 ts = 100 pós-spikes/imagem
- **Omniglot kernel=5, padding=2:** 1 winner por POSIÇÃO espacial × 28×28 posições × 100 ts = 78400 pós-spikes/imagem total

Pré-spikes da entrada Omniglot (após inversão, fundo preto/traços brancos): ~5-10% dos 784 pixels com intensidade alta → ~50-100 pré-spikes/timestep × 100 ts = 5000-10000 pré-spikes/imagem.

**Razão estimada R = pré:pós ≈ 5000/78400 = 0.06** (vs. 10.1 em MNIST kernel=28 e ~1 no paper Diehl & Cook).

Com `A_pre=0.01, A_post=-0.0105` (paper original) e essa razão invertida, intuição diz que LTD deveria ficar fraco, mas o produto `apost · pre_patches` no STDP rule depende não-linearmente das densidades. Análise teórica não basta, precisa medir.

**Próximo passo (Etapa 1):** adaptar `tests/test_spike_balance.py` pra `STDPHopfieldModel.layer1`, medir empiricamente, decidir calibração.

---

### Etapa 1 — Calibração do regime de spikes ⚠️ (parcial)

**Objetivo:** medir empiricamente a razão R = pré:pós-spikes nas duas layers, ajustar `A_pre/A_post/theta_plus` pra estabilizar pretreino.

**Implementação:** criado `tests/test_spike_balance_omniglot.py` (análogo ao da Semana 1, mas pra `STDPHopfieldModel` em Omniglot, mede ambas layers separadamente).

#### 1.1 — Medição inicial (config Semana 1)

```
Config: A_pre=0.01, A_post=-0.0105, theta_plus=0.0005
```

| Layer | Pré-spikes/img | Pós-spikes/img | Razão R |
|-------|---------------|---------------|---------|
| Layer 1 (1×28×28 → 8×28×28) | 575.8 | 382.4 | **1.51** |
| Layer 2 (8×14×14 → 16×14×14) | 280.4 | 403.7 | **0.69** |

Comparações:
- Estimativa teórica prévia: R≈0.06 (errado: assumi muito pré-spike e ignorei threshold dinâmica)
- MNIST kernel=28 (Semana 1): R=10.1
- Paper Diehl & Cook: R≈1

Conclusão: regime conv real está MAIS PERTO do paper (R≈1) do que do MNIST sanity. Não é o problema dominante.

#### 1.2 — Trace step-by-step de layer 1 (100 timesteps em 1 batch)

| t | pre | post | w1 mean | w1 max | theta max |
|---|-----|------|---------|--------|-----------|
| 0 | 40 | 1 | 0.137 | 0.298 | 0.001 |
| 5 | 39 | 387 | 0.270 | 0.892 | 0.133 |
| 10 | 39 | 1034 | 0.624 | **1.000** | 0.819 |
| 20 | 35 | 995 | 0.654 | 1.000 | 1.747 |
| 30 | 37 | 722 | 0.415 | 1.000 | 2.203 |
| 50 | 31 | 295 | 0.233 | 1.000 | 2.916 |
| 70 | 42 | **0** | **0.000** | 0.000 | 2.927 |
| 99 | 38 | 0 | 0.000 | 0.000 | 2.927 |

**Mecanismo identificado:** oscilação instável.
1. Inicialização baixa, primeiros spikes esparsos
2. LTP cresce rápido (post denso × apre crescente)
3. Pesos saturam em 10 timesteps → todos os filtros ficam super-responsivos
4. Theta sobe junto, mas atrasada
5. LTD ataca (apost alto × pre denso) + theta alta inibe disparos
6. Pesos colapsam pra 0
7. Theta locked-out em ~2.9 (não decai com tau_theta=1e7), sistema "morto"

**Calculo de magnitude:** A_pre × delta_LTP por timestep ≈ 0.01 × ~85 = ~0.86 por param. Pesos cruzam 1.0 em 5-10 timesteps. Não há tempo pra homeostasis ter efeito.

#### 1.3 — Calibração: A_pre 100× menor

Diagnóstico aponta que A_pre é grande demais pra esse regime denso de spikes. Reduzido `A_pre = 0.0001`, `A_post = -0.000105` (mantém razão R original).

**Trace step-by-step com nova config (mesma imagem, 100 ts):**

| t | post | w1 mean | w1 max | theta max |
|---|------|---------|--------|-----------|
| 0 | 1 | 0.137 | 0.298 | 0.001 |
| 10 | 390 | 0.141 | 0.305 | 0.227 |
| 30 | 249 | 0.145 | 0.311 | 0.750 |
| 50 | 209 | 0.138 | 0.310 | 1.108 |
| 70 | 119 | 0.129 | 0.303 | 1.306 |
| 99 | 91 | 0.121 | 0.301 | 1.505 |

**Comportamento gentle:** pesos oscilam levemente (0.12-0.14), theta cresce gradualmente (até 1.5), spikes ativos (~100-400/ts).

#### 1.4 — Re-rodando smoke do pretreino com nova config

| Config | n_imgs | w1 final | w2 final | Acurácia 5w1s |
|--------|--------|----------|----------|---------------|
| Sem checkpoint (random) | — | — | — | 20.92% (z=0.2) |
| A_pre=0.01 (paper, Etapa 0) | 500 | μ0.000 | μ0.609 | 20.00% (constante) |
| **A_pre=0.0001 (calibrado)** | 500 | **μ0.114** | **μ0.200** | **23.08% (z=0.4)** |
| A_pre=0.0001 (calibrado) | 5000 | **μ0.999/σ0.011 (saturado)** | μ0.725 | 20.52% (chance de novo) |

**Observação crítica:** a calibração estende a vida útil dos pesos (500 imgs OK), MAS em treino mais longo (5000 imgs) a saturação volta. step 100 já atingiu w1=0.806; step 200 → 0.959; step 600 → 0.999. Layer 1 converge pra estado homogêneo (todos pesos = 1) → filtros indistinguíveis → sem sinal discriminativo.

**Acurácia 5000-img (20.52%) ≈ chance** confirma que features colapsadas.

---

## Síntese da Sessão #7

### O que ficou validado

- ✅ Pipeline Omniglot integra end-to-end (ProtoNet 85.88%, Pixel kNN 45.76%, evaluate fecha)
- ✅ Bug de pickling em `data.py` corrigido (lambda → função module-level)
- ✅ Razão pré:pós empírica medida (R1=1.51, R2=0.69 — perto do regime do paper)
- ✅ A_pre 100× menor que paper resolve o **transitório** (500 imgs OK, pesos vivos)

### O que ficou exposto (bloqueio novo)

**LTP saturação em escala média/longa** mesmo com A_pre calibrado. Em 600+ steps de treino, w1 atinge 0.999 com std 0.011 — todos os pesos colapsam para 1.0. Filtros viram homogêneos, perdem sinal discriminativo, eval volta pra chance (20.52%).

A homeostasis (theta) não consegue acompanhar a velocidade da saturação em regime denso. theta_plus=0.0005 calibrado pra MNIST sanity é insuficiente pra Omniglot (que tem ~400× mais pós-spikes/timestep).

### Hipóteses pra próxima sessão

| Hipótese | Custo | Justificativa |
|----------|-------|---------------|
| H_norm: normalizar Σw por filtro após cada update (Σw=1) | ~30 min | Impede saturação por construção. Diehl & Cook usam isso implicitamente via inibição de condutância. |
| H_mult: STDP multiplicativo (Δw ∝ w·delta em vez de delta) | ~1h | Naturalmente decai perto de w_max. Suportado pela teoria SOFT-bounded STDP. |
| H_theta_omn: theta_plus calibrado pra Omniglot (talvez 5e-6 ou menor) | ~10 min | Atual deixa theta crescer rápido demais e satura; menor pode forçar feedback contínuo. |
| H_omniglot_inhib: inibição lateral via subtração de membrana (D&C original) | ~2h | Tira do k-WTA hard masking, vai pra soft inhibition. Mais fiel ao paper. |

### Estado final do código

- `config.py`: **A_pre=0.0001, A_post=-0.000105** (calibrado pra arquitetura conv, mantém pesos vivos em pretreino curto). theta_plus=0.0005 inalterado. Comentário no código explica histórico.
- `data.py`: bug pickling corrigido (`_invert_intensity` substitui lambda).
- `tests/test_spike_balance_omniglot.py`: novo, pode ser reutilizado pra recalibrar quando arquitetura mudar.
- Checkpoint atual em `checkpoints/stdp_model.pt`: 5000 imgs, **saturado** (w1≈1.0). Não é o melhor — checkpoint anterior de 500 imgs era melhor (não saturado), mas foi sobrescrito.

### Decisão pra próxima sessão

**Não rodar pretreino completo (24k imgs) sem resolver saturação** — vai dar acurácia próxima de chance, desperdiçando ~1h de GPU. Recomendado começar pela H_norm ou H_mult que atacam saturação por construção (custo curto, alto ROI). H_theta_omn é teste rápido que pode dar pista (custo ínfimo).

---

## Sessão #8 — H_theta_omn ❌ DESCARTADA

**Hipótese:** recalibrar `theta_plus` pra Omniglot pode quebrar saturação observada na sessão #7.

**Setup:** baseline 5000 imgs / 1 epoch / k=1 WTA, mesmo seed (42), variando apenas `theta_plus`.

**Resultados:**

| theta_plus | w1 final | w2 final | theta1 max | theta2 max | Acurácia 5w1s |
|------------|----------|----------|-----------|-----------|---------------|
| 0.0005 (sessão #7 baseline) | μ0.999/σ0.011 (saturado) | μ0.725 | (não medido na #7) | — | 20.52% (z=0.2) |
| **0.005** (10× maior) | μ0.167/σ0.110 (estável!) | μ0.158 (constante) | **9.0** | 6.2 | **20.00%** (IC zero, predição constante — theta silencia tudo) |
| **0.001** (2× maior) | μ0.892/σ0.150 (saturando) | μ0.323 | **33.0** | 17.6 | **20.32%** (z=0.2) |

**Padrão estrutural revelado:**

theta cresce **monotonicamente** em todos os casos — porque `tau_theta=1e7 ms` (do paper) não permite decay efetivo no nosso regime (5000 imgs × 100 ts = 500k chances de update vs decay desprezível por step). Resultado: theta_plus controla a velocidade de crescimento mas não a estabilização. Trade-off:

- **theta_plus alto** (0.005): theta atinge ~7 rapidamente → v_thresh_eff ≈ 8 → todos filtros silenciados → embedding constante → predição constante
- **theta_plus baixo** (0.0005): theta não freia o suficiente nas primeiras 100s de steps → pesos saturam em 0.999 → filtros indistinguíveis → embedding sem sinal
- **theta_plus médio** (0.001): trade-off ruim em ambos eixos (theta=33 + w1=0.89)

**Causa raiz: tau_theta inadequado pro nosso regime, não theta_plus.** Diehl & Cook calibram tau_theta=1e7 ms pra setup com refractory longo + 350ms/imagem (paper original tem ~7000 timesteps por imagem com poucos spikes ativos). Nosso regime tem 78400 spikes denso em 100 ms, então tau_theta efetivo deveria ser drasticamente menor (~1e3 ou 1e4 ms) pra theta ter ciclo de decay dentro do treino.

**Conclusão:** H_theta_omn descartada como solução isolada. Próxima sessão: H_norm (normalização de Σw, ataca saturação por construção) ou nova hipótese H_tau_theta (recalibrar tau_theta pro regime denso). Restaurado `theta_plus=0.0005` (estado conhecido, baseline pra próximas comparações).

---

## Estado pós-sessão #8

- `config.py`: theta_plus=0.0005 (restaurado), tau_theta=1e7 (problema isolado nesta sessão), A_pre=0.0001, A_post=-0.000105 (calibrados na sessão #7)
- Hipóteses descartadas (Semana 2): H_theta_omn (sozinha)
- Hipóteses vivas: **H_norm** (próxima recomendação por STRATEGY.md), H_mult, H_omniglot_inhib, H_tau_theta (nova, derivada do diagnóstico desta sessão)
- Acurácia consolidada: 17.76% (MNIST sanity) / ~20% Omniglot (chance, todos os tunings de Semana 2 falharam até agora)
- Sessões consecutivas sem progresso (>chance): **1** (esta). STRATEGY.md prevê revisão se chegar a 3.
