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

---

## Sessão #9 — H_tau_theta ✅ PRIMEIRO SINAL ACIMA DE CHANCE

**Hipótese:** tau_theta=1e7 (do paper) não permite decay efetivo no nosso regime — reduzir 100-1000× pode estabilizar homeostasis.

**Setup:** mesma config da sessão #8 (5000 imgs / 1 epoch / k=1 WTA / A_pre=0.0001 / theta_plus=0.0005 / seed treino=42), variando apenas `tau_theta_ms`.

### Iter 1: tau_theta_ms=1e4 (1000× menor que paper)

**Pretreino:** 5000 imgs em 126.4s (39.6 imgs/s).

| Métrica | Esperado (critério) | Observado | Status |
|---|---|---|---|
| `w1` saudável | μ ∈ [0.1, 0.5], σ > 0.05 | μ=0.999, σ=0.001 | ❌ saturado |
| `theta1` estabiliza | pico < 5 | max=20.68, mean=20.63, range [20.55, 20.68] | ❌ |
| Acurácia 5w1s (z>1) | > 25% | **35.98%** IC95% [35.17, 36.79], **z≈1.3** | ✅ **+15.98 p.p.** |

**Surpresa metodológica:** proxies estruturais (w1, theta) falharam mas critério funcional bateu com folga. Pesos saturados → filtros teoricamente indistinguíveis → embeddings deveriam ser constantes → predição deveria ser chance. Mas não é.

### Verificações de robustez (3/3 passaram)

| Verificação | Setup | Resultado | Critério | Status |
|---|---|---|---|---|
| **V1 — eval seed diferente** | mesmo ckpt, eval seed=100 | 36.06% IC95% [35.31, 36.83] | 30-40% | ✅ |
| **V2 — retrain seed diferente** | retrain seed=43, eval seed=42 | 35.96% IC95% [35.17, 36.71] | 30-40% | ✅ |
| **V3 — escalar dificuldade** | mesmo ckpt, 20w1s | 9.80% IC95% [9.58, 10.01], +4.80 p.p., z≈1.4 | >8-10% | ✅ |

Sinal não é ruído de seed (V1), não depende da combinação seeds-treino/seeds-eval (V2), e escala em dificuldade preservando ~5 p.p. acima de chance e z≈1 (V3).

### Mecanismo: NÃO é theta diferenciada (conjectura inicial refutada)

Conjectura inicial pós-Iter 1: "theta diferenciada por filtro carrega o sinal via timing" (range [20.55, 20.68] no seed=42 sugere variabilidade discriminativa).

**Refutado pelo V2:** retrain com seed=43 produziu theta1 com range AINDA MAIS APERTADO [20.78, 20.86] (Δ=0.08 vs Δ=0.13 no seed=42), e ainda assim acurácia idêntica (35.96% vs 35.98%). Se theta diff fosse o canal do sinal, seed=43 daria menos sinal.

**Mecanismo real: a investigar.** Hipóteses pra próxima sessão:
- Initialization bias preservado na saturação: pesos saturam em 0.999 com σ=0.001, mas a *direção* (qual neurônio satura primeiro, ordem temporal de saturação) pode codificar sinal mesmo no estado terminal.
- Dinâmica temporal LIF residual: spike timing dentro de cada filtro depende de input × peso saturado + ruído + reset, gerando spike trains discriminativos mesmo com pesos quase iguais.
- Combinação não-linear filter1 × filter2 via pooling: pequenas diferenças (σ=0.001) amplificadas pela não-linearidade do segundo layer.

### Estado final do código

- `config.py`: **tau_theta_ms=1e4** (alterado, fixado nesta sessão como decisão arquitetural — ver PLAN.md). Resto inalterado.
- Checkpoints salvos:
  - `checkpoints/stdp_model_iter1_seed42.pt` — Iter 1 original (35.98%)
  - `checkpoints/stdp_model_iter1_seed43.pt` — V2 reproduce (35.96%)
  - `checkpoints/stdp_model.pt` — sobrescrito por V2 (idêntico a seed43)

### Conclusão

H_tau_theta ✅ **CONFIRMADA empiricamente como solução parcial.** Primeiro sinal acima de chance do projeto inteiro (35.98% / 9.80% em 5w1s / 20w1s, z≈1.3-1.4 ambos). Sessões consecutivas sem progresso: **0** (resetado).

Importante: ainda longe das metas finais (≥90% 5w1s, ≥70% 20w1s do CONTEXT.md §4 — vs 85.88% do ProtoNet baseline). Sinal é **prova de viabilidade**, não estado de produção. Próximas sessões devem investigar mecanismo antes de tunar pra escala maior.

---

## Sessão #10 — Investigação de mecanismo via 3 ablações

**Pré-condição:** sessão #9 confirmou sinal robusto (35.98% 5w1s) mas mecanismo desconhecido. Antes de aceitar `tau_theta=1e4` ou amplificar, descartar 3 explicações triviais. **Sem tuning, sem escala** — só investigação.

**Critério (de STRATEGY.md "Pós-Sessão #9"):** se algum teste atingir ~36% sem o componente esperado, descoberta é artefato e reverte `tau_theta=1e4` pra `1e7`. Se nenhum atinge, sinal é real e mecanismo continua misterioso.

### A1 — Ablação de `_proj` (flatten 784D direto)

**Setup:** `tests/ablate_no_proj.py` monkey-patcha `extract_features` pra retornar `flat` (16×7×7 = 784D, taxa de spikes pós-pool2) direto, sem a projeção ortogonal não-quadrada de 784→64. Mesmo checkpoint Iter 1, mesmo 5w1s 1000 eps seed=42.

**Resultado:**

| Setup | Acurácia 5w1s | IC95% | z |
|---|---|---|---|
| Baseline Iter 1 (com `_proj`) | 35.98% | [35.17, 36.79] | 1.3 |
| **A1 — sem `_proj` (flat 784D)** | **36.12%** | [35.37, 36.90] | 1.3 |

**Interpretação:** sinal é **invariante à projeção**. Apesar de `_proj` reduzir 784→64 (descartando 720 direções), Hopfield+cosseno em 784D extrai praticamente a mesma informação que em 64D projetado. _proj **não é o canal** — sinal vive em estrutura mais grosseira do espaço pós-pool. A ablação não explica o sinal e portanto não invalida tau_theta=1e4 (no critério literal: 36.12% **não conta como passar de 36% sem o componente** — mas é exatamente 36%, então a leitura é "componente é irrelevante", não "componente carrega o sinal").

### A2 — Zerar conv weights (mantém theta treinada)

**Setup:** `tests/ablate_zero_conv.py` carrega checkpoint Iter 1, zera `layer1.conv.weight` e `layer2.conv.weight`, **preserva `theta1` e `theta2`** treinadas (μ=20.6 e 30.8). Salva como `stdp_model_zeroed.pt`. `evaluate.py` 5w1s 1000 eps seed=42.

**Predição rigorosa:** w=0 ⇒ I=conv(spk)=0 ⇒ membrana só recebe leakage (decay) ⇒ nunca atinge `v_thresh+theta ≈ 21` ⇒ zero spikes pós ⇒ embedding zero pra todas as imagens ⇒ predição constante ⇒ chance.

**Resultado:**

| Setup | Acurácia 5w1s | IC95% |
|---|---|---|
| **A2 — conv zerada + theta_iter1** | **20.00%** | [20.00, 20.00] (IC zero) |

**Interpretação:** chance exata, predição constante (IC zero). **Conv é necessária** — não há bypass não-óbvio (ex: theta sozinha modulando algo, ou Poisson chegando direto no Hopfield). Resultado é exatamente a predição rigorosa, **valida o pipeline conceitualmente** (o caminho de informação realmente passa por conv→spikes→embedding).

**Implicação:** theta sozinha (sem pesos não-zero) **não carrega sinal**. Logo, qualquer mecanismo que produz 35.98% precisa envolver os pesos da conv — mesmo saturados em 0.999 com σ=0.001.

### A3 — Pesos random U(0,1) sem treino algum

**Setup:** `tests/ablate_random_weights.py` cria `STDPHopfieldModel` novo, sobrescreve `layer1.conv.weight` e `layer2.conv.weight` com U(0,1) (range que cobre os 0.999 saturados do Iter 1). `theta=0` (modelo nunca foi treinado). Salva como `stdp_model_random_u01.pt`. `evaluate.py` 5w1s 1000 eps seed=42.

**Predição:** se ≈36%, STDP não está aprendendo nada útil — toda a estrutura vem da arquitetura + Hopfield + Poisson. Se cai pra chance, treino contribui.

**Resultado:**

| Setup | Acurácia 5w1s | IC95% | z |
|---|---|---|---|
| Baseline Iter 1 (treinado) | 35.98% | [35.17, 36.79] | 1.3 |
| **A3 — random U(0,1) sem treino** | **32.89%** | [32.19, 33.62] | 1.1 |
| Random U(0, 0.3) sem treino (sessão #7) | 20.92% | (chance) | 0.2 |
| Diferença Iter 1 − A3 | **3.09 p.p.** | IC95% não se sobrepõem | — |

**Interpretação:** **descoberta importante e desconfortável.** Pesos random U(0,1) sem treino algum entregam 32.89% — quase tudo do "sinal" que Iter 1 produz. STDP+homeostasis treinada agrega só **~3 p.p.** acima desse baseline arquitetural.

Comparando os 3 pontos:

- U(0, 0.3) + theta=0: 20.92% (chance) — magnitude baixa, não satura, embedding ralo.
- U(0, 1.0) + theta=0: 32.89% (z≈1.1) — magnitude alta cobrindo regime saturado.
- Iter 1 saturado em 0.999 σ=0.001 + theta=20.6: 35.98% (z≈1.3).

O salto principal é de magnitude (0.3→1.0): **+12 p.p. ganhos só por escalar o range dos pesos**, sem treino. STDP em cima desse regime ganha **+3 p.p.** adicionais.

**Mecanismo provável (revisado):** o sinal não vem de "STDP aprendendo features de Omniglot" — vem de **conv com pesos saturados em magnitude alta produzir embeddings cuja taxa de spikes é discriminativa pela natureza esparsa do input** (caracteres Omniglot, traços brancos sobre fundo preto). Random pesos altos já fazem 90% do trabalho. STDP só ajusta a margem.

### Critério de decisão (de STRATEGY.md "Pós-Sessão #9")

> "Se algum teste de ablação atinge ~36% sem o componente ablacionado: descoberta é artefato. Reverte decisão `tau_theta=1e4`."

**A3 atinge 32.89% sem treino algum.** Não é exatamente 36%, mas é tão próximo (3 p.p., IC95% [32.19, 33.62]) que a interpretação literal do critério é **ambígua**:

- **Leitura estrita:** "atinge ~36%" → 32.89% não é 36% → critério não dispara → não reverte.
- **Leitura do espírito:** "STDP não é o que carrega sinal" → A3 mostra que STDP carrega só 3 p.p. de 16 p.p. acima de chance → ~80% do sinal é arquitetural → critério dispara em espírito → reverte.

**Esta decisão é do Luis, não minha.** O que está empiricamente claro:

1. ✅ A1 não invalida (sinal sobrevive sem _proj).
2. ✅ A2 confirma pipeline (conv é necessária, theta sozinha não basta).
3. ⚠️ A3 mostra que STDP+homeostasis treinada **agrega ~3 p.p.** sobre baseline arquitetural com pesos random U(0,1). Maior parte do sinal de 35.98% vem de "pesos com magnitude alta" + Poisson + Hopfield, não de STDP.

### Estado final pós-#10 (pendente decisão)

- Código: inalterado (`tau_theta_ms=1e4` ainda em config.py).
- Checkpoints novos:
  - `stdp_model_zeroed.pt` (A2: conv=0, theta_iter1)
  - `stdp_model_random_u01.pt` (A3: random U(0,1), theta=0)
- Scripts: `tests/ablate_no_proj.py`, `tests/ablate_zero_conv.py`, `tests/ablate_random_weights.py`.
- Sessões consecutivas sem sinal>chance: ainda 0 (sinal real, mas com mecanismo agora parcialmente entendido como **arquitetural+magnitude**, não **STDP**).

**Hipóteses pra próxima sessão (caso decisão seja reverter):**
- Ablação adicional A3b: random U(0,1) + theta_iter1 (preservada). Se for ≈36%, STDP+homeostasis **realmente** não contribui. Se for 33%, theta agrega algo sobre magnitude.
- Investigar por que magnitude alta sozinha gera 33%: estrutura da conv saturada vs estrutura random — o que é discriminativo aqui?
- Se decisão é reverter `tau_theta=1e4`: voltar pra `tau_theta=1e7` e atacar H_norm (próxima da ordem de STRATEGY.md).

### A3b — Random U(0,1) + theta_iter1 preservada

**Setup:** carrega checkpoint Iter 1 pra extrair theta1 e theta2 treinadas, sobrescreve apenas os conv weights com U(0,1) random, preserva theta. `tests/ablate_random_with_theta.py`. 5w1s 1000 eps seed=42.

**Resultado:**

| Setup | Acurácia 5w1s | IC95% | z |
|---|---|---|---|
| A3 — random U(0,1) + theta=0 | 32.89% | [32.19, 33.62] | 1.1 |
| **A3b — random U(0,1) + theta_iter1 (~20.6, ~30.8)** | **32.65%** | [31.95, 33.36] | 1.1 |
| Δ (theta_iter1 vs theta=0) | **−0.24 p.p.** | dentro do IC | — |

**Interpretação:** theta treinada (~20.6, ~30.8) **não agrega nada** sobre magnitude alta dos pesos random. A diferença entre A3 e A3b é dentro do ruído.

### Decomposição final dos 16 p.p. acima de chance (Iter 1 = 35.98%)

| Componente | Contribuição | Evidência |
|---|---|---|
| Magnitude alta dos pesos (U(0, 0.3) → U(0, 1.0)) | **+12 p.p.** | A3 (32.89%) − sessão #7 baseline U(0,0.3) (20.92%) |
| Estrutura específica dos pesos saturados pelo STDP | **+3 p.p.** | Iter 1 (35.98%) − A3b (32.65%) — isolado controlando theta |
| Theta treinada (homeostasis adaptativa) | **~0 p.p.** | A3 vs A3b: −0.24 p.p. dentro do IC |
| Total acima de chance | 16 p.p. | |

**Implicação mecanística:** o sinal de 35.98% é **arquitetural+magnitude** dominantemente. STDP contribui na margem (3 p.p.) via estrutura espacial sutil que sobrevive à saturação σ=0.001. Homeostasis (theta) é **inerte** no regime testado.

### Decisão pós-#10

**Reverter `tau_theta_ms=1e7`** (paper original). Leitura do espírito do critério de STRATEGY: STDP+homeostasis explica só 3 p.p. de 16 p.p. — ~80% do sinal é arquitetural. Decisão arquitetural da sessão #9 (`tau_theta=1e4`) foi prematura: o que destravou sinal **não foi** tau_theta calibrado, foi **magnitude saturada** dos pesos (que aparece em qualquer config que satura) **+ Hopfield + Poisson**.

`tau_theta=1e4` consegue saturar pesos (porque homeostasis efetiva freia LTD ineficiente o suficiente). `tau_theta=1e7` também saturaria com config diferente. O sinal não vem da homeostasis funcionar — vem da magnitude alta + estrutura de Omniglot esparsa.

### Estado final pós-#10

- `config.py`: revertido `tau_theta_ms=1e7` (estado conhecido pré-#9).
- `PLAN.md`: decisão arquitetural sessão #9 substituída por nota de reversão + lições aprendidas.
- `STRATEGY.md`: nova framing pra "Pós-Sessão #10" — pergunta é "como amplificar sinal arquitetural via STDP?", não "como STDP aprende features?".
- Sessões consecutivas sem sinal>chance: **0** (sinal arquitetural existe, é real). O que mudou é entendimento — não há regressão de sinal.

---

## Sessão #11 — H_visualize: o que os pesos saturados estão capturando?

**Pré-condição:** STRATEGY.md "Pós-Sessão #10" recomendou H_visualize antes de H_norm/H_mult. Se filtros saturados estão capturando algo Gabor-like apesar de σ=0.001, otimização vale. Se são ruído puro, abordagem precisa mudar mais radicalmente.

**Setup:** análise sobre os 2 checkpoints existentes (sem treino novo): `stdp_model_iter1_seed42.pt` (35.98%) e `stdp_model_random_u01.pt` (32.89%). Script `tests/visualize_filters_session_11.py`. Saída em `figs/sessao_11/` (10 PNGs, 222 KB total).

### Métricas quantitativas

| Métrica | Iter 1 (saturado) | Random U(0,1) |
|---|---|---|
| Layer 1 — μ global | 0.9985 | 0.5018 |
| Layer 1 — σ global | 0.0011 | 0.2971 |
| Layer 1 — σ espacial por filtro (média) | 0.00080 | 0.29266 |
| Layer 1 — cosine raw off-diag (mean) | **1.0000** (todos quase constantes) | 0.7277 |
| Layer 1 — **cosine *centered* off-diag** (mean) | **0.2047** (estrutura compartilhada!) | −0.0561 (independentes) |
| Layer 2 — cosine raw off-diag (mean) | 1.0000 | 0.7473 |
| Layer 2 — **cosine *centered* off-diag** (mean) | **0.5461** (alta correlação espacial) | −0.0057 |

A coluna **centered cosine** é o achado central: subtrai a média de cada filtro antes de calcular cosseno, isolando estrutura espacial fina (escala da σ=0.001) do offset constante. Iter 1 tem **estrutura espacial sistemática e compartilhada** entre filtros; random tem padrões independentes.

### Interpretação visual (`figs/sessao_11/`)

**`layer1_filters_delta.png` (Iter 1 menos média do random):**
Os 8 filtros do Iter 1 mostram **praticamente o mesmo padrão espacial** — vermelho/positivo nas regiões superiores e centro-esquerda, azul/negativo concentrado no canto inferior direito. Padrão visualmente idêntico em todos os 8 filtros. Confirma o cosine centered alto: STDP convergiu todos os filtros pra uma única "direção espacial sistemática".

**`layer1_filters_iter1_centered.png`:**
Cada filtro zero-mean mostra padrões parecidos (com variação local). σ ~0.001 mas estruturado. Não é ruído gaussiano — é informação sub-pixel.

**`layer1_filters_random_centered.png`:**
Padrões espaciais visivelmente diferentes entre os 8 filtros — cada um capta uma direção arbitrária do espaço de pesos.

**`layer2_filters_iter1.png` (16×8 grid):**
Saturação visível (predominância de amarelo = alto), mas **canais de entrada `in1` e `in2`** mostram estrutura claramente diferente — zonas mais escuras concentradas no canto superior esquerdo, replicadas nos 16 filtros de saída. Outros canais (`in0`, `in3`-`in7`) saturam mais homogeneamente. Hipótese mecânica: filtros 1 e 2 da Layer 1 (out1, out2) eram os que mais disparavam, então as conexões da Layer 2 que recebiam input deles receberam mais updates LTD → menos saturação local, mais estrutura.

**`cosine_matrix_layer1.png` e `_layer2.png`:**
Iter 1 cosine raw é uniformemente vermelho (~1.0 em tudo) porque filtros são quase constantes. Random mostra diagonal forte + off-diagonal moderada (~0.7-0.8 por causa de média positiva U(0,1)).

### O que o STDP está capturando? Hipótese consolidada

**Não é Gabor / orientação / frequência espacial.** É **matched filter pra estatística do Omniglot:**

- Caracteres do Omniglot têm distribuição não-uniforme de "tinta" no bounding box (após inversão pra fundo preto + traços brancos): centro e superior tendem a ter mais ativação que canto inferior direito.
- STDP no regime saturado converge **todos os filtros** pra esse padrão estatístico médio (não pra padrões diversos).
- Resultado: efetivamente **1 dimensão útil** (replicada 8×16 vezes), capturando "quão bem a imagem casa com o protótipo médio do Omniglot".
- Random U(0,1) tem **8 dimensões independentes** com padrões arbitrários — cada uma é uma projeção aleatória, mais ruidosa individualmente mas mais diverso.

**Por que o matched filter (Iter 1) bate random em 3 p.p.?** Possivelmente porque a 1 dimensão útil tem signal-to-noise melhor pra discriminar Omniglot grosseiramente. Random tem mais dimensões mas cada uma é mais ruidosa — o ganho de diversidade não compensa a perda de "matched-ness".

### Implicação pra hipóteses H_norm / H_mult

**Trade-off não-óbvio revelado:**

- H_norm/H_mult vão **impedir saturação** → preservar variância **dentro de cada filtro** → MAS perder o "matched filter" trivial (porque sem saturação, STDP não convergiria todos pra mesmo lugar — ou convergiria, mas com magnitude menor).
- O que se quer: **filtros distintos (cosine centered baixo) E informativos individualmente** (não matched filter trivial, não ruído puro).
- Diehl & Cook resolve isso com **homeostasis distribuindo spike rate entre filtros** — testamos, não funciona no nosso regime denso.

**Conclusão honesta:** o gargalo arquitetural é mais profundo que tau_theta ou normalização. STDP no regime saturado **descobre 1 protótipo médio**; STDP sem saturação **não converge** (sessões #2-#5 documentaram). Pra ter diversidade real, precisa **mecanismo que force filtros a competir por nichos diferentes** — não k-WTA por posição (que já temos), não homeostasis Diehl & Cook (testado), mas algo como inibição lateral entre filtros independente de posição, ou competição via normalização de Σ por filtro.

### Estado final pós-#11

- Nenhuma mudança em código de produção (`config.py`, `model.py` inalterados).
- Adicionado: `experiment_01_oneshot/tests/visualize_filters_session_11.py`, `figs/sessao_11/` (10 PNGs).
- Achado central: filtros saturados não são ruído. Capturam estatística média do dataset, mas não oferecem diversidade de features.
- Hipóteses futuras refinadas:
  - **H_norm**: ainda viável, mas com expectativa baixa — pode preservar variância sem destravar diversidade. Custo: ~30 min.
  - **H_mult**: idem, soft-bound natural.
  - **H_filter_diversity** (nova, derivada do achado): mecanismo explícito de competição entre filtros independente de posição (ex: penalidade no loss STDP por similaridade entre filtros, ou normalização Σw por filtro como em H_norm mas combinada com ortogonalização periódica). Custo: ~1-2h, mais ambicioso.
- Sessões consecutivas sem sinal>chance: **0** (não rodou treino novo, sinal arquitetural ainda em 35.98%).
