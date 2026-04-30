# WEEKLY-1.md — Experimento 02 Continual Learning

> Sessões #22-#23 (2026-05-01).
> Para framing estratégico, ver `STRATEGY.md` "Confirmação Pós-Sessão #21".
> Para plano operacional, ver `experiment_02_continual/PLAN.md`.

---

## Sessão #22 (admin)

Confirmação das 5 decisões pós-#21 + literatura review (`PAPERS.md`).
Sem código. Sem experimento. Detalhes em `STRATEGY.md` e `PAPERS.md`.

---

## Sessão #23 — Baseline naive sequential fine-tuning

**Objetivo:** estabelecer floor de performance pra família de propostas de continual learning sem replay (#26-#28). Sem nenhuma defesa contra catastrophic forgetting.

### Setup implementado em `baseline_naive.py`

- **Tasks:** 50 × 5 classes (250 classes do background set, sampled deterministicamente por seed).
- **Per classe:** 14 train + 6 test instances (Omniglot tem 20 por classe; split deterministic via shuffle seedado).
- **Episode protocol:** 5-way 1-shot 5-query (matches project convention 5w1s).
- **Phases:**
  1. **Warmup:** 500 episodes joint sobre tasks 1-5 (treinamento ProtoNet padrão sobre 25 classes).
  2. **Sequential fine-tune:** 100 episodes por task × tasks 6-50 (cada task um pacote).
  3. **Eval final:** 50 episodes por task × 50 tasks (encoder pós-tudo, prototypes fresh do test pool).
- **Métricas:**
  - **ACC** = média de accuracy final em todas 50 tasks
  - **BWT** = média de (acc_final[t] − acc_just_after[t]) pra t=1..50 (negativo = forgetting)
- **5 seeds** (42-46), IC95% bootstrap, total ~9.8 min em RTX 4070.

### Resultados

| Seed | ACC final | BWT |
|---|---|---|
| 42 | 82.08% | −12.83 p.p. |
| 43 | 84.40% | −10.74 p.p. |
| 44 | 86.09% | −9.08 p.p. |
| 45 | 78.79% | −16.42 p.p. |
| 46 | 81.56% | −13.21 p.p. |
| **Média** | **82.58%** | **−12.46 p.p.** |
| **IC95% bootstrap** | [80.47, 84.63] | [−14.64, −10.50] |

### Comparação com critério da sessão

| Métrica | Esperado | Observado | Diferença |
|---|---|---|---|
| ACC | 30-50% | **82.58%** | +32.58 a +52.58 p.p. ACIMA |
| BWT | −30 a −50 p.p. | **−12.46 p.p.** | ~17-37 p.p. MENOS forgetting |

**Resultado consistentemente acima do critério da sessão.** Antes de declarar bug, análise mecanística:

### Por que naive ProtoNet é naturalmente robusto a forgetting?

1. **Não há classifier head treinado** — em modelos típicos de continual learning (e.g., Resnet+softmax classifier), fine-tuning em novas tasks reescreve os pesos do classifier, destruindo o mapeamento aprendido pras tasks antigas. ProtoNet **não tem isso**: a "decisão" é prototype-based, computada FRESH a cada episódio do support set.

2. **Encoder aprende métrica genérica de Omniglot.** O encoder ProtoNet é treinado pra mapear caracteres similares próximos no espaço de embedding. Isso é uma propriedade GLOBAL do dataset, não específica de uma task. Treinar em tasks específicas pode refinar a métrica mas não destruir totalmente.

3. **Tasks são "novas classes do mesmo dataset"**. As 50 tasks compartilham o mesmo *domain* (caracteres Omniglot). Features aprendidas em qualquer task transferem pra outras. Isso é diferente de continual learning cross-domain (e.g., MNIST → SVHN → CIFAR), onde forgetting é dramático.

### Verificações de que o sequencial é real

- **Just_after acc varia entre tasks** (91.4%, 86.2%, 96.2%, ...): se encoder estivesse congelado seria constante. Variação confirma updates ocorrendo.
- **BWT é negativo** (−12.46 p.p.): há forgetting real, só que moderado.
- **Tempo de execução** (9.8 min pra 5 seeds): consistente com 5000 train episodes × 5 seeds = 25K backward passes.

**Não é bug. É comportamento estrutural do ProtoNet em continual learning.**

### Implicação pra hipótese central do Marco 1

A hipótese pós-#22 (ver `STRATEGY.md` "Confirmação Pós-Sessão #21" item b):

> "Pode plasticidade meta-aprendida (estilo C2) ou ProtoNet+k-WTA (estilo C3), sem replay, atingir ACC ≥75% E BWT ≥−10% em Split-Omniglot 50-tasks, batendo EWC por ≥3 p.p.?"

**Naive ProtoNet sequential já atinge ACC 82.58% e BWT −12.46.** Isso significa:

- **Critério "ACC ≥75%"** é trivialmente atingido pelo BASELINE NAIVE.
- **Critério "BWT ≥−10%"** está MUITO próximo (apenas 2.46 p.p. de gap).
- **Margem pra bater EWC por ≥3 p.p.** vai ser pequena se EWC só sobe pra ~88% (o que é plausível em um setup já fácil).

### Re-framing necessário

A sessão #23 revela que **o setup como definido na #22 é insuficientemente adversarial pra pergunta científica original**. Forgetting moderado em ProtoNet sequencial é o estado natural, não algo dramático que "novas técnicas" resolvem.

3 caminhos pra próximas sessões (decisão fica pra Luis, não me comprometo):

**Opção 1 — Manter setup, ajustar pergunta:**
- Reformular critério: bater EWC por ≥1.5 p.p. (margem realista dado floor alto).
- Foco em BWT (atualmente -12.46, alvo -5 a -8): mostrar que C2/C3 podem reduzir forgetting moderado de naive pra near-zero.
- Resultado defensável academicamente: "plasticidade local meta-aprendida reduz forgetting ProtoNet de -12 p.p. pra -X p.p."

**Opção 2 — Adversarializar setup pra forçar forgetting:**
- **Sem warmup:** encoder fresh aprendendo on-the-fly task por task. Sem pretreino genérico.
- **Mais episodes per task** (200-500): mais drift sequencial.
- **Tasks por alfabeto:** agrupar 5 caracteres do MESMO alfabeto por task (mais similares, mais interferência).
- **K-shot maior:** k=5 ao invés de k=1 → encoder vê mais variação por episódio, pode overfit a task atual.
- Esperado: ACC cai pra 30-60%, BWT vira −20 a −40 p.p. — match expectativas originais da #23.

**Opção 3 — Mudar dataset cross-domain:**
- Substituir Split-Omniglot por benchmark cross-dataset (e.g., Omniglot → MNIST → FashionMNIST).
- Forgetting cross-domain é dramático (literatura confirma).
- Custo: setup novo, mais infra; possivelmente fora do scope side project.

**Recomendação minha (provisional, decisão é do Luis):** **Opção 2** — adversarializar setup. Mantém Split-Omniglot (infraestrutura pronta, 5h/semana viável), mas força forgetting que dá margem real pra avaliar propostas C2/C3-continual.

Mudanças concretas pra Opção 2 (a discutir):
- `--n-warmup-tasks 0 --warmup-episodes 0` (skip warmup)
- `--finetune-episodes 300` (3× mais drift)
- (futuro: agrupar tasks por alfabeto em vez de classes random)

### Estado final pós-#23

- **Código novo:** `experiment_02_continual/baseline_naive.py` (5 seeds, ~9.8 min em RTX 4070).
- `model.py`, `config.py`, `baselines.py` (experiment_01) intocados.
- **Floor ESTABELECIDO** mas mais alto que esperado: ACC 82.58% / BWT −12.46.
- **Re-framing necessário** pro Marco 1 antes de implementar EWC (#24): decidir Opção 1, 2, ou 3 acima.
- Sessões consecutivas sem sinal>chance: **0** (mantido — esta sessão produziu resultado robusto, mesmo que diferente do esperado).

### Próxima sessão

**Antes de #24 (EWC baseline), decidir entre Opção 1/2/3 acima.** Preferencialmente em sessão administrativa curta (30 min) — não emendar com implementação sob critério ainda incerto.

Se Luis escolher Opção 2 e quiser confirmar empiricamente que adversarializar funciona, pode rodar variação de `baseline_naive.py` com `--warmup-episodes 0 --finetune-episodes 300` em ~15-20 min antes de #24. Isto valida que setup adversarial produz o forgetting esperado, antes de investir em EWC.

---

## Sessão #25 — Implementação Opção D (alphabets + skip warmup) — INSUFICIENTE

**Pré-condição:** sessão #24 escolheu Opção D em STRATEGY.md "Reformulação Pós-Sessão #23". Esperado naive cair pra 40-55% ACC.

### Implementação

`baseline_naive.py` refatorado:

- Novo `CombinedOmniglot` wrapper: concatena background (964 chars / 30 alfabetos) + evaluation (659 chars / 20 alfabetos) = 1623 chars / **50 alfabetos totais**. Renumera labels da evaluation com offset.
- Nova `build_tasks_by_alphabet`: parse de `dataset._characters[i]` pra extrair alfabeto via path split. Subsample n_chars_per_task=14 chars/alfabeto (todos os 50 têm ≥14, nenhum droppado). Ordem das tasks aleatorizada por seed.
- Flag `--task-mode {alphabet,random}` (default alphabet). Modo random preserva setup #23 pra reprodução.
- Defaults pós-#24: `--warmup-episodes 0`, `--n-warmup-tasks 0` (skip warmup).
- Guard em phase warmup: skipped se `n_warmup_tasks=0` ou `warmup_episodes=0`.

Refatoração dentro do orçamento (~25 min). Sanity 1 seed (finetune=30, eval=15) confirmou loop fechado: ACC 78.66%, BWT -5.88.

### Resultado 5 seeds (defaults: finetune=100, eval=50)

| Métrica | Setup #23 (random splits + warmup) | **Setup #25 (alphabets + skip warmup)** | Δ |
|---|---|---|---|
| ACC final | 82.58% IC [80.47, 84.63] | **80.65%** IC [79.23, 82.06] | **-1.93 p.p.** |
| BWT | -12.46 IC [-14.64, -10.50] | **-9.26** IC [-10.72, -7.88] | **+3.20 p.p.** |
| Tempo | 9.8 min | 11.3 min | — |

ACC final por seed (alphabet mode): 79.16%, 80.48%, 83.06%, 81.86%, 78.68%.
BWT por seed: -10.94, -9.71, -7.10, -7.53, -11.00.

### Critério da sessão (definido na #25)

| Range | Decisão | Resultado |
|---|---|---|
| ACC 40-55% | **VALIDADO** → próximo é EWC | NÃO |
| ACC <35% | Adversarial demais | NÃO |
| **ACC >60%** | **Insuficiente, re-considerar** | **SIM (80.65%)** |

**Opção D FALHA pelo critério.** Naive cai apenas ~2pp em relação ao setup #23. BWT até MELHORA (menos forgetting), o oposto do esperado.

### Por que Opção D não funcionou?

Hipótese: alfabetos não são suficientemente "ortogonais" pra ProtoNet. Encoder ProtoNet aprende uma métrica genérica de Omniglot — "como mapear caracteres similares perto" — que **transfere bem entre alfabetos também**. Não é só random splits que compartilham features genéricas; alfabetos compartilham as mesmas features básicas (linhas, curvas, simetrias).

Adicionalmente, **subsample 14 chars/task em alphabet mode (vs 5 em random mode) aumenta diversidade DENTRO da task** — encoder tem mais variação pra aprender, fica menos especializado, menos sujeito a esquecer.

**O fundamental ProtoNet sem classifier head é robusto a forgetting em Omniglot. Não há setup trivial dentro de Omniglot que faça naive cair pra 40-55%.**

### Caminhos pra próximas sessões (decisão é do Luis, não me comprometo)

3 opções pra fazer naive realmente cair:

**Opção D' (intensificar fine-tune):** rodar com `--finetune-episodes 500` ou 1000. Encoder overfit massivamente a task atual → drift muito maior. Custo: ~30-45 min de experimento. Risco: se ainda não cair, esgota essa rota.

**Opção C revisitada (cross-domain):** Omniglot → MNIST → FashionMNIST → KMNIST → CIFAR-10 (em escala 28×28 grayscale). Forgetting cross-domain é dramático. Custo: implementação ~2-3 sessões. Viola decisão (b) Pós-#21 ("Split-Omniglot 50-tasks") mas é defensável dado #25 mostrar que Split-Omniglot é insuficiente.

**Opção E (mudar paradigma de eval):** trocar ProtoNet eval por **classifier head sequential learning**. Encoder + camada linear N-way (N = número total de classes). Treino sequencial atualiza head → forgetting clássico aparece. Custo: ~1 sessão de re-arquitetura. Resolve a causa raiz (prototypes fresh são o que protege ProtoNet).

**Recomendação minha (provisional):** **Opção E** — mudar paradigma de eval. Razões:
- Setup mais alinhado com literatura mainstream de continual learning (EWC/SI/GEM são todos sobre classifier-based methods)
- Resolve a causa raiz identificada (sem classifier head treinado, ProtoNet é robusto)
- Custo moderado (~1 sessão)
- Não viola a decisão (b) Pós-#21 — ainda usamos Split-Omniglot, só mudamos como avaliamos

**Risco da Opção E:** muda significativamente o que estamos medindo. Pergunta científica precisa re-formular novamente: "plasticidade meta-aprendida em encoder + classifier head bate EWC por 3pp?". Mais sessões administrativas.

**Decisão pendente do Luis** antes de #26. Idealmente sessão administrativa curta de 30 min.

### Estado final pós-#25

- `baseline_naive.py` agora suporta dois modos (alphabet | random) — código robusto pra ambos.
- Setup Opção D mensurado e documentado como insuficiente.
- Pergunta científica oficial **broken pela segunda vez** (sessão #23 brokei a primeira; setup adversarial proposto na #24 não consertou).
- Sessões consecutivas sem sinal>chance: 0 (resultado é informativo, não null).
- **Marco 1 está em risco** se naive não cair pra range adversarial. Próxima sessão precisa decidir caminho ou aceitar que Marco 1 vai ter margem científica pequena.

---

## Sessão #26 (admin) — revisão estratégica pós-#25

Sem código. Documentou 4 caminhos (1 setup-E, 2 reformular pergunta, 3 pivot, 4 encerrar Marco 1) em STRATEGY.md "Revisão Pós-Sessão #25". Decisão pendente do Luis.

---

## Sessão #27 — Caminho 5d (3 arquiteturas STDP+C2+continual): scaffolds + B sanity

**Pré-condição:** Luis escolheu testar empiricamente uma 5ª opção (não previa nas 4 da #26): combinar STDP biofísico + plasticidade local meta-aprendida + continual learning sequencial em 3 arquiteturas. STRATEGY.md ganhou "Decisão Pós-Sessão #26: Caminho 5d (3 arquiteturas)".

**Aviso explícito registrado:** STDP biofísico tem barreira estrutural conhecida (#1-#13, ~36% Omniglot one-shot via matched filter trivial). Termo Hebbian puro foi mostrado dispensável em sessão #18 (C2 contribuição vem de B,C,D). Esta sessão testa se em CONTEXTO CONTINUAL (com pressão temporal) essas conclusões mudam.

### Scaffolds criados

`experiment_02_continual/c2_continual_arch_a.py`, `_b.py`, `_c.py` — 3 arquiteturas com docstring detalhada (hipótese específica, métricas-alvo, ablações pré-definidas). A e C ficam com `# TODO: implementar` até B passar sanity.

### Possibilidade B — regra híbrida unificada (IMPLEMENTADA)

Regra de plasticidade por peso:
```
Δw_ij = η · (A_ij · pre_j · post_i · trace_j + B_ij · pre_j + C_ij · post_i + D_ij)
```

`trace_j[t] = decay · trace_j[t-1] + pre_j[t]` (STDP-like exponential trace ao longo do inner loop).

Meta-params: A, B, C, D por peso em 2 layers (`784→128→32`) + decay scalar global. Total ~417K params (mesma escala de C2). Pesos iniciais zero, encoder linear (sem tanh), prototype classifier (cosine, β=8).

Hipótese: **trace temporal pode tornar termo Hebbian A·pre·post·trace não-trivial em CL** (em #18 sem trace, A foi dispensável — talvez timing matters em sequência de tasks).

### Sanity (1 seed, n_inner=5, finetune-eps=30, eval-eps=15)

| Métrica | Valor | Critério da sessão |
|---|---|---|
| ACC final | **47.89%** | passa (>40%, <95%) |
| BWT | -2.05 p.p. | informativo (menos forgetting que naive) |
| decay aprendido | 0.500 → 0.443 | gradiente flui ✓ |
| just-after acc range | 36-68% | adaptação ocorre, alta variância |
| Tempo | 52.6s | 1 seed reduzido |

**Loop funciona empiricamente:**
- Plasticidade atualiza pesos (que começam em zero) → encoder discriminativo emerge
- Gradiente flui pelos meta-params (decay treinou)
- Inner loop produz embeddings que classificam queries acima de chance

### Análise honesta

**ACC=47.89% está -32.76 p.p. abaixo do baseline naive ProtoNet (80.65%).** Comparação: ProtoNet usa CNN-4 (~110K params em conv layers, treinada via SGD), B usa encoder linear (sem capacidade convolucional, sem não-linearidades). Esperado que B fique abaixo em ACC bruto. Mas **a hipótese central do Caminho 5d era que B agregaria sobre baseline via trace temporal — isso seria ACC > 80.65%, não < 50%.**

Com defaults (n_inner=10, finetune=100, eval=50), ACC provavelmente sobe pra 55-70% em 5 seeds completos. Mesmo com upside, é **improvável bater baseline 80.65%**.

### O que isso significa pra Caminho 5d

A hipótese de B estava em duas partes:
1. Loop converge em CL: ✓ confirmado (sanity 47.89%)
2. Bate baseline naive: ✗ improvável (gap de 33 p.p. com config sanity reduzida)

**Implicação pra A e C:** Possibilidades A (STDP+C2 stacked) e C (two-timescale) provavelmente herdarão a mesma limitação fundamental — encoder mais simples que CNN-4. STDP biofísico em A já mostrou barreira em #1-#13 (~36%). Adicionar STDP a um encoder linear não vai magicamente bater CNN-4.

**Possibilidade que muda o quadro:** se B com defaults completos (5 seeds) atingir 70-80% E ablação A=0 mostrar que A·pre·post·trace contribui ≥3 p.p., aí Caminho 5d tem viabilidade. Caso contrário, é improvável que A e C salvem.

### Decisão pendente do Luis (não tomada nesta sessão)

3 opções a considerar antes de #28:

**Opção α:** Rodar B completo (5 seeds, defaults). Custo: ~10-15 min execução. Confirma se ACC sobe substancialmente com config completa. Se sim (≥70%), justifica investir em A e C. Se não (≤55%), Caminho 5d fica difícil de defender.

**Opção β:** Aceitar que encoder linear é teto e adicionar capacidade (CNN encoder + plasticidade meta-aprendida). Implica re-design não-trivial. Custo: ~2 sessões.

**Opção γ:** Voltar pros 4 caminhos da #26. Sanity B mostrou que Caminho 5d enfrenta mesma limitação fundamental que C2 isolado em one-shot — gap de capacity vs CNN-4 baseline.

### Estado final pós-#27

- **Scaffolds criados:** `c2_continual_arch_a.py`, `_b.py` (implementado), `_c.py`.
- **B sanity:** 47.89% ACC, loop funciona.
- `model.py`, `config.py`, `baseline_naive.py`, `c2_simplified.py`, `c2_meta_hebbian.py` intocados (conforme restrição).
- Sessões consecutivas sem sinal>chance: 0 (47.89% > chance 20%).
- **Sinal: empírico mas insuficiente.** B sanity confirma viabilidade técnica do paradigma híbrido, mas magnitude do gap até baseline naive sugere arquitetura precisa repensada.

### Próxima sessão idealmente

Sessão admin curta (15-30 min) decidindo entre Opção α (rodar B 5 seeds completos), Opção β (CNN encoder), ou Opção γ (voltar pros 4 caminhos). Sem decisão clara, opção α é a mais barata pra produzir mais dado antes de decidir.

---

## Sessão #28 — Caminho 5e (kitchen sink): scaffold + sanity 5 tasks → 50 tasks

**Pré-condição:** sessão #27 mostrou que Possibilidade B (encoder linear) tem gap de capacity vs CNN-4. Luis escolheu Caminho 5e: combinar **CNN-4 (de C3) + plasticidade meta-aprendida (de C2) + trace STDP-like (de B) + k-WTA esparso (de C3b) + continual sequencial**. STRATEGY.md ganhou "Decisão Pós-Sessão #27: Caminho 5e (arquitetura combinada)" registrando aceitação explícita de risco (complexidade, custo computacional, originalidade incremental).

### Arquitetura específica (Opção 2: hybrid backprop + plasticidade)

```
Image → CNN-4 (SGD via backprop, persistem cross-task) → 64D
     → Linear plasticity W (64×64, inner-loop adapted, reset zero per episode)
        Δw = η·(A·pre·post·trace + B·pre + C·post + D)
     → k-WTA (k=16, 75% sparsity)
     → Prototype classifier (cosine, β=8)

Trainable params:
- CNN-4 weights (slow, 111K params)
- A, B, C, D, decay (slow meta-params, 16K + 1 = 16385 params)
```

Total ~128K trainable. CNN persiste cross-task (sujeito a forgetting), W reseta a cada episode (inner-loop adaptation).

### Sanity 5 tasks (1 seed, n_inner=5, finetune=30, eval=15)

| Métrica | Valor | Critério |
|---|---|---|
| ACC final | **71.20%** | passa (>30%, <95%) ✓ |
| BWT | -6.24 p.p. | informativo |
| decay aprendido | 0.500 → 0.523 | gradiente flui ✓ |
| Tempo | 5.7s | trivial |

**Loop fecha sem bug em escala reduzida.** Prosseguiu pra sanity 50 tasks.

### Sanity 50 tasks (1 seed, n_inner=5, finetune=30, eval=15)

| Métrica | Valor | Critério |
|---|---|---|
| ACC final | **75.34%** | passa (~50-85% range esperado) ✓ |
| BWT | **-5.78 p.p.** | melhor que naive (-9.26) |
| decay aprendido | 0.500 → 0.639 | cresceu (trace mais longo) |
| just-after acc | varia 78-92% | adaptação ocorre |
| Tempo | 54.4s | viável pra escala completa |

### Comparação consolidada

| Modelo | ACC | BWT | Notas |
|---|---|---|---|
| ProtoNet vanilla (one-shot sem continual, sessão #20) | 94.55% | — | upper bound sem continual |
| Naive ProtoNet em CL (sessão #25) | 80.65% | -9.26 | baseline a bater |
| **5e sanity 50 tasks reduzido (esta sessão)** | **75.34%** | **-5.78** | **gap de -5.31 p.p. ACC, +3.48 p.p. BWT** |
| Possib. B (encoder linear, sessão #27 sanity) | 47.89% | -2.05 | -27.45 p.p. abaixo de 5e |

### Análise da configuração reduzida

Sanity rodou com:
- `n_inner=5` (vs default 10) — metade dos passos de plasticidade
- `finetune-episodes=30` (vs default 100) — 1/3 dos episodes de treino por task
- `eval-episodes=15` (vs default 50) — 1/3 dos episodes de avaliação

**Esperado com defaults completos:** ACC sobe (mais episodes de treino convergem melhor), BWT pode piorar levemente (mais drift). Estimativa: ACC 78-85%, BWT -6 a -10. Se atingir 82-85% em 5 seeds com defaults, **passa critério de sucesso (≥85%)** ou fica em zona "mediano" (81-85%) que justifica ablações.

### Sinais qualitativos do mecanismo

- **decay aprendeu pra cima** (0.500→0.639): network "quer" trace temporal mais longo. Indica termo `A·pre·post·trace` está sendo usado, não apenas ignorado.
- **BWT melhor que naive** (-5.78 vs -9.26): plasticidade local em camada final pode estar reduzindo forgetting comparado a backprop puro nas mesmas camadas.
- **Just-after acc 78-92%:** encoder + plasticidade adapta bem por task. Drop pra 75% no final é forgetting moderado mas mais leve que naive.

### Plano sessões #29-#37 (orientativo)

| # | Tipo | Objetivo |
|---|---|---|
| **29** | Code | Sanity 5 seeds completos (n_inner=10, finetune=100, eval=50). Tempo esperado: 30-60 min |
| 30 | Code | Ablação 1 — `--decay-fixed-zero` (remove trace, equivale c2_simplified em CL) |
| 31 | Code | Ablação 2 — `A=0` fixo (testa contribuição do termo Hebbian com trace) |
| 32 | Code | Ablação 3 — `--k-wta 64` (remove k-WTA, mantém densidade total) |
| 33 | Code | Ablação 4 — remove plasticidade (CNN+SGD vanilla = ProtoNet baseline em CL) |
| 34-35 | Code | EWC baseline em mesmo setup (compara com regularization) |
| 36-37 | Admin | Status check + análise + decisão sucesso/pivot |
| 38-42 | Code/admin | Paper draft ou refinement conforme #36-#37 |

Cancelable em qualquer ponto. Critério explícito (de STRATEGY.md "Decisão Pós-Sessão #27"): se #29 (5 seeds) não bater 75% ACC, sessão admin re-avalia se vale continuar 5e ou pivotar pra Caminho 2 (paper de robustez).

### Estado final pós-#28

- **Código novo:** `experiment_02_continual/c5e_combined.py` (~330 linhas, scaffold + implementação completa do modelo combinado).
- **Restrições respeitadas:** `model.py`, `config.py`, `baselines.py`, `c3_protonet_sparse.py`, `c2_simplified.py`, `c2_meta_hebbian.py`, `baseline_naive.py`, `c2_continual_arch_b.py` intocados. `c2_continual_arch_a.py` e `_c.py` ficam como scaffolds (não atacados nesta sessão).
- **Sanity passou em 2 escalas** (5 tasks: 71.20%; 50 tasks: 75.34%).
- Sessões consecutivas sem sinal>chance: 0 (75.34% > chance 20%).

### O que esta sessão NÃO fez (conforme protocolo)

- Não rodou 5 seeds completos
- Não fez ablações
- Não comparou com EWC
- Não tentou debug freestyle (não foi necessário — sanity passou)

Próxima sessão idealmente é #29 (sanity 5 seeds completos com defaults) pra confirmar magnitude do sinal antes de investir em ablações.
