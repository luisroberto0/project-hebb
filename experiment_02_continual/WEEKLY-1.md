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
