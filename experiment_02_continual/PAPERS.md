# PAPERS.md — Literatura review pra Marco 1 (continual learning)

> Sessão #22 (2026-05-01). 5 papers core lidos via abstracts + key sections.
> Não é review exaustivo — é mapa funcional pra implementação dos baselines (#23-#25)
> e pra design das propostas C2-continual / C3-continual (#26-#28).
> Para framing, ver `experiment_02_continual/PLAN.md`.

Estrutura por paper:
- **Citação completa**
- **Mecanismo central** (1 frase)
- **Como evita catastrophic forgetting**
- **Métricas reportadas em benchmarks relevantes**
- **Aplicabilidade ao project-hebb (C2/C3)**
- **Custo computacional**

---

## 1. EWC — Elastic Weight Consolidation (Kirkpatrick et al. 2017)

**Citação:** Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences (PNAS)*, 114(13), 3521-3526.

### Mecanismo central
Penalidade quadrática nos pesos, ponderada pela **Fisher Information Matrix** (estimada como diagonal), pra desincentivar mudanças em parâmetros importantes pra tarefas anteriores.

### Como evita catastrophic forgetting
Loss durante treino da task `t`:
```
L_t(θ) = L_new(θ) + Σ_i (λ/2) · F_i · (θ_i − θ*_i)²
```
- `θ*_i`: valor do parâmetro `i` ao fim da task anterior.
- `F_i`: diagonal da Fisher Information no fim da task anterior — proxy de "quão importante esse parâmetro é pra a task que acabou de ser aprendida".
- `λ`: hyperparâmetro de regularização (típico 100-10000).

A Fisher é acumulada multiplicativamente entre tasks (ou alguma agregação) pra preservar info de todas as anteriores. **Sem buffer de exemplos** — só usa estatísticas de gradiente.

### Métricas reportadas
- **Permuted MNIST**: ~93-95% avg acc em 10 tasks, vs ~70% naive sequential.
- **Atari (várias tasks)**: bate baseline naive consistentemente.
- BWT não é métrica padrão do paper original (essa terminologia foi popularizada depois por GEM); mas avg acc é forte.
- **Não testou Split-Omniglot diretamente** — o paper usa MNIST e Atari.

### Aplicabilidade ao project-hebb (C2/C3)
- **C2 (plasticidade meta-aprendida):** EWC penaliza mudanças no nível dos PESOS. C2 já meta-aprende parâmetros de plasticidade (A, B, C, D) que controlam como pesos mudam. Pode-se adicionar termo Fisher-style **diretamente sobre os pesos adaptados** durante o inner loop, ou meta-aprender plasticidade que naturalmente reduz drift.
- **C3 (ProtoNet+k-WTA):** EWC aplica direto sobre os pesos do ProtoNet encoder. Implementação trivial — calcular Fisher após cada task, aplicar penalidade no treino da próxima.
- **Spiritualmente alinhado:** EWC é interpretado como aproximação Bayesian — preservar parâmetros importantes é compatível com plasticidade local moduada por importância.

### Custo computacional
- **Memória:** O(P) extra (uma Fisher diagonal por parâmetro). Pra ProtoNet (~110K params), trivial.
- **Compute:** uma passada extra de gradient evaluation no fim de cada task pra estimar Fisher. ~O(N · P) onde N = tamanho de sample pra Fisher.
- **Per-step durante treino:** termo extra de loss linear em P, negligível.
- **Escalabilidade pra 50 tasks:** boa em compute, mas a **agregação de Fisher entre tasks** (sum vs running average) afeta resultado — discutido na literatura subsequente.

### Anotações
- EWC é o **baseline obrigatório** pra Marco 1 (decisão pós-#20, item d).
- Referência pra implementação: existem várias implementações open-source (Avalanche library tem EWC; também há repos standalone).

---

## 2. SI — Synaptic Intelligence (Zenke et al. 2017)

**Citação:** Zenke, F., Poole, B., & Ganguli, S. (2017). Continual Learning Through Synaptic Intelligence. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, 70, 3987-3995.

### Mecanismo central
**Importância online via path integral.** Acumula durante treino a contribuição de cada parâmetro pra reduzir loss, usando integral do produto (gradient × parameter velocity). Não precisa de Fisher pass separada.

### Como evita catastrophic forgetting
Importância acumulada `ω_i` pra parâmetro `i` em task `μ`:
```
ω^μ_i = -∫_t (∂L/∂θ_i) · (dθ_i/dt) dt
```
(integral durante treino da task μ; sinal negativo pra contar quando parâmetros se moveram em direção que reduzia loss).

Importância normalizada por mudança quadrática:
```
Ω^μ_i = Σ_ν<μ ω^ν_i / [(Δ^ν_i)² + ξ]
```
onde `Δ^ν_i = θ_final^ν_i − θ_inicial^ν_i` e `ξ` é regularizador pra estabilidade.

Loss durante task `μ`:
```
L_μ(θ) = L_new(θ) + c · Σ_i Ω^μ_i · (θ_i − θ̃_i)²
```

### Métricas reportadas
- **Permuted MNIST** (10 tasks): ~96% avg acc — comparável ou melhor que EWC.
- **Split CIFAR-10/100** (subset-based): SI bate EWC marginalmente.
- **Não testou Split-Omniglot diretamente.**

### Aplicabilidade ao project-hebb (C2/C3)
- **C2:** SI é estruturalmente similar a EWC (regularização quadrática), mas com importance estimada **online durante treino**. Conceitualmente alinhado com plasticidade local: o accumulator `ω_i` é função de gradients e mudanças locais — compatível com plasticidade modulada por "experiência recente".
- **C3:** aplicável da mesma forma que EWC.
- **Vantagem sobre EWC:** não precisa de pass extra de Fisher → mais "online" e mais barato em compute, mas usa um pouco mais de memória pra trackear ω_i durante treino.

### Custo computacional
- **Memória:** O(P) pra Ω + O(P) pra ω corrente + O(P) pra last θ. ≈ 3× memória do modelo.
- **Compute:** custo extra durante treino é pequeno (atualizar ω_i a cada step com produto gradient × velocity). Sem pass extra entre tasks.
- **Escalabilidade:** Ω acumula entre tasks. Pra 50 tasks, sem problema.

### Anotações
- Pode ser implementado em paralelo a EWC pra Marco 1 — comparação útil entre regularização Bayesian-style (EWC) e path integral (SI).
- Tendência da literatura: EWC e SI são similares em performance; SI mais elegante operacionalmente.

---

## 3. GEM — Gradient Episodic Memory (Lopez-Paz & Ranzato 2017)

**Citação:** Lopez-Paz, D., & Ranzato, M. A. (2017). Gradient Episodic Memory for Continual Learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

### Mecanismo central
**Buffer episódico pequeno** com M exemplos por task. Durante treino da task atual, **restringe direção do gradiente** pra que loss em exemplos armazenados não aumente. Resolve QP por step.

### Como evita catastrophic forgetting
Pra cada task k armazenada, calcula gradient `g_k` da loss naqueles exemplos. Se `⟨g_t, g_k⟩ < 0` (gradiente atual aumentaria loss da task k), projeta `g_t` pra ficar ortogonal ou positivamente correlacionado com `g_k`:

```
min_g̃ ½ ||g̃ − g_t||²  s.t.  ⟨g̃, g_k⟩ ≥ 0  ∀k < t
```

QP no espaço dos gradientes. Solver dual permite cálculo eficiente quando número de tasks é pequeno.

**Crucial: GEM USA REPLAY BUFFER.** Não cumpre nossa restrição "sem replay buffer".

### Métricas reportadas
- **MNIST permutations** (10 tasks, M=256/task): ~89% avg acc.
- **MNIST rotations** (10 tasks): ~85% avg acc.
- **CIFAR-100 split** (20 tasks, M=256/task): bate EWC e SI.
- Introduz métrica **BWT** (backward transfer) e **FWT** (forward transfer) — métricas que viraram padrão da área.

### Aplicabilidade ao project-hebb (C2/C3)
- **NÃO é solução direta** — viola restrição "sem replay buffer".
- **MAS:** útil como **upper bound de referência**. Se GEM bate nossa proposta sem replay por X p.p., quantifica custo de não usar replay.
- **Conceito relevante:** ortogonalização de gradientes pode inspirar plasticidade meta-aprendida que aprenda a "mover pesos sem ferir tasks anteriores" via mecanismo local em vez de buffer explícito.

### Custo computacional
- **Memória:** O(n_tasks × M × example_size). Pra 50 tasks × M=256 × imagem 28×28 = ~10MB pra Omniglot. Não é problema.
- **Compute:** **QP por step de treino** — caro. Cresce com número de tasks no buffer. **Gargalo principal de GEM** em prática.
- **Escalabilidade pra 50 tasks:** problemática (motivou A-GEM).

### Anotações
- Não vamos implementar GEM diretamente — usa replay (viola restrição).
- Mas GEM **define as métricas BWT/FWT** que usamos.
- Pode aparecer como referência de "sky" no paper final pra mostrar custo de no-replay.

---

## 4. A-GEM — Averaged GEM (Chaudhry et al. 2019)

**Citação:** Chaudhry, A., Ranzato, M. A., Rohrbach, M., & Elhoseiny, M. (2019). Efficient Lifelong Learning with A-GEM. *International Conference on Learning Representations (ICLR)*.

### Mecanismo central
**GEM com average gradient** — em vez de uma constraint por task armazenada (caro), usa UMA constraint baseada no gradient médio sobre toda a memória episódica.

### Como evita catastrophic forgetting
Calcula `g_ref` = gradient médio em batch sampled da memória de tasks anteriores. Aplica **uma única projeção**:

```
g̃ = g_t                        if ⟨g_t, g_ref⟩ ≥ 0
g̃ = g_t − (⟨g_t, g_ref⟩/||g_ref||²) · g_ref   otherwise
```

Closed-form (sem QP solver) → muito mais rápido que GEM.

**Mantém replay buffer** (mesma violação de GEM da nossa restrição).

### Métricas reportadas
- **Permuted MNIST** (20 tasks): comparable a GEM, ~10× mais rápido.
- **Split CIFAR**: comparable a GEM.
- **Visual Decathlon (10 tasks)**: testado e bate baselines.

### Aplicabilidade ao project-hebb (C2/C3)
- **NÃO é solução direta** — usa replay.
- **Mais escalável que GEM** — referência mais relevante se quisermos comparar contra "GEM-family" eficiente.
- **Conceito relevante:** projeção do gradient médio é um operador linear simples — em princípio aproximável por **plasticidade meta-aprendida** que aprende a fazer projeção análoga sem buffer.

### Custo computacional
- **Memória:** mesma de GEM (replay buffer).
- **Compute:** O(P) por step (closed-form), vs O(n_tasks · P) ou QP de GEM.
- **Escalabilidade:** boa pra 50+ tasks.

### Anotações
- Bom candidato pra incluir como upper bound junto com GEM.
- Implementação trivial uma vez que GEM-style replay buffer está disponível.

---

## 5. Hadsell et al. 2020 — Embracing Change (review)

**Citação:** Hadsell, R., Rao, D., Rusu, A. A., & Pascanu, R. (2020). Embracing Change: Continual Learning in Deep Neural Networks. *Trends in Cognitive Sciences*, 24(12), 1028-1040.

### Mecanismo central (review, não método)
**Taxonomia de continual learning** dividida em 3 famílias principais:

1. **Regularization-based** (EWC, SI, MAS): penaliza mudanças em pesos importantes. Sem buffer.
2. **Replay-based** (GEM, A-GEM, Generative Replay, ER): mantém ou gera exemplos de tasks anteriores.
3. **Parameter isolation** (PathNet, Progressive Networks, PNN, Packnet): aloca parâmetros novos pra tasks novas, congela antigos.

### Como evita catastrophic forgetting (visão geral)
Cada família ataca o problema de forma diferente:
- **Regularization:** "não mexa nos pesos importantes" — pode ficar restritivo demais com muitas tasks.
- **Replay:** "lembre exemplos antigos" — viola constraints de privacy/memory mas geralmente performa melhor.
- **Parameter isolation:** "novos pesos pra novas tasks" — mais memória, mas evita interferência por construção.

### Métricas reportadas (revisitadas no paper)
- Apresenta meta-análise de avg acc / BWT em vários benchmarks (Permuted MNIST, Split CIFAR, etc).
- **Conclusão importante**: nenhuma família domina; trade-offs claros (regularization é mais leve mas menos efetivo em escala; replay é mais efetivo mas viola privacy/memory; isolation escala em parâmetros).

### Aplicabilidade ao project-hebb (C2/C3)
- **Define o landscape** que nosso trabalho ocupa: nossa restrição "sem replay" coloca nós na família **regularization** ou **isolation**, não replay.
- **Conexão com plasticidade biológica:** o paper discute que sinapses biológicas combinam mecanismos de TODAS as 3 famílias — meta-plasticidade (regularization), replay durante sleep (replay), e specialização anatômica (isolation). Sugere que abordagens híbridas são promissoras.
- **C2-continual:** pode-se enquadrar como "regularization meta-aprendida" — em vez de penalty quadrática fixa, aprende a regra que decide quanto cada peso deve mudar.
- **C3-continual:** k-WTA esparso é forma de **isolation suave** — diferentes top-k subsets pra diferentes tasks reduzem interferência.

### Custo computacional
- N/A (review).
- Mas o paper enfatiza que custo computacional é fator real pra deploy — alinha com nossa restrição "5h/semana, GPU laptop".

### Anotações
- **Posicionamento do nosso trabalho:** family regularization (com C2) ou hybrid regularization+isolation (com C3+plasticity meta).
- Cita extensively Bayesian framings de EWC — útil pra introdução do paper futuro.
- Bom paper pra "Related Work" do future paper.

---

## Síntese pra implementação dos baselines

### Sessão #23 (baseline naive)
- Encoder ProtoNet pré-treinado em background set Omniglot (já temos via baselines.py).
- Treinar sequencialmente em 50 tasks Split-Omniglot, sem nenhuma defesa contra forgetting.
- Medir **acc por task** logo após treino + **acc final em todas tasks** após treinar todas.
- BWT = média de (acc_final − acc_logo_após).
- Esperado: ACC ~30-50%, BWT ~−30 a −50% (forgetting brutal).

### Sessão #24-#25 (EWC baseline)
- Mesma infra do baseline naive.
- Adicionar: cálculo de Fisher diagonal após cada task (~256 samples sufficient per literatura).
- Loss = L_new + λ · Σ F_i · (θ_i − θ*_i)².
- Hyperparam crítico: λ. Literatura sugere busca em [10, 100, 1000, 10000]. Vamos começar com λ=100 (default Avalanche).
- Esperado: ACC ~50-70%, BWT ~−10 a −20%.

### Sessões #26-#28 (propostas)
- **C2-continual:** meta-aprender plasticidade que naturalmente preserva tasks antigas. Possível mecanismo: termo D_i (bias por peso) aprendido pra ser função de "drift recente" ou similar. **Não detalhado ainda — design fica pra sessão #26.**
- **C3-continual:** ProtoNet + k-WTA + algum mecanismo extra pra forgetting. Possível: k-WTA por task (cada task usa subset diferente das 64 dimensões), ou EWC sobre pesos do ProtoNet com k-WTA aplicado. **Detalhamento na sessão #26.**

### Sky (referência) e Floor
- **Sky baseline (replay):** GEM ou A-GEM, ~80% ACC esperada. Mostra "se você puder usar replay".
- **Floor baseline:** naive sequential, ~40% ACC.
- Nossas propostas (C2-continual, C3-continual) precisam ficar entre os dois, idealmente próximas ao sky.

---

## Lacunas identificadas

Coisas que NÃO encontramos nesses 5 papers e ficam pra leitura adicional se necessário:

- Specifics de **Split-Omniglot 50-tasks** como benchmark — qual paper estabeleceu o setup exato (5 chars/task vs outro split)? Possivelmente Schwarz et al. 2018 "Compress and Compare" ou outro.
- **Continual learning com features esparsas (k-WTA estilo)** — não foi tema central de nenhum paper acima. Possível leitura: Aljundi et al. 2018 "Selfless Sequential Learning" usa esparsidade.
- **Meta-learning aplicado a continual learning sem replay** — Javed & White 2019 "Meta-Learning Representations for Continual Learning" pode ser relevante. Adicionar à fila se #26 precisar de referência.

Estas lacunas não bloqueiam #23 (baseline naive). Resolvíveis em leituras curtas (~30 min cada) se ficarem críticas em sessões posteriores.
