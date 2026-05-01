# Lit review — Cross-domain few-shot classification

> Notas estruturadas de papers core pra Marco 2-A. Foco: setup cross-domain, métricas reportadas, aplicabilidade ao C3, limitações.
> Sessão #52 (2026-05-14). 5 papers. Adicionar mais conforme exploração evolui.

---

## 1. Triantafillou et al. 2020 — Meta-Dataset

**Citation:** Triantafillou, E., Zhu, T., Dumoulin, V., Lamblin, P., Evci, U., Xu, K., ... & Larochelle, H. (2020). *Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples.* ICLR 2020.

### Mecanismo central

Não é um método — é um **benchmark**. Define avaliação cross-domain few-shot multi-source:

- 10 datasets: ImageNet, Omniglot, Aircraft, **CUB-200-2011**, Quick Draw, Fungi, VGG Flower, Traffic Signs, MSCOCO, MNIST
- Episódios variáveis: n_way e k_shot amostrados aleatoriamente por episódio (não fixo 5w1s)
- Class hierarchies preservadas (sample classes balanceadas por categoria)
- Treino em sub-conjunto de datasets, eval em todos (incluindo OOD)

### Setup cross-domain usado

- **Treino multi-source:** ImageNet + Omniglot + Aircraft + CUB + ... (8 datasets)
- **Test:** todos 10, incluindo Traffic Signs e MSCOCO que nunca aparecem no treino (OOD puros)
- Variação de domínio: natural images, characters, signs, sketches

### Métricas reportadas (ProtoNet, 5w1s típico)

| Test dataset | ACC ProtoNet (treino multi) |
|---|---|
| ImageNet | ~50% |
| Omniglot | ~95% (in-distribution) |
| Aircraft | ~50% |
| **CUB** | **~50-60%** |
| Quick Draw | ~50% |
| Fungi | ~38% |
| VGG Flower | ~85% |
| Traffic Signs (OOD) | ~50% |
| MSCOCO (OOD) | ~40% |

### Aplicabilidade ao C3

**Indireta.** Meta-Dataset usa multi-source training (incluindo CUB no treino, geralmente). Marco 2-A é cenário mais adversarial: **single-source** (Omniglot only) → CUB. Não há paper na literatura que faça exatamente isso.

Achado relevante: nenhum método domina cross-domain. Métodos best-in-class em ImageNet caem dramaticamente em domains distantes. Sugere que C3 (treinado só em Omniglot, dataset radicalmente diferente de CUB) terá desempenho cross-domain **fraco**.

### Limitações

- Multi-source treino mascara dificuldade do single-source extreme (que é nosso caso)
- 84×84 RGB padrão; nada sobre 28×28 grayscale (nossa primeira passada)
- Não cobre setup Omniglot-only → CUB

---

## 2. Tseng et al. 2020 — Cross-Domain Few-Shot via FWT

**Citation:** Tseng, H. Y., Lee, H. Y., Huang, J. B., & Yang, M. H. (2020). *Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation.* ICLR 2020.

### Mecanismo central

Adiciona **feature-wise transformation (FWT) layers** após cada batch-norm do encoder. FWT aplica scale γ e bias β learnados que simulam distribution shift cross-domain durante meta-treino. γ, β meta-aprendidos via outer loop (similar MAML).

Hipótese: meta-treino com FWT que perturba features durante treino força encoder a aprender representações mais robustas a shifts.

### Setup cross-domain usado

- **Treino:** mini-ImageNet (single-source) ou multi-source
- **Test:** CUB, Cars, Places, Plantae

### Métricas reportadas (5w1s, single-source mini-ImageNet → target)

| Method | mini-ImageNet (in-distrib) | CUB (cross) | Cars (cross) | Places | Plantae |
|---|---|---|---|---|---|
| MatchingNet | 49.8% | 35.9% | 30.8% | 49.9% | 32.7% |
| **ProtoNet** | **49.4%** | **38.0%** | **29.3%** | **53.3%** | **33.8%** |
| RelationNet | 49.3% | 42.4% | 29.1% | 48.6% | 33.2% |
| **+FWT (best)** | — | **47.5%** | **31.6%** | **55.8%** | **35.8%** |

### Aplicabilidade ao C3

**Importante referência.** Mostra ProtoNet baseline mini-ImageNet→CUB ≈ **38%** em 5w1s. Como mini-ImageNet é fonte muito mais próxima de CUB que Omniglot (RGB natural images vs binary characters), expectativa pra Omniglot→CUB é **abaixo de 38%**.

Predição refinada do Marco 2-A: ProtoNet baseline cross-domain (Omniglot→CUB) provavelmente em ~25-35% range. C3 cross-domain similar ou pior.

### Limitações

- Não usa Omniglot como source. Omniglot→CUB é setup ainda mais extremo.
- 84×84 RGB; não testa 28×28 grayscale.
- FWT requer meta-treino (não aplicável a C3 com encoder congelado).

---

## 3. Chen et al. 2019 — A Closer Look at Few-Shot Classification

**Citation:** Chen, W. Y., Liu, Y. C., Kira, Z., Wang, Y. C. F., & Huang, J. B. (2019). *A Closer Look at Few-Shot Classification.* ICLR 2019.

### Mecanismo central

**Análise crítica de baselines.** Não propõe método novo — propõe Baseline e Baseline++ (cosine-similarity classifier após pretrain padrão) e mostra que competem com métodos few-shot sofisticados.

**Achado central cross-domain (Section 4.2):** quando há domain shift entre treino e teste, baselines simples (Baseline, Baseline++) **superam** métodos meta-learning sofisticados (ProtoNet, MatchingNet, MAML, RelationNet).

### Setup cross-domain usado

- **Treino:** mini-ImageNet
- **Test cross-domain:** CUB-200
- Splits CUB padrão estabelecidos aqui (100 train / 50 val / 50 test classes), depois adotados por Tseng 2020 e outros

### Métricas reportadas (mini-ImageNet → CUB, 5w5s)

| Method | mini-ImageNet (in) | CUB (cross) | Δ |
|---|---|---|---|
| **Baseline** | 62.5% | **65.6%** | **+3.1** |
| **Baseline++** | 66.4% | **62.0%** | -4.4 |
| MatchingNet | 63.5% | 53.1% | -10.4 |
| ProtoNet | 64.2% | 62.0% | -2.2 |
| MAML | 63.1% | 51.3% | -11.8 |
| RelationNet | 65.3% | 57.7% | -7.6 |

Para **5w1s cross-domain** (Tabela 4 do paper): baselines ainda competem ou superam meta-learning quando domain shift presente.

### Aplicabilidade ao C3

**Direta e relevante.** Sugere que **encoder simples treinado em domínio fonte (Baseline = pretrain + cosine classifier) pode bater encoder meta-trained sofisticado em cross-domain**. C3 é encoder ProtoNet meta-trained — pode sofrer cross-domain pelo padrão documentado aqui.

Ponto contraintuitivo positivo pra C3: k-WTA pode atuar como regularizador que reduz overfit ao source, mantendo features mais genéricas. Mas: Chen 2019 testou só pretrain genérico (não encoders bio-inspirados), então é especulativo.

### Limitações

- mini-ImageNet → CUB é setup "moderado" (RGB→RGB, similar resolution). Omniglot→CUB é mais extremo.
- Não testa 28×28 grayscale.
- Não cobre k-WTA / sparsity como variável.

---

## 4. Phoo & Hariharan 2021 — STARTUP

**Citation:** Phoo, C. P., & Hariharan, B. (2021). *Self-training for Few-shot Transfer Across Extreme Task Differences.* ICLR 2021.

### Mecanismo central

**Self-training na target domain** com labels pseudo-rotulados pelo encoder source. Hipótese: domain gap massivo precisa adaptação ativa na target, não só treino source robusto.

Pipeline:
1. Encoder pretrained em source (com unlabeled access a target images)
2. Self-training iterativo: gerar pseudo-labels em target unlabeled, treinar com mistura source+pseudo-target
3. Eval few-shot na target

### Setup cross-domain usado

**Setup adversarial extremo** (BSCD-FSL benchmark de Guo et al. 2020):
- **Source:** ImageNet (mini ou full)
- **Targets:** ChestX (medical X-ray), ISIC (skin lesions), EuroSAT (satellite), CropDiseases (plant pathology)

Distance shift drástico entre source e target — esses domínios têm estatística visual muito distante de natural photos.

### Métricas reportadas (5w5s, "extreme task difference")

| Method | ChestX | ISIC | EuroSAT | CropDis |
|---|---|---|---|---|
| ProtoNet (mini-ImageNet) | 24.1% | 39.6% | 73.3% | 79.7% |
| MatchingNet | 22.4% | 36.7% | 64.4% | 66.4% |
| **STARTUP (proposto)** | **26.9%** | **47.2%** | **82.3%** | **93.0%** |

5w1s não-reportado diretamente em main tables, mas inferível como ~0.7× dos 5w5s.

### Aplicabilidade ao C3

**Conceitualmente próxima.** "Extreme task differences" é exatamente cenário de Marco 2-A (Omniglot→CUB). STARTUP confirma que ProtoNet baseline em extreme shift fica em **22-40% range pra 5w5s**, prováveis 15-30% pra 5w1s.

**Implicação central:** sem self-training na target, encoder source não atinge performance utilizável em extreme shift. C3 com encoder congelado (sem self-training) provavelmente fica em range similar.

Self-training não é aplicável diretamente ao Marco 2-A nesta passada (encoder C3 está congelado por design — preserva comparabilidade com paper C3). Pode ser extensão Marco 2-A.2 se #61-#62 justificarem.

### Limitações

- Não usa Omniglot como source. Mas extreme shift confirmado em outros setups.
- 5w5s mais reportado que 5w1s.
- Self-training requer unlabeled target — Marco 2-A não usa unlabeled CUB no encoder treino.

---

## 5. Wah et al. 2011 — CUB-200-2011 Dataset

**Citation:** Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011). *The Caltech-UCSD Birds-200-2011 Dataset.* Tech. Rep. CNS-TR-2011-001, California Institute of Technology.

### Conteúdo central

Paper de dataset. 200 bird species, 11.788 imagens, ~60/classe.

| Característica | Valor |
|---|---|
| Classes | 200 (bird species) |
| Total imagens | 11.788 |
| Imagens por classe | ~60 (range 41-60) |
| Resolução nativa | variável, ~500×500 típico |
| Canais | RGB |
| Annotations | bounding boxes (1 per image), part locations (15 parts), attribute labels (312 binary), class labels |
| Origem | extension of CUB-200 (Welinder 2010) com mais imagens e annotations |

### Splits cross-domain few-shot padrão

Estabelecidos posteriormente (não no paper original):

- **Hilliard et al. 2018** (early CUB few-shot): 100 train / 50 val / 50 test classes
- **Chen et al. 2019** (mais usado): mesmo split 100/50/50, eval no test
- **Tseng et al. 2020:** segue Chen 2019

Marco 2-A vai usar split Chen 2019 / Tseng 2020 (100 train / 50 val / 50 test) por consistência com literatura.

### Aplicabilidade ao Marco 2-A

**Dataset oficial.** Define a target domain. Sem este paper, não há experimento.

**Pré-processamento decisão:**
- Primeira passada (#52-#58): resize 28×28 grayscale (preserva comparabilidade C3)
- Segunda passada condicional (#61-#62): 84×84 RGB se #58 mostrar gap massivo

### Limitações do dataset

- Bird species é fine-grained (intra-class variation alta, inter-class às vezes sutil) — torna few-shot inerentemente difícil mesmo in-distribution
- Imagens têm bounding boxes mas usar/não-usar afeta dificuldade — Marco 2-A primeira passada **NÃO usa bounding boxes** (closer to "raw" cross-domain test)
- Resolução variável: padronizar pra single resolution introduz artifacts dependendo do método (center crop vs resize com aspect ratio preservation)

### Disponibilidade

- Site oficial: http://www.vision.caltech.edu/datasets/cub_200_2011/
- Tar size ~1.1 GB
- Mirrors: Hugging Face datasets, Kaggle (validar legitimidade antes de usar)
- `torchvision.datasets` **NÃO tem CUB-200 nativo** (até cutoff jan/2026)

---

## Síntese para Marco 2-A

### Predição refinada pós-lit-review

| Modelo | ACC 5w1s CUB esperado (28×28 grayscale, primeira passada) |
|---|---|
| Pixel kNN cross-domain | 22-28% (chance refs Tseng/Chen) |
| ProtoNet baseline (Omniglot encoder, congelado) | 25-35% (extrapola Tseng 38% mini-ImageNet→CUB pra source mais distante) |
| **C3 (encoder #20 Omniglot k=16, congelado)** | **20-35%** (similar ou pior que ProtoNet baseline) |
| ProtoNet **retreinado** em CUB (28×28 gray) | 35-55% (resolução baixa degrada vs 84×84 típico de literatura) |

**Gap esperado C3 vs ProtoNet retreinado:** −5 a −30 p.p. (predição mais conservadora pós-lit-review que predição inicial −25 a −50 p.p., porque resize 28×28 grayscale derruba ProtoNet retreinado também).

**Critério literal continua difícil:** C3 ≥ ProtoNet retreinado + 5 p.p. exigiria C3 ≈ 40-60% (alto pra cross-domain extreme em 28×28 grayscale).

### Setups com precedente direto

**Não há.** Omniglot single-source → CUB single-target em 28×28 grayscale **não está na literatura**. Esta exploração tem componente metodológica original (extreme shift case study). Documentar isso como contribuição honesta.

### Posicionamento do paper potencial (pós-#66)

Se Marco 2-A produzir achado, dois framings possíveis:

1. **Positivo (critério atingido):** "Bio-inspired sparse encoder generalizes across extreme visual domains" — improvável pela predição, mas seria contribuição forte se ocorrer.
2. **Negativo (critério falha):** "Limits of bio-inspired encoder transfer under extreme visual domain shift" — paper de caracterização honesta. Workshop-scope. Achado relevante pra missão pós-LLM (informa que encoders bio-inspirados precisam adaptação na target, conforme STARTUP / Phoo 2021 sugere).

Em ambos casos, Wah 2011 + Chen 2019 + Tseng 2020 + Phoo 2021 + Triantafillou 2020 são as 5 refs core pro paper.

### Refs adicionais a considerar conforme sessões evoluem

- Guo et al. 2020 — *A Broader Study of Cross-Domain Few-Shot Learning* (BSCD-FSL benchmark)
- Hilliard et al. 2018 — *Few-Shot Learning with Metric-Agnostic Conditional Embeddings* (split CUB original)
- Snell et al. 2017 — Prototypical Networks (já citado em paper C3)
- Lake et al. 2015 — Omniglot (já citado em paper C3)
