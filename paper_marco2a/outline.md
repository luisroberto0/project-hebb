# Paper Marco 2-A — Outline detalhado

> Status: criado #58. Atualizar checkboxes conforme seções drafted.
> Word-count target total: 3500-4500 (paralelo paper C3 #35 = 3789 palavras).

---

## Title (tentative)

**"When Sparsity Stops Mattering: k-WTA Effect Collapse Under Extreme Domain Shift"**

Alternativas:
- "Domain-Locked Sparsity: Cross-Domain Failure of k-WTA in Few-Shot Learning"
- "Anti-Transfer in Bio-Inspired Encoders: Empirical Characterization on Omniglot→CUB"
- "From In-Domain Tolerance to Cross-Domain Irrelevance: A k-WTA Sparsity Study"

Decisão final pós-draft completo (Luis decide com material concreto).

---

## Estrutura geral

| Seção | Word target | Status |
|---|---|---|
| Abstract | 150-180 | ⏳ #62 |
| 1. Introduction | 600-800 | ✅ #58 |
| 2. Background | 700-900 | ✅ #58 |
| 3. Method | 500-700 | ⏳ #59 |
| 4. Experiments | 800-1000 | ⏳ #59-#60 |
| 5. Discussion | 700-900 | ⏳ #61 |
| 6. Conclusion | 150-200 | ⏳ #62 |
| **Total** | **~3500-4500** | — |

---

## Section 1: Introduction (✅ #58)

**Pontos-chave:**

- §1.1 Motivação: sparsity é hallmark cortical (Olshausen 1996, Ahmad 2019); paper C3 mostrou tolerância in-domain; pergunta natural cross-domain
- §1.2 Contribuição: 4 sparsities × 5 seeds × CUB cross-domain; decomposição de gargalos; anti-transfer characterization
- §1.3 Honestidade: não é "método novo cross-domain"; caracterização empírica com limitações
- §1.4 Roadmap

**Achado central a destacar:**
> "k-WTA preserves ProtoNet in-domain by 1.45 p.p. at 75% sparsity. Cross-domain to CUB-200, the same operation produces a sparsity-accuracy spread of 0.52 p.p. — statistical noise. The effect that mattered in-domain disappears under extreme domain shift."

---

## Section 2: Background (✅ #58)

**Pontos-chave:**

- §2.1 Cross-domain few-shot learning: Triantafillou 2020 (Meta-Dataset), Tseng 2020 (FWT), Chen 2019 (Closer Look), Phoo & Hariharan 2021 (STARTUP / extreme task differences). Gap: Omniglot single-source → CUB sem precedente direto.
- §2.2 k-WTA e sparse coding: Maass 2000 (k-WTA computational power), Olshausen 1996 (sparse coding), Ahmad 2019 (HTM). Reference paper C3 in-domain results.
- §2.3 Bio-plausible learning + transfer: in-domain successes vs. cross-domain limits (less established).
- §2.4 Datasets: Omniglot brief, CUB-200-2011 (Wah 2011) characterization. "Extreme task differences" framing (Phoo & Hariharan 2021).

---

## Section 3: Method (⏳ #59)

**Pontos-chave:**

- §3.1 Architecture: CNN-4 ProtoEncoder (Snell 2017), 64-D embedding, k-WTA over embedding
- §3.2 k-WTA layer: definição matemática top-k, gradient flow
- §3.3 Source training: Omniglot 5w1s5q, 5000 episodes, Adam lr=1e-3
- §3.4 Cross-domain protocol: encoder weights frozen, eval em CUB test split (28×28 grayscale, splits oficiais train_test_split.txt)
- §3.5 Sparsity sweep: k ∈ {8, 16, 32, 64}, sparsity ∈ {87.5%, 75%, 50%, 0%}
- §3.6 Sanity floors: Pixel kNN (sem encoder), Random encoder + k-WTA k=16 (sem treino)
- §3.7 Re-trained baseline: ProtoNet retreinado direto em CUB-200 train split, em 28×28 grayscale (compat) e 84×84 RGB (literatura standard, com AdaptiveAvgPool pra preservar 64-D embed)
- §3.8 Evaluation: 5 seeds × 1000 episodes 5w1s5q, IC95% bootstrap, mean +/- std inter-seed

---

## Section 4: Experiments (⏳ #59-#60)

**Tabela principal (Section 4.2):** 7 condições (5 seeds × 1000 eps cada)

| Modelo | Input | ACC | IC95% inter-seed |
|---|---|---|---|
| ProtoNet retreinado CUB 84×84 RGB | (3, 84, 84) | 49.84% | [49.38, 50.59] |
| ProtoNet retreinado CUB 28×28 gray | (1, 28, 28) | 34.31% | [34.06, 34.55] |
| Pixel kNN cross-domain | (1, 28, 28) raw | 22.81% | [22.69, 22.97] |
| C3 k=32 (50%) cross-domain | (1, 28, 28) | 22.20% | [21.77, 22.57] |
| C3 k=64 = ProtoNet baseline cross-domain | (1, 28, 28) | 22.13% | [21.90, 22.36] |
| C3 k=16 (75%) cross-domain | (1, 28, 28) | 22.09% | [21.84, 22.34] |
| Random encoder + k-WTA k=16 | (1, 28, 28) | 21.91% | [21.76, 22.03] |
| C3 k=8 (87.5%) cross-domain | (1, 28, 28) | 21.68% | [21.34, 22.04] |
| chance | — | 20.00% | — |

**Tabela 2 — In-domain vs cross-domain (Section 4.3):**

| k | Sparsity | Omniglot in-domain (paper C3) | CUB cross-domain | Δ in→cross |
|---|---|---|---|---|
| 8 | 87.5% | 90.77% | 21.68% | -69.09 p.p. |
| 16 | 75% | 93.10% | 22.09% | -71.01 p.p. |
| 32 | 50% | 93.35% | 22.20% | -71.15 p.p. |
| 64 | 0% vanilla | 94.55% | 22.13% | -72.42 p.p. |
| **Spread k=8 vs k=64** | — | **3.78** | **0.52** | — |

**Subsections:**

- §4.1 Setup: hardware (RTX 4070), tempo total wall-clock
- §4.2 Cross-domain sweep table (acima) + análise estatística (ICs sobrepostos)
- §4.3 In-domain vs cross-domain comparison (k-WTA effect collapse)
- §4.4 Bottleneck decomposition (decomposição random→C3→retreinado 28×28→retreinado 84×84)
- §4.5 Anti-transfer evidence (encoder treinado ≈ random)
- §4.6 Pixel kNN dominates encoded representations (achado contraintuitivo)

**Figuras (a gerar #60):**

- Fig 1: cross-domain accuracy bar chart com 7 condições + chance line. Highlight k-WTA spread.
- Fig 2: in-domain vs cross-domain dual-panel — left Omniglot (paper C3 numbers), right CUB cross-domain. Shows effect collapse visually.
- Fig 3 (opcional): bottleneck decomposition waterfall — random → C3 → retreinado 28×28 → retreinado 84×84.

---

## Section 5: Discussion (⏳ #61)

**Pontos-chave:**

- §5.1 Why k-WTA effect collapses cross-domain: hipótese mecanística — encoder Omniglot aprende features hyper-especializadas em traços binários; sucessivos MaxPools em CUB destroem informação crítica antes de qualquer sparsity ter efeito; pixel direto preserva mais que CNN-4 forwarding (achado contraintuitivo).
- §5.2 Anti-transfer mechanism: encoder treinado em fonte distante INTRODUZ viés, não generaliza. Padrão consistente com STARTUP (Phoo & Hariharan 2021): self-training na target é necessário em extreme task differences.
- §5.3 Implications for bio-plausible learning: sparsity é compatível in-domain (paper C3) mas neutra cross-domain. Não é antiviral nem benéfica — é invisível em transfer extremo.
- §5.4 Comparison with literature: Tseng 2020 mini-ImageNet→CUB ProtoNet baseline = 38%; Marco 2-A Omniglot→CUB ProtoNet baseline = 22%. Setup mais extremo, sinal mais fraco. Confirma escala de "extreme task differences".
- §5.5 Limitations: single source dataset, single target dataset, CNN-4 (não testou ResNet/ViT), 28×28 grayscale como input principal (84×84 RGB usado apenas no baseline retreinado).
- §5.6 Future work: testar source mais próxima (mini-ImageNet→CUB), testar k-WTA em camadas intermediárias, testar self-training na target (STARTUP-style).

---

## Section 6: Conclusion (⏳ #62)

**Pontos-chave (~150-200 palavras):**

- 1 parágrafo: contribuição central + quantitativos
- 1 parágrafo: implicação + trabalho futuro

---

## Material já disponível vs. a produzir

| Item | Disponível em | Status |
|---|---|---|
| 7 condições caracterizadas | `experiment_03_crossdomain/WEEKLY-1.md` | ✅ |
| Lit review (5 papers core) | `experiment_03_crossdomain/PAPERS.md` | ✅ |
| Decomposição mecanística | `WEEKLY-1.md` sessão #56 + #57 | ✅ |
| Tabela in-domain vs cross-domain | `WEEKLY-1.md` sessão #57 | ✅ |
| Scripts reproduzíveis | `experiment_03_crossdomain/*.py` | ✅ |
| Checkpoints | `experiment_01_oneshot/checkpoints/` | ✅ (gitignored) |
| Figuras 300 DPI | — | ⏳ #60 |
| Bibliography refs.bib | `paper_marco2a/refs.bib` | em curso #58 |
| LaTeX consolidado main.tex | — | ⏳ #63 |

---

## Cronograma estimado pós-#58

| Sessão | Conteúdo |
|---|---|
| #59 | Methods + Experiments draft |
| #60 | Figures + tabelas refinement |
| #61 | Discussion |
| #62 | Conclusion + abstract + slim revision |
| #63 | LaTeX consolidação + refs.bib final |
| #64-65 | Peer review interno |
| #66 | Admin obrigatória, decisão final |

Total: ~6-7 sessões. Dentro do limite hard #66.
