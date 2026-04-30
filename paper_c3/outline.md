# Outline detalhado — paper C3

Estrutura final esperada: 6-8 páginas (workshop format).

---

## Title (TBD final)

Tentative: **"k-WTA Sparsity Preserves Prototypical Network Performance in Few-Shot Learning"**

Alternativas:
- "Biologically-Inspired Sparsity in Prototypical Networks: An Empirical Analysis on Omniglot"
- "How Much Sparsity Can Prototypical Networks Tolerate? An Empirical Study with k-WTA"

## Abstract (TBD — escrever depois de Discussion fechar)

~150-200 palavras. Estrutura: pergunta → método → resultados principais → contribuição.

---

## Section 1: Introduction (~1 página, ~500-700 palavras) ✅ draft sessão #31

**Subsection 1.1: Hook**
Sparsity neural-inspirada é caracteristica da computação cortical (Olshausen & Field 1996; Ahmad & Hawkins 2019). Deep learning mainstream usa representações densas. Pergunta natural: quanto sparsity métodos modernos toleram?

**Subsection 1.2: Gap**
- Few-shot learning via Prototypical Networks (Snell 2017) é well-studied, mas sparsity não é restrição padrão
- k-WTA é estudado em Hopfield (Maass 2000), Sparse Distributed Memory (Kanerva 1988), arquiteturas Numenta-style — pouca literatura aplicando a prototype-based metric learning
- Gap: ninguém quantificou explicitamente quanto sparsity ProtoNet tolera em few-shot benchmarks

**Subsection 1.3: Contribuição**
Empirical study sobre Omniglot: k-WTA esparso aplicado ao embedding final de ProtoNet, treinado end-to-end com gradiente fluindo pelos top-k channels. 3 níveis de sparsity (50%, 75%, 87.5%). Validação obrigatória com random encoder + k-WTA pra isolar contribuição do treino.

**Subsection 1.4: Roadmap do paper**
1 frase por seção seguinte.

**Figs/tabelas planejadas:** nenhuma na intro.

---

## Section 2: Background (~1 página, ~500-700 palavras) ✅ draft sessão #31

**Subsection 2.1: Prototypical Networks**
Snell 2017. Pipeline: encoder CNN-4 → embedding → centróide por classe → distância euclidiana. Treinado episodically com cross-entropy sobre `-cdist²`. Estado da arte simples e forte em Omniglot.

**Subsection 2.2: k-WTA e sparse coding**
k-WTA em redes neurais (Maass 2000): determinismo competitivo, mantém top-k ativações. Conexão com sparse coding (Olshausen & Field 1996), Sparse Distributed Memory (Kanerva 1988), HTM (Ahmad & Hawkins 2019).

**Subsection 2.3: Omniglot benchmark**
Lake 2015. 1623 caracteres, 50 alfabetos. Padrão de fato pra few-shot. ProtoNet vanilla atinge ~98% em 5w1s no setup original (encoder maior).

**Subsection 2.4: Trabalhos relacionados**
Brevemente:
- ProtoNet variants (Matching Nets, Vinyals 2016; MAML, Finn 2017) — sem foco em sparsity
- Sparse representations em deep learning (vários, mas não em prototype methods)
- Biologically plausible deep learning (review papers)

**Figs/tabelas planejadas:** Figura 1 (opcional) — diagrama da arquitetura ProtoNet+kWTA.

---

## Section 3: Method (~1 página) ✅ draft sessão #33

**Subsection 3.1: Arquitetura C3**
ProtoEncoder CNN-4 padrão (Snell 2017): 4 blocos Conv-BN-ReLU-MaxPool, 64 filtros, output 64D. k-WTA aplicado no embedding final.

**Subsection 3.2: k-WTA layer**
`top-k(z)` mantém k maiores ativações por exemplo, zera o resto. Aplicado em training E eval (consistência). Gradient flui pelos top-k channels — encoder aprende a colocar info útil nas k dimensões dominantes.

**Subsection 3.3: Protocolo experimental**
- 3 níveis: k=32 (50% sparsity), k=16 (75%), k=8 (87.5%)
- Treino: 5000 episodes 5w1s, Adam lr=1e-3, mesma loss prototypical
- Eval: 1000 eps 5w1s e 20w1s, IC95% bootstrap
- Validação obrigatória: random encoder + k-WTA k=16 sem treino

**Figs/tabelas planejadas:** Figura 2 — diagrama do pipeline com k-WTA layer destacada.

---

## Section 4: Experiments (~2 páginas) ✅ draft sessão #33

**Subsection 4.1: Setup**
Hardware (RTX 4070 laptop), seed=42, baselines comparados (Pixel kNN 45.76%, ProtoNet vanilla 94.55%).

**Subsection 4.2: Resultados principais**
Tabela 1: 3 sparsities × 5w1s + 20w1s × IC95%. Comparação com ProtoNet baseline e Pixel kNN.

**Subsection 4.3: Curva sparsity × accuracy**
Figura 3: curva mostrando 0% (94.55), 50% (93.35), 75% (93.10), 87.5% (90.77) — destacar plateau até 75%.

**Subsection 4.4: Validação random encoder + k-WTA**
Tabela ou caixa: random encoder + k-WTA k=16 = 37.60%. Discussão: confirma que ganho vem do treino, não da estrutura k-WTA estática.

**Subsection 4.5: Robustez 20w1s**
Discussão de como sparsity escala em N-way maior — 20w1s mantém 75-82% mesmo com sparsity alta.

**Figs/tabelas planejadas:**
- Tabela 1: resultados principais
- Figura 3: curva sparsity × ACC
- Tabela 2 ou box: validação random+kWTA

---

## Section 5: Discussion (~1-2 páginas) — TODO sessão #33

**Subsection 5.1: Por que k-WTA não derruba ProtoNet?**
Hipóteses mecanísticas:
1. Esparsidade já existe naturalmente em CNN-4+ReLU+MaxPool
2. Treino redistribui informação pras top-k dimensões (gradient signal)
3. cdist² entre vetores esparsos preserva discriminação

**Subsection 5.2: Implicações pra bio-plausible deep learning**
Sparsity neural-inspirada é compatível com prototype-based metric learning. Sugere que features esparsas podem coexistir com gradient-based optimization.

**Subsection 5.3: Limitações**
- Omniglot é dataset visualmente simples
- k-WTA aplicado só no embedding (não em camadas intermediárias)
- Backprop end-to-end usado (não plasticidade local)
- Cross-domain generalização TBD

**Subsection 5.4: Trabalho futuro**
- Sweep mais fino de k
- k-WTA em camadas intermediárias
- Combinação com plasticidade local meta-aprendida (referência a Marco 1 do projeto, opcional)
- Datasets cross-domain

**Subsection 5.5: Continual learning como bonus** (opcional, se decidir incluir Marco 1 como apêndice)
Mencionar brevemente que ProtoNet+kWTA foi testado em continual setup mas resultados ficaram abaixo do baseline naive — achado mecanístico documentado em apêndice.

---

## Section 6: Conclusion (~1 parágrafo) — TODO sessão #33

Recap em 3-4 frases: pergunta → método → resultado principal → implicação.

---

## Appendix A (opcional) — Marco 1: Caracterização de Robustez de ProtoNet a Catastrophic Forgetting

Conteúdo: 1-2 páginas resumindo achados de `experiment_02_continual/WEEKLY-1.md`:
- ProtoNet sequencial é naturalmente robusto a forgetting em Omniglot (~80% ACC)
- Mecanismos bio-inspirados (plasticidade meta-aprendida, trace STDP-like, k-WTA esparso) não conseguem bater
- Causa: prototypes-fresh-no-eval + encoder métrica genérica
- Insight pra pesquisa futura: plasticidade na camada errada (após CNN) não previne CNN drift

Decisão final sobre incluir apêndice: na sessão #33 (depende de espaço e fit narrativo).

---

## Tabelas e figuras planejadas (consolidado)

| # | Tipo | Conteúdo | Pronto? |
|---|---|---|---|
| Fig 1 | Diagrama (opcional) | Arquitetura ProtoNet vanilla | TODO |
| Fig 2 | Diagrama | Pipeline ProtoNet + k-WTA | TODO #34 |
| Fig 3 | Plot | Sparsity × ACC (curva) | TODO #34 (dados prontos) |
| Tab 1 | Resultados | 3 sparsities × 5w1s/20w1s + IC95% + comparação baselines | TODO #32 (dados prontos) |
| Tab 2 | Validação | Random encoder + k-WTA vs trained | TODO #32 (dados prontos) |

Figuras gerar via matplotlib a partir dos outputs já cacheados (sessão #20).

---

## Word count target (workshop, ~6-8 páginas)

| Seção | Palavras alvo |
|---|---|
| Abstract | 150-200 |
| Introduction | 500-700 |
| Background | 500-700 |
| Method | 500-700 |
| Experiments | 800-1000 |
| Discussion | 800-1000 |
| Conclusion | 100-150 |
| Total (sem refs/apêndice) | ~3500-4500 |

Conferir contra template oficial do workshop quando NeurIPS 2026 anunciar formato.
