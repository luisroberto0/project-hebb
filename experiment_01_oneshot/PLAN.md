# Experimento 01 — One-shot Omniglot com STDP + Memória Episódica

> Primeiro experimento da pesquisa em IA biologicamente inspirada.
> Objetivo: provar que plasticidade local + memória esparsa conseguem
> aprender com 1 exemplo por classe — algo que LLMs e CNNs convencionais
> não fazem sem retreino massivo.

---

## 1. Pergunta de pesquisa

**Uma rede de spiking neurons treinada por STDP (não-supervisionado, local, sem backprop) e acoplada a uma memória episódica esparsa consegue classificação one-shot competitiva no Omniglot, usando uma fração dos parâmetros das abordagens de meta-learning baseadas em backprop?**

Versão pop: aprender um caractere novo vendo ele uma única vez, do jeito que o cérebro faz, em um notebook de consumidor.

---

## 2. Por que essa pergunta importa

Três respostas, em ordem de ambição:

**Curto prazo (publicável).** A literatura de SNN é dominada por surrogate gradients — aprender spikes com backprop, basicamente deep learning disfarçado. Pouquíssimos trabalhos exploram STDP combinado com memória episódica moderna (Hopfield) pra few-shot. Isso é um nicho com espaço pra contribuição mensurável.

**Médio prazo (tese).** Se funciona em Omniglot, abre caminho pra atacar memória episódica de verdade — armazenar não só padrões visuais mas eventos temporais, contextos. Esse é o pilar 2 da sua pesquisa de 3 pilares (junto com causal e one-shot).

**Longo prazo (visão).** Se você consegue um sistema que aprende rapidamente com poucos exemplos e roda em CPU/GPU consumer, você está provando o princípio fundamental do projeto: inteligência não precisa de força bruta de GPU. Cada paper nessa linha é um tijolo na direção do "cérebro funcional" que você quer construir.

---

## 3. Hipótese

Features aprendidas por STDP convolucional são esparsas e ortogonais o suficiente pra que uma camada de memória episódica do tipo Hopfield Moderna (Ramsauer et al. 2020) consiga armazenar e recuperar protótipos de classe a partir de exposições únicas, atingindo:

- **5-way 1-shot:** ≥ 90% acurácia (target: dentro de 5% do Prototypical Networks)
- **20-way 1-shot:** ≥ 70% acurácia
- **Parâmetros treináveis:** < 100k (vs ~110k do MAML em CNN-4)
- **Sem backprop end-to-end:** STDP pra features, memória episódica pra inferência, readout linear opcional só pra calibração

Hipótese-nula informativa: se as features STDP forem indistinguíveis de features aleatórias (random projection), a hipótese principal cai.

---

## 4. Arquitetura

```
            Imagem 105×105 (Omniglot)
                    │
                    ▼
        ┌───────────────────────────┐
        │  Codificação em spikes    │   T=100 timesteps,
        │  (Poisson rate coding)    │   intensidade pixel → taxa
        └───────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │  Conv-STDP layer 1        │   8 filtros 5×5,
        │  LIF + STDP local         │   max-pool temporal 2×2
        └───────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │  Conv-STDP layer 2        │   16 filtros 5×5,
        │  LIF + STDP local         │   max-pool 2×2
        └───────────────────────────┘
                    │
                    ▼
              flatten + L2-norm
                    │
                    ▼
        ┌───────────────────────────┐
        │  Memória episódica        │   Modern Hopfield Net,
        │  (Hopfield Moderna)       │   armazena protótipos
        └───────────────────────────┘
                    │
                    ▼
        argmin distância → classe
```

**Por que cada peça:**

- **Codificação Poisson** é o padrão biológico — pixels mais brilhantes → mais spikes por unidade de tempo. Alternativa testada na fase 2: temporal coding (time-to-first-spike), mais parcimonioso.
- **LIF (Leaky Integrate-and-Fire)** é o neurônio de complexidade mínima que captura dinâmica essencial: integra entradas, decai, dispara, reseta.
- **STDP local** atualiza pesos olhando só pra timing pré/pós-sináptico. Sem gradiente, sem global error signal — é o que permite o sistema rodar online e em hardware neuromorfo (Loihi, SpiNNaker) no futuro.
- **Modern Hopfield Net** é a chave. Diferente do Hopfield clássico (capacidade O(N)), o moderno tem capacidade exponencial e é matemática equivalente à atenção do Transformer. Mas aqui usamos com pesos ligados a memórias armazenadas (não treinados), o que dá comportamento episódico real.
- **L2-norm** antes da memória estabiliza similaridade (cosseno em vez de produto interno bruto).

---

## 5. Protocolo de treino

**Fase A — Pretreino STDP (offline, não-supervisionado).**
Usa o split de "background" do Omniglot: 30 alfabetos, ~1200 caracteres, ~24k imagens. Apresenta cada imagem por T timesteps, deixa STDP atualizar pesos das duas camadas convolucionais. Sem labels. Critério de parada: estabilização da norma média dos pesos (early stopping baseado em sliding window).

Hiperparâmetros iniciais (Diehl & Cook 2015, adaptados):
- `tau_pre = tau_post = 20 ms`
- `A_pre = 0.01, A_post = -0.0105`
- `w_max = 1.0, w_init ~ U(0, 0.3)`
- Inibição lateral entre filtros da mesma camada (winner-take-all soft)

**Fase B — Avaliação few-shot (online, sem mais aprendizado de pesos).**
Para cada episódio (1000 episódios por configuração):
1. Sorteia N classes do split de "evaluation" (20 alfabetos, ~420 caracteres novos)
2. Sorteia K exemplos de support por classe (K=1 pra one-shot)
3. Passa cada support pela rede STDP (já congelada), extrai vetor de features, armazena na memória Hopfield
4. Para cada query, extrai features, recupera classe mais próxima na memória
5. Mede acurácia do episódio

**Fase C (opcional) — Readout linear supervisionado.** Se a similaridade pura na memória não for suficiente, treina uma camada linear sobre as features STDP usando supervisão few-shot (logistic regression em scikit-learn em cima das ativações). Isso ainda é sem backprop end-to-end e mantém o espírito biológico.

---

## 6. Baselines de comparação

| Baseline | Tipo | Parâmetros | Backprop end-to-end | Acurácia 5w1s esperada |
|---|---|---|---|---|
| Random | trivial | 0 | n/a | 20% |
| Pixel kNN | trivial | 0 | não | ~70% |
| Siamese Net (Koch 2015) | DL | ~3M | sim | ~92% |
| Matching Networks (Vinyals 2016) | DL | ~110k | sim | ~98% |
| Prototypical Networks (Snell 2017) | DL | ~110k | sim | ~98% |
| MAML (Finn 2017) | meta-DL | ~110k | sim | ~98% |
| **STDP + Hopfield (este trabalho)** | bio | <100k | **não** | target ≥90% |

A coluna importante é "Backprop end-to-end". Nosso ponto não é bater o estado da arte em acurácia bruta — é fazer competitivamente *sem backprop global*, *sem labels no pretreino*, *com plasticidade local*.

---

## 7. Métricas

**Primária:** Acurácia média em 1000 episódios de 5-way 1-shot e 20-way 1-shot.

**Secundárias:**
- 5-way 5-shot e 20-way 5-shot (pra ver se ganho com mais exemplos é razoável)
- Esparsidade média das ativações (% de neurônios disparando) — um teste de fidelidade biológica
- Tempo de inferência por episódio (CPU vs GPU)
- Robustez: acurácia com transformações (rotação, escala, ruído)

**Diagnósticas:**
- Similaridade entre features de mesma classe vs classes diferentes (clustering visualizado com t-SNE)
- Distribuição de pesos finais por filtro (esperamos formas tipo "Gabor" ou trechos de borda)
- Curva de aprendizado durante a fase A (norma de peso ao longo do tempo)

---

## 8. Roadmap (6 semanas)

> **Status atualizado 2026-04-27 (sessão #6):** Semana 1 fechada como caso patológico documentado. Semana 2 ativa. Detalhes de status por semana abaixo.

**Semana 1 — Sanity check. — CONCLUÍDA COM RESSALVA.**
Reproduzir Diehl & Cook 2015 em MNIST. Critério original: STDP não-supervisionado ≥85%.
- **Resultado obtido:** 17.76% (5 sessões de iteração, espaço de hiperparâmetros exaurido)
- **Status:** caso patológico documentado — MNIST com kernel=28 produz output espacial (1,1) que torna k-WTA degenerado (todos filtros disputam mesma posição). Ver `WEEKLY-1.md` § resumo executivo final e `PLAN.md` raiz § decisão arquitetural de 2026-04-27 sobre o pivot.
- **Stack validada por outros meios:** `tests/test_assignment.py` (assign_labels e evaluate corretos via 3 casos sintéticos), `tests/test_spike_balance.py` (razão pré:pós-spikes mensurável), homeostasis implementada e validada mecanicamente (theta com variância controlada, distribuição de filtros uniformizável).
- **Decisão:** aceitar ressalva, prosseguir pra Semana 2 onde a arquitetura conv real (kernel=5 + pool, output multi-posição) muda fundamentalmente a dinâmica de k-WTA.

**Semana 2 — Adaptação pra Omniglot. — ATIVA.**
Pipeline de dados (split background/evaluation, augmentação opcional, codificação spike). Conv-STDP layer 1 funcionando. Inspeção visual dos filtros aprendidos.
- **Infra disponível:** `data.py` (Omniglot loader + EpisodeSampler validados), `train.py` (loop pretreino com TensorBoard), `model.py:STDPHopfieldModel` (2 layers conv + pool + memória Hopfield), `evaluate.py` (N-way K-shot com IC bootstrap), `baselines.py` (Pixel kNN + ProtoNet). Pipeline validado end-to-end em sessões anteriores; primeiro experimento Omniglot ainda não executado.
- **Próximos passos detalhados:** ver `WEEKLY-2-NEXT.md`.

**Semana 3 — Pipeline completo.**
Conv-STDP layer 2 + extração de features + memória Hopfield. Primeira medição em 5-way 1-shot. Mesmo se for ruim (40-60%), é o sinal de que o sistema fecha.

**Semana 4 — Tuning e baselines.**
Implementar Pixel kNN e Prototypical Networks como comparação. Tunar hiperparâmetros do STDP via grid search pequeno (10-20 configurações).

**Semana 5 — Avaliação completa.**
1000 episódios em todas as configs (5w1s, 5w5s, 20w1s, 20w5s). Análise estatística (intervalos de confiança bootstrap). Diagnósticos visuais.

**Semana 6 — Documentação e iteração.**
Notebook de análise reproduzível, escrita de results, decisão sobre próximos experimentos. Se resultados promissores: rascunho de paper pra workshop (ICLR Tiny Papers, NeurIPS workshops). Se não: análise de falha e replanejamento.

---

## 9. Ferramentas

- **PyTorch + snnTorch** — codificação spike, LIF, treinamento
- **Brian2** — protótipos de regras STDP customizadas (mais legível que código vetorizado)
- **NumPy + SciPy** — análise pós-hoc
- **scikit-learn** — baselines (kNN, logistic regression)
- **TensorBoard / wandb** — logging
- **matplotlib + seaborn** — visualizações de filtros, t-SNE, curvas

---

## 10. Riscos e mitigações

**Risco 1 — Features STDP não generalizam pra Omniglot.** STDP convolucional foi validado em MNIST mas Omniglot tem mais variação de estilo. Mitigação: começar com STDP global (não convolucional) e só adicionar convolução depois que rate coding básico funcionar.

**Risco 2 — Capacidade da memória Hopfield insuficiente pra 20-way.** Modern Hopfield tem capacidade exponencial em teoria, mas na prática features colineares colidem. Mitigação: aumentar dimensionalidade do embedding pré-memória; experimentar embeddings binários (mais ortogonais por construção).

**Risco 3 — Pretreino STDP demora muito (24k imagens × 100 timesteps).** Mitigação: começar com subset de 500 imagens; otimizar com `prefs.codegen.target = "cython"` em Brian2 ou implementar STDP vetorizado em PyTorch puro.

**Risco 4 — "Não bate o estado da arte" como crítica de revisor.** Mitigação: o framing precisa ser claro desde o título — não é competição com MAML em acurácia, é prova de princípio biológica com restrições de hardware/algoritmo. Análogo ao que Lillicrap fez com feedback alignment.

---

## 11. Critérios de sucesso e falha

**Sucesso forte (paper publicável em workshop):**
- 5w1s ≥ 90% e 20w1s ≥ 70%
- Filtros visualmente coerentes (formas tipo Gabor)
- Esparsidade média ≥ 90% (% de neurônios silenciosos)

**Sucesso parcial (insight sólido, replanejar):**
- 5w1s entre 70-90%, 20w1s entre 50-70%
- Sistema funciona end-to-end mas precisa de truques (readout supervisionado etc.)
- Vale documentar e publicar como negative result instrutivo

**Falha (replanejar):**
- 5w1s < 70%
- Investigar: features são ruins? memória não armazena bem? codificação spike inadequada?

Em qualquer caso, em 6 semanas você sai com aprendizado profundo sobre uma direção concreta, e isso é o que define progresso real em pesquisa.

---

## 12. Referências essenciais

**Plasticidade e SNN:**
- Diehl, P. U., & Cook, M. (2015). *Unsupervised learning of digit recognition using spike-timing-dependent plasticity*. Frontiers in Computational Neuroscience.
- Song, S., Miller, K. D., & Abbott, L. F. (2000). *Competitive Hebbian learning through spike-timing-dependent synaptic plasticity*. Nature Neuroscience.
- Tavanaei, A., et al. (2019). *Deep learning in spiking neural networks*. Neural Networks.

**Memória episódica moderna:**
- Ramsauer, H., et al. (2020). *Hopfield Networks Is All You Need*. ICLR 2021.
- Krotov, D., & Hopfield, J. J. (2016). *Dense associative memory for pattern recognition*. NeurIPS.

**Few-shot e Omniglot:**
- Lake, B. M., et al. (2015). *Human-level concept learning through probabilistic program induction*. Science. ← O paper fundador do Omniglot.
- Snell, J., et al. (2017). *Prototypical Networks for Few-shot Learning*. NeurIPS.
- Finn, C., et al. (2017). *Model-Agnostic Meta-Learning*. ICML.

**Conv-STDP em particular:**
- Kheradpisheh, S. R., et al. (2018). *STDP-based spiking deep convolutional neural networks for object recognition*. Neural Networks.
- Falez, P., et al. (2019). *Multi-layered Spiking Neural Network with Target Timestamp Threshold Adaptation*. IJCNN.
