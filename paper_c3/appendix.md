# Appendix A — Marco 1: ProtoNet Robustness to Catastrophic Forgetting

> Status: SUPPLEMENTARY MATERIAL (sessão #34).
> **Decisão pós-#34: NÃO incluir no main paper draft.**
> Disponível como material suplementar mediante solicitação ou em `experiment_02_continual/` no repositório.
>
> Justificativa da decisão (registrada em `paper_c3/README.md`):
> - Marco 1 explora continual learning sem replay; main paper foca sparsity.
> Narrativa diferente, conexão é "ambos vinculados ao mesmo projeto pós-LLM",
> mas inclusão pode parecer filler num workshop paper de 6-8 páginas.
> - Conteúdo abaixo permanece como rascunho útil pra:
>   (a) submissão como supplementary material se workshop permitir
>   (b) seção de blog post / repositório associado ao paper
>   (c) base pra paper futuro standalone sobre robustez de ProtoNet a forgetting
>
> Word count: ~500 palavras.

---

## A.1 Setup

Em exploração paralela ao trabalho principal sobre sparsity, testamos ProtoNet em setting de continual learning sequencial sem replay buffer. O objetivo era avaliar se mecanismos bio-inspirados (plasticidade meta-aprendida, trace STDP-like, k-WTA esparso) poderiam superar baseline naive em forgetting reduction.

**Benchmark:** Split-Omniglot por alfabeto. Os 50 alfabetos do Omniglot (30 background + 20 evaluation) viraram 50 tasks sequenciais. Para cada task, amostramos 14 caracteres do alfabeto (todos os 50 alfabetos têm ≥14 caracteres), com split determinístico de 14 instâncias treino + 6 teste por caractere. Episódios 5-way 1-shot 5-query amostrados internamente em cada task.

**Protocolo:** Treino sequencial de 50 tasks, sem warmup, sem replay buffer, sem nenhuma defesa explícita contra catastrophic forgetting no baseline naive. Cada task é treinada por 100 episódios (Adam lr=1e-3, mesma loss prototypical do main paper). Avaliação ao final em 50 episódios por task pra cada uma das 50 tasks.

**Métricas:** Average Accuracy (ACC) sobre as 50 tasks, e Backward Transfer (BWT) — média de (acc_final[t] − acc_just_after[t]) — onde negativo indica forgetting.

## A.2 Quatro abordagens testadas

| Abordagem | ACC 5w1s | BWT (p.p.) | Mecanismo |
|---|---|---|---|
| Naive ProtoNet (random splits + warmup) | 82.58% | −12.46 | sem defesa; setup standard |
| Naive ProtoNet (alphabets + skip warmup) | 80.65% | −9.26 | mais adversarial; baseline final |
| Possibilidade B (encoder linear + plasticidade meta-aprendida) | 47.89% | −2.05 | gap de capacity vs CNN-4 |
| Caminho 5e (CNN-4 + plasticidade + trace + k-WTA) | 74.78% | −16.80 | combina mecanismos do main paper com plasticidade local |

**Resultado central:** nenhuma abordagem bio-inspirada superou o baseline naive ProtoNet em ACC. Caminho 5e (que combina k-WTA do main paper com plasticidade local meta-aprendida) ficou 5.87 p.p. abaixo do naive em ACC e 7.54 p.p. pior em BWT.

## A.3 Achado mecanístico

ProtoNet metric learning é **inerentemente robusto a catastrophic forgetting em Omniglot**. Razões prováveis:

1. **Não há classifier head treinado pra "esquecer".** Prototypes são computados FRESH a cada episódio do support set. O único componente que persiste cross-task é o encoder.
2. **Encoder aprende métrica genérica de Omniglot.** Mapear caracteres similares próximos no espaço de embedding é propriedade global do dataset, não específica de task. Treinar em tasks específicas pode refinar mas não destruir totalmente.
3. **Tasks são "novas classes do mesmo dataset".** Mesma família visual (caracteres). Features transferem. Diferente de continual learning cross-domain (e.g., MNIST → CIFAR), onde forgetting é dramático.

A consequência prática é que para Omniglot specificamente, naive ProtoNet sequential já é um baseline forte, deixando margem pequena para métodos mais elaborados demonstrarem ganho.

## A.4 Implicação para o main paper

Para o main paper sobre sparsity (este documento), Marco 1 confirma que ProtoNet metric learning tem propriedades intrínsecas de robustez que vão além apenas da preservação sob k-WTA esparso. Isto fortalece a interpretação em Section 5.1 (Discussion) de que CNN-4 + ReLU + MaxPool já produz uma representação com propriedades favoráveis (esparsidade natural, generalização entre classes da mesma família visual), e que k-WTA explícito reforça essas propriedades sem introduzir descontinuidade abrupta.

A combinação que mais aproximaria o objetivo "post-LLM cognitive architectures" — k-WTA esparso (sparsity bio-inspirada) + plasticidade local meta-aprendida (Hebbian-like, sem backprop) + continual setup (online learning) — não foi bem-sucedida em Omniglot conforme dados de Caminho 5e acima. Isso fica como observação empírica negativa documentada, com implicação prática: para combinar esses três princípios de forma produtiva, datasets mais difíceis ou arquiteturas com mais capacity podem ser necessários.

## A.5 Documentação completa

Detalhes das 9 sessões de exploração de Marco 1 (#21-#29) estão em `experiment_02_continual/WEEKLY-1.md` no repositório do projeto. Setup específico de cada abordagem em scripts dedicados (`baseline_naive.py`, `c2_continual_arch_b.py`, `c5e_combined.py`).
