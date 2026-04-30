# Section 1: Introduction

> Status: draft sessão #31. Revisar na sessão #33.
> Word count target: 500-700 palavras.
> Tom: honesto, sem overclaim. Workshop paper, não conference.

---

Sparse coding é uma característica observada na computação cortical: somente uma fração pequena de neurônios responde ativamente a cada input, e essa esparsidade aparenta ter papel funcional na eficiência energética e na separação de representações [olshausen1996emergence; ahmad2019dense]. Em deep learning mainstream, contudo, representações aprendidas tendem a ser densas — ativações pós-ReLU costumam ter densidade de 30-50% e nenhuma restrição arquitetural força esparsidade adicional. A pergunta natural é: até que ponto métodos modernos toleram restrições explícitas de sparsity sem perda de performance?

Few-shot learning via Prototypical Networks (ProtoNet) [snell2017prototypical] tornou-se um dos baselines mais usados em metric learning. Sua simplicidade — encoder convolucional + classificação por distância ao centróide — viabiliza experimentação controlada. Variants e extensions de ProtoNet são extensivamente estudadas, incluindo Matching Networks [vinyals2016matching] e MAML [finn2017model], mas restrições de esparsidade explícita não fazem parte do design padrão. Por outro lado, mecanismos de winner-take-all (k-WTA) e codificação esparsa têm tradição estabelecida em outros contextos: redes de Hopfield modernas [ramsauer2021hopfield], Sparse Distributed Memory [kanerva1988sparse], e arquiteturas inspiradas em córtex como HTM [ahmad2019dense]. A interseção — aplicar k-WTA explícito a métodos prototype-based em few-shot — recebeu pouca atenção empírica direta.

Este paper apresenta um estudo empírico controlado dessa interseção. Aplicamos k-WTA esparso ao embedding final de uma ProtoNet padrão (CNN-4 de Snell et al. [snell2017prototypical]), treinada end-to-end em Omniglot [lake2015human]. Variamos o nível de sparsity em três pontos (50%, 75%, 87.5%) e comparamos contra o baseline ProtoNet completo. Para isolar a contribuição do treino sob restrição de sparsity (vs. a estrutura k-WTA estática), incluímos uma validação obrigatória usando encoder com pesos aleatórios + k-WTA aplicado, sem treino.

**Resultado principal:** k-WTA com 75% das ativações zeradas (k=16 de 64 dimensões) preserva 93.10% acurácia em Omniglot 5-way 1-shot, dentro de −1.45 p.p. do baseline ProtoNet completo (94.55%). Mesmo em 87.5% sparsity (k=8) a performance permanece em 90.77%. A validação random + k-WTA fica em 37.60% — confirmando que o ganho vem do treino sob restrição, não da estrutura k-WTA aplicada a features arbitrárias.

A contribuição deste trabalho é deliberadamente focada e empírica:

1. **Quantificamos** a tolerância a sparsity de ProtoNet em Omniglot via 3 pontos de operação (50%/75%/87.5%) com IC95% bootstrap sobre 1000 episódios de avaliação.
2. **Demonstramos** que codificação esparsa neural-inspirada é compatível com prototype-based metric learning, sem perda significativa de performance até 75% sparsity.
3. **Validamos** via baseline com encoder random que o resultado depende da otimização sob a restrição, não da k-WTA estrutural sozinha.
4. **Discutimos** implicações pra deep learning bio-plausível e pra continual learning (apêndice opcional sobre testes adicionais em setup de continual learning).

Não claim originalidade arquitetural — k-WTA é técnica clássica e ProtoNet é amplamente estudada. A originalidade está na caracterização empírica explícita: até onde sabemos, não há estudo prévio quantificando sistematicamente quanto sparsity ProtoNet tolera em few-shot benchmarks. Este resultado serve como evidência adicional de que princípios de codificação esparsa biológica são compatíveis com deep representation learning, complementando trabalho prévio em redes de Hopfield modernas e SDM.

**Roadmap do paper.** Seção 2 revisa background sobre Prototypical Networks, k-WTA, e o benchmark Omniglot. Seção 3 descreve a arquitetura C3 (ProtoNet + k-WTA) e o protocolo experimental. Seção 4 apresenta resultados em 5-way e 20-way 1-shot, a curva sparsity × accuracy, e a validação com encoder random. Seção 5 discute mecanismos prováveis pela qual k-WTA não derruba ProtoNet, limitações do estudo, e direções futuras. Seção 6 conclui. Apêndice opcional documenta exploração paralela em continual learning.

---

## Notas de revisão (pra sessão #33)

- [ ] Verificar word count (atual ~600 palavras, dentro do alvo)
- [ ] Conferir se claims estão consistentes com Tabela 1 (a escrever em #32)
- [ ] Confirmar refs.bib tem todas as citações em colchetes
- [ ] Considerar se "deliberadamente focada" soa defensivo demais — talvez "focused empirical study" mais direto
- [ ] Decidir se mantém menção ao apêndice continual learning na intro ou só na discussion
