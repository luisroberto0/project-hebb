# Section 5: Discussion

> Status: draft sessão #34. Revisar na sessão #35 (revisão geral).
> Word count target: 600-900 palavras. Atual: ~750.
> Tom: honesto sobre limitações, modesto sobre claims, técnico sobre mecanismo.

---

## 5.1 Por que k-WTA preserva ProtoNet performance

A robustez observada — sparsity de 75% custa apenas −1.45 p.p. em acurácia — não é necessariamente surpreendente quando consideramos a estrutura interna do encoder CNN-4. As ativações pré-k-WTA já são moderadamente esparsas por construção: ReLU zera valores negativos (tipicamente ~50% dos pré-ativações em camadas convolucionais bem treinadas), e MaxPool seleciona o máximo dentro de cada janela 2×2 espacial, descartando os outros três valores. Após quatro blocos com essas operações, o embedding final já tem distribuição com cauda longa em valores baixos próximos a zero.

A operação k-WTA aplicada ao embedding 64D, portanto, **reforça** uma esparsidade que já existe parcialmente, em vez de introduzir descontinuidade abrupta. O encoder treinado sob essa restrição aprende — via gradient flow pelos top-$k$ channels selecionados em cada exemplo — a concentrar informação discriminativa nas dimensões com maior magnitude esperada. Isso é coerente com a hipótese de Ahmad e Scheinkman [ahmad2019dense] de que representações esparsas oferecem benefícios de robustez e separabilidade quando o aprendizado é otimizado para elas, e não apenas avaliado sob a restrição.

Adicionalmente, a métrica de classificação — distância euclidiana ao quadrado entre embeddings — degrada graciosamente sob esparsidade pareada. Quando ambos vetores comparados são $k$-esparsos com k-WTA aplicado de forma consistente, a contribuição dimensional pra `cdist²` vem dominantemente das poucas dimensões ativas, e a discriminação entre classes próximas no espaço de embedding pode ser preservada se essas dimensões são informativas.

Vale notar que a robustez observada em 20-way 1-shot (75-82% para sparsities testadas) é menor em magnitude absoluta mas consistente em padrão com 5-way 1-shot. Em N-way maior, o problema é mais difícil mas a sensibilidade à sparsity não amplifica desproporcionalmente — o gap entre C3b (75% sparsity) e C3c (87.5% sparsity) é 5.28 p.p. em 20w1s vs 2.33 p.p. em 5w1s, refletindo dificuldade absoluta sem sugerir colapso.

## 5.2 Implicações para bio-plausible learning

Sparsity é frequentemente discutida como princípio biológico organizador, com referências a estimativas de 2-4% de unidades ativas em córtex [ahmad2019dense]. Os níveis testados neste paper (12.5% ativações em C3c, 25% em C3b) são consideravelmente menos agressivos que essas estimativas biológicas, em grande parte por limitação dimensional do embedding (64D não permite níveis de sparsity próximos aos biológicos sem reduzir capacidade representacional).

Apesar dessa diferença de magnitude, o resultado central — que sparsity explícita não impede aprendizado discriminativo via gradient — é empiricamente compatível com a tese mais ampla de que codificação esparsa pode ser implementada em arquiteturas deep modernas sem custo proibitivo de performance. Não estamos claiming que k-WTA aplicada ao embedding de uma CNN-4 é "biologicamente realista" — claramente não é. Mas a viabilidade de combinar princípios bio-inspirados (sparse coding) com técnicas mainstream (deep metric learning) é demonstrada empiricamente, fornecendo um ponto de comparação para trabalhos futuros que busquem maior fidelidade biológica.

## 5.3 Limitações

Várias limitações concretas demarcam o escopo deste estudo:

- **Single dataset.** Todos os experimentos usam Omniglot. Embora este seja o benchmark padrão para few-shot learning, sua simplicidade visual (caracteres binários grayscale) pode não generalizar para domínios com texturas e iluminação realistas, como Mini-ImageNet ou ImageNet.
- **Single seed.** Resultados reportados usam apenas `seed=42`. Os IC95% bootstrap capturam variabilidade da amostragem episódica de avaliação, mas não da inicialização do encoder ou da ordem episódica de treino. Múltiplos seeds dariam intervalos mais robustos.
- **k-WTA aplicada apenas no embedding final.** Não exploramos esparsidade em camadas intermediárias, onde poderia interagir diferentemente com aprendizado de features hierárquicas.
- **Sweep limitado a 3 pontos.** Os níveis testados (50%, 75%, 87.5%) deixam espaço pra interpolação. O regime entre 87.5% e 100% (toda informação em uma única dimensão) não é explorado e pode revelar comportamento qualitativamente diferente.
- **Backprop end-to-end ainda usado.** A operação k-WTA é aplicada na forward pass mas o treino usa SGD padrão. Variantes que combinem k-WTA com regras de plasticidade local (ex: Hebbian, STDP) ficam fora do escopo deste paper.
- **Comparação com baselines da era 2017.** ProtoNet permanece referência mas arquiteturas mais recentes (ResNet encoders, transformers para few-shot) atingem accuracies mais altas em Omniglot. Não posicionamos este trabalho como state-of-the-art em acurácia bruta — o foco é caracterização da relação sparsity-accuracy num backbone amplamente estudado.

## 5.4 Trabalho futuro

Algumas extensões diretas:

1. **Sweep mais fino entre $k=8$ e $k=16$** ($k \in \{4, 6, 10, 12\}$) pra caracterizar precisamente o threshold onde a degradação acelera.
2. **Múltiplos seeds (5-10)** pra obter variabilidade de treino completa.
3. **k-WTA em camadas intermediárias** — aplicar a saídas de cada bloco Conv-BN-ReLU-MaxPool, observar interação com aprendizado de features hierárquicas.
4. **Datasets mais difíceis** — Mini-ImageNet ou tieredImageNet para testar generalização da relação sparsity-accuracy fora de Omniglot.
5. **Combinação com plasticidade local meta-aprendida.** Em exploração paralela vinculada a este projeto (ver discussão em 5.5), testamos plasticidade meta-aprendida em continual learning. Combinar k-WTA esparso com regras de plasticidade local poderia abordar simultaneamente codificação esparsa e online learning.

## 5.5 Observação relacionada: ProtoNet em continual learning

Em exploração paralela conduzida no mesmo projeto, testamos ProtoNet em setting de continual learning sequencial sem replay (treino sequencial em 50 grupos de classes, sem buffer de exemplos antigos). Observamos que o naive baseline ProtoNet — sem nenhuma defesa contra catastrophic forgetting — atinge ~80% de average accuracy em 50 tasks, com forgetting moderado (BWT ≈ −9 p.p.). Esta robustez parece intrínseca ao mecanismo prototype-based: como prototypes são computados FRESH a cada episódio do support set, não há classifier head treinado pra "esquecer". Mecanismos bio-inspirados que testamos (plasticidade local meta-aprendida, k-WTA em arquitetura combinada) não superaram este baseline em nossas condições experimentais. Documentação completa em material suplementar disponível mediante solicitação.

---

## Notas de revisão (pra sessão #35)

- [ ] Verificar word count (atual ~750, target 600-900) ✓
- [ ] Conferir se 5.5 é apropriado em Discussion ou se move pra Future Work
- [ ] Considerar se 5.1 deve ter sub-figura ilustrando ativações pré vs pós k-WTA (TBD #35)
- [ ] Refinar tom de 5.2 — pode soar defensivo, considerar reformular
- [ ] Verificar consistência de "C3a/b/c" notation ao longo do paper
