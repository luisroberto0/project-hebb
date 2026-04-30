# Section 2: Background

> Status: draft sessão #31. Revisar na sessão #33.
> Word count target: 500-700 palavras.
> Tom: descritivo, sem opinião. Citar refs corretamente.

---

## 2.1 Prototypical Networks

Prototypical Networks (ProtoNet), introduzidas por Snell et al. [snell2017prototypical], abordam few-shot classification computando um *protótipo* por classe — definido como o centróide das embeddings dos exemplos do support set — e classificando queries por proximidade ao protótipo mais próximo no espaço de embedding.

Formalmente, dado um episódio N-way K-shot com support set $\mathcal{S}_c = \{x_i\}_{i=1}^K$ para cada classe $c \in \{1, ..., N\}$, e um encoder $f_\theta$, o protótipo da classe $c$ é
$$\mathbf{p}_c = \frac{1}{K} \sum_{x_i \in \mathcal{S}_c} f_\theta(x_i)$$
e a probabilidade da query $x_q$ pertencer à classe $c$ é dada por softmax sobre distâncias euclidianas quadradas:
$$P(y=c | x_q) = \frac{\exp(-d(f_\theta(x_q), \mathbf{p}_c))}{\sum_{c'} \exp(-d(f_\theta(x_q), \mathbf{p}_{c'}))}$$
com $d(\cdot, \cdot) = \|\cdot - \cdot\|^2$. O encoder $\theta$ é treinado via cross-entropy episodically — cada batch de treino é um episódio amostrado.

A arquitetura padrão usada por Snell et al. e replicada na maioria dos trabalhos subsequentes é uma CNN-4: 4 blocos sequenciais Conv-BN-ReLU-MaxPool, cada um com 64 filtros, kernel 3×3, padding 1. Para input Omniglot 28×28, o output após 4 max-poolings 2×2 é 64×1×1, achatado para um embedding 64-dimensional. Esta escolha simples é suficiente para atingir ~98% acurácia em Omniglot 5-way 1-shot no setup original [snell2017prototypical].

ProtoNet é estado da arte simples e forte para few-shot benchmarks. Variações como Matching Networks [vinyals2016matching] e Model-Agnostic Meta-Learning (MAML) [finn2017model] foram propostas com diferentes mecanismos de inferência, mas ProtoNet permanece como baseline de referência por sua simplicidade conceitual e desempenho competitivo.

## 2.2 k-WTA e codificação esparsa

A regra winner-take-all (WTA) — manter ativa apenas a unidade com maior resposta a um input — é estudada em redes neurais desde décadas como mecanismo competitivo [maass2000computational]. A generalização k-WTA mantém as $k$ ativações maiores e zera as demais, produzindo representações esparsas com nível de sparsity controlado: dado $z \in \mathbb{R}^n$, define-se
$$\text{kWTA}_k(z)_i = \begin{cases} z_i & \text{se } z_i \in \text{top-}k(z) \\ 0 & \text{caso contrário} \end{cases}$$

Codificação esparsa tem fundamentos em neurociência computacional: Olshausen e Field [olshausen1996emergence] mostraram que aprender bases que reconstroem imagens naturais com restrição de sparsity recupera filtros similares a campos receptivos de células simples no córtex visual. Este resultado motivou décadas de trabalho em sparse coding tanto como modelo cortical quanto como ferramenta de processamento de sinais.

Em arquiteturas inspiradas em córtex, esparsidade de código é princípio central. Sparse Distributed Memory (SDM), proposta por Kanerva [kanerva1988sparse], usa endereçamento esparso para implementar memória associativa com propriedades de generalização. Trabalhos mais recentes em hierarchical temporal memory (HTM) e arquiteturas relacionadas argumentam que sparsity (tipicamente 2-4% de unidades ativas) é crítica para robustez e capacidade [ahmad2019dense].

Em deep learning, conexões com sparse coding aparecem em modelos como Sparse Autoencoders e em mecanismos de atenção esparsa, mas k-WTA explícito como restrição arquitetural em métodos prototype-based de few-shot não é prática estabelecida na literatura mainstream. Hopfield Networks Modernas [ramsauer2021hopfield] redescobriram conexões entre energy-based memory e attention, com discussão tangencial de sparsity, mas focam em capacidade exponencial de armazenamento, não em sparsity de código durante treino end-to-end.

## 2.3 Omniglot benchmark

Omniglot, introduzido por Lake et al. [lake2015human], consiste de 1623 caracteres distintos coletados de 50 alfabetos do mundo, com 20 instâncias por caractere desenhadas por colaboradores diferentes via Mechanical Turk. O dataset foi propositadamente desenhado como contraponto ao MNIST: enquanto MNIST tem 10 classes com milhares de exemplos cada, Omniglot tem milhares de classes com 20 exemplos cada — invertendo o desequilíbrio típico classes-vs-exemplos.

A divisão padrão para few-shot evaluation usa o "background set" (964 caracteres de 30 alfabetos) para treino e o "evaluation set" (659 caracteres de 20 alfabetos) para avaliação. Episódios N-way K-shot são amostrados sorteando N classes do evaluation set, K exemplos como support, e exemplos adicionais como query. O protocolo padrão é 5-way 1-shot e 20-way 1-shot, com chance de 20% e 5% respectivamente.

Omniglot tornou-se padrão de fato para few-shot learning por três razões: (i) classes em quantidade suficiente para evitar overfitting trivial; (ii) imagens visualmente simples (28×28 grayscale após preprocessing padrão), permitindo iteração rápida em hardware modesto; (iii) reprodutibilidade — divisões padrão e protocolo episódico bem estabelecidos.

## 2.4 Trabalhos relacionados

Esparsidade em deep learning few-shot tem algumas conexões indiretas: (i) Matching Networks [vinyals2016matching] usa atenção sobre o support set, que pode ser visto como uma forma soft de sparsity sobre exemplos; (ii) MAML [finn2017model] adapta pesos via gradiente sem restrição arquitetural de sparsity. Trabalhos em deep learning bio-plausível [ahmad2019dense; diehl2015unsupervised] enfatizam sparsity de código como princípio, mas tipicamente em contextos onde não há comparação direta com baselines deep modernos como ProtoNet.

A combinação específica — k-WTA explícito aplicado ao embedding final de ProtoNet, com gradiente fluindo pelos top-k channels durante treino — não foi documentada explicitamente em trabalhos prévios que conhecemos. Este paper preenche essa lacuna empírica.

---

## Notas de revisão (pra sessão #33)

- [ ] Verificar word count (atual ~700 palavras, no limite alto do alvo)
- [ ] Conferir notação matemática (LaTeX inline com $...$ — converter pra MathJax/equation no LaTeX final)
- [ ] Decidir se inclui Figura 1 (diagrama ProtoNet vanilla) ou só descreve textualmente
- [ ] Refinar seção 2.4 — pode ficar mais conciso ou mais expansivo dependendo de reviewer feedback
- [ ] Verificar se citação a Diehl & Cook 2015 em 2.4 é necessária ou removível (refere ao Marco 1 do projeto, pode ser só na discussion)
