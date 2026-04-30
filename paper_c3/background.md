# Section 2: Background

> Status: slim revisado sessão #35. Word count: ~700 palavras (target).
> Tom: descritivo, sem opinião. Citar refs corretamente.

---

## 2.1 Prototypical Networks

Prototypical Networks (ProtoNet), introduzidas por Snell et al. [snell2017prototypical], computam um *protótipo* por classe — centróide das embeddings dos exemplos do support set — e classificam queries por proximidade ao protótipo mais próximo no espaço de embedding.

Formalmente, dado um episódio N-way K-shot com support set $\mathcal{S}_c = \{x_i\}_{i=1}^K$ para cada classe $c$, e um encoder $f_\theta$, o protótipo é
$$\mathbf{p}_c = \frac{1}{K} \sum_{x_i \in \mathcal{S}_c} f_\theta(x_i)$$
e a probabilidade da query $x_q$ pertencer à classe $c$ é softmax sobre distâncias euclidianas ao quadrado:
$$P(y=c | x_q) = \frac{\exp(-\|f_\theta(x_q) - \mathbf{p}_c\|^2)}{\sum_{c'} \exp(-\|f_\theta(x_q) - \mathbf{p}_{c'}\|^2)}$$
O encoder é treinado via cross-entropy episodically.

A arquitetura padrão é uma CNN-4: 4 blocos sequenciais Conv-BN-ReLU-MaxPool, cada um com 64 filtros, kernel 3×3, padding 1. Para input Omniglot 28×28, o output após 4 max-poolings 2×2 é 64×1×1, achatado para um embedding 64-dimensional.

## 2.2 k-WTA e codificação esparsa

A regra winner-take-all (WTA) é estudada em redes neurais como mecanismo competitivo [maass2000computational]. A generalização k-WTA mantém as $k$ ativações maiores e zera as demais:

$$\text{kWTA}_k(z)_i = \begin{cases} z_i & \text{se } z_i \in \text{top-}k(z) \\ 0 & \text{caso contrário} \end{cases}$$

Codificação esparsa tem fundamentos em neurociência computacional. Olshausen e Field [olshausen1996emergence] mostraram que aprender bases que reconstroem imagens naturais com restrição de sparsity recupera filtros similares a campos receptivos de células simples no córtex visual. Sparse Distributed Memory [kanerva1988sparse] usa endereçamento esparso para implementar memória associativa. Trabalhos mais recentes em arquiteturas inspiradas em córtex argumentam que sparsity (tipicamente 2-4% de unidades ativas) é crítica para robustez e capacidade [ahmad2019dense].

Em deep learning, conexões com sparse coding aparecem em modelos como Sparse Autoencoders e atenção esparsa, mas k-WTA explícito como restrição arquitetural em métodos prototype-based de few-shot não é prática estabelecida na literatura. Hopfield Networks Modernas [ramsauer2021hopfield] redescobriram conexões entre energy-based memory e attention, mas focam em capacidade de armazenamento, não em sparsity de código durante treino end-to-end.

## 2.3 Omniglot benchmark

Omniglot [lake2015human] consiste de 1623 caracteres distintos coletados de 50 alfabetos, com 20 instâncias por caractere. O dataset foi desenhado como contraponto ao MNIST: enquanto MNIST tem 10 classes com milhares de exemplos cada, Omniglot tem milhares de classes com 20 exemplos cada — invertendo o desequilíbrio típico classes-vs-exemplos.

A divisão padrão usa o "background set" (964 caracteres de 30 alfabetos) para treino e o "evaluation set" (659 caracteres de 20 alfabetos) para avaliação. Episódios N-way K-shot sorteiam N classes do evaluation set, K exemplos como support, e exemplos adicionais como query. O protocolo padrão é 5-way 1-shot e 20-way 1-shot, com chance de 20% e 5% respectivamente. Omniglot tornou-se padrão de fato para few-shot learning pela quantidade de classes, simplicidade visual (28×28 grayscale após preprocessing) e reprodutibilidade do protocolo episódico.

## 2.4 Trabalhos relacionados

Esparsidade em deep learning few-shot tem conexões indiretas: Matching Networks [vinyals2016matching] usa atenção sobre o support set, forma soft de sparsity sobre exemplos; MAML [finn2017model] adapta pesos via gradiente sem restrição arquitetural de sparsity. Trabalhos em deep learning bio-plausível [ahmad2019dense] enfatizam sparsity de código como princípio, tipicamente sem comparação direta com baselines como ProtoNet.

A combinação específica — k-WTA explícito aplicado ao embedding final de ProtoNet, com gradiente fluindo pelos top-$k$ channels durante treino — não foi documentada explicitamente em trabalhos prévios que conhecemos. Este paper preenche essa lacuna empírica.
