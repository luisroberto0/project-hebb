# Section 3: Method

> Status: slim revisado sessão #35. Word count: ~750 palavras (target 700-1000).
> Tom: técnico, reproduzível, sem opinião.

---

## 3.1 Architecture

A arquitetura C3 é uma extensão minimalista do Prototypical Network padrão de Snell et al. [snell2017prototypical], adicionando uma única camada k-WTA esparsa após o embedding final do encoder. Mantemos todos os outros componentes idênticos ao baseline para isolar o efeito da esparsidade.

O encoder é a CNN-4 padrão: quatro blocos sequenciais, cada um com convolução 3×3 com 64 filtros (padding 1), batch normalization, ReLU, e max-pooling 2×2. Para input Omniglot 28×28, o output é um tensor 64×1×1, achatado em embedding $f_\theta(x) \in \mathbb{R}^{64}$.

Aplicamos k-WTA ao embedding produzindo $z = \text{kWTA}_k(f_\theta(x))$. A classificação procede como na ProtoNet padrão: para cada classe $c$ no support set computamos o protótipo $\mathbf{p}_c = \frac{1}{K} \sum_{x_i \in \mathcal{S}_c} z_i$, e classificamos cada query $x_q$ pela classe de protótipo mais próximo em distância euclidiana ao quadrado.

## 3.2 k-WTA layer

A operação winner-take-all com $k$ vencedores [maass2000computational] mantém as $k$ entradas com maior valor:

$$\text{kWTA}_k(z)_i = \begin{cases} z_i & \text{se } i \in \arg\text{top-}k(z) \\ 0 & \text{caso contrário} \end{cases}$$

A operação é aplicada por exemplo (não em batch global), produzindo padrões esparsos individuais. Em PyTorch, a implementação usa `torch.topk(z, k)` seguido de máscara binária por multiplicação elementar. O gradient flui através dos $k$ canais ativos via backpropagation padrão.

Aplicamos k-WTA tanto durante o treino quanto durante avaliação, mantendo a transformação consistente. Isso permite que o encoder $f_\theta$ aprenda a colocar informação discriminativa nas dimensões que tendem a entrar entre os top-$k$.

## 3.3 Training

Seguimos o protocolo episódico padrão. Cada episódio amostra $N=5$ classes, $K=1$ exemplo de support por classe, e $Q=5$ queries por classe. A loss é cross-entropy categórica sobre as distâncias da query aos protótipos:

$$\mathcal{L}(\theta) = -\sum_{x_q \in \mathcal{Q}} \log \frac{\exp(-\|z_q - \mathbf{p}_{y_q}\|^2)}{\sum_{c=1}^{N} \exp(-\|z_q - \mathbf{p}_c\|^2)}$$

Treinamos cada configuração por 5000 episódios totais com Adam [kingma2015adam] (learning rate $10^{-3}$), sem agendamento, sem regularização explícita, sem dropout. As classes amostradas vêm do background set padrão de Omniglot (964 caracteres em 30 alfabetos).

## 3.4 Evaluation

Avaliamos cada configuração em 1000 episódios independentemente amostrados do evaluation set padrão (659 caracteres em 20 alfabetos, disjuntos das classes de treino). Reportamos resultados em 5-way 1-shot e 20-way 1-shot. A acurácia é a fração de queries corretamente classificadas, agregada por episódio.

Para incerteza, computamos intervalos de confiança de 95% via bootstrap percentil sobre as acurácias por episódio (1000 reamostragens com reposição, quantis 2.5% e 97.5%).

Todos os experimentos usam um único seed determinístico (`seed=42`). **Esta é uma limitação reconhecida** — múltiplos seeds dariam ICs mais robustos para a variabilidade total do procedimento, abordada na Seção 5.3.

## 3.5 Sparsity ablation

Testamos três níveis de sparsity:

| Variante | $k$ | Sparsity | Ativações ativas |
|---|---|---|---|
| C3a | 32 | 50% | 32 de 64 |
| C3b | 16 | 75% | 16 de 64 |
| C3c | 8 | 87.5% | 8 de 64 |

A escolha de potências de 2 facilita comparação. Nosso embedding 64-dimensional limita os pontos de operação testáveis em comparação a regimes biológicos típicos de 2-4% sparsity em redes muito maiores [ahmad2019dense].

## 3.6 Random encoder validation

Para distinguir o efeito de treinar sob restrição do efeito da própria estrutura k-WTA, executamos uma validação com encoder não treinado: CNN-4 com inicialização padrão Kaiming, pesos congelados, k-WTA com $k=16$ aplicado, mesmo protocolo de avaliação. Esta condição responde diretamente: quanto da performance C3b vem do treino sob restrição vs. da estrutura k-WTA aplicada a features arbitrárias?
