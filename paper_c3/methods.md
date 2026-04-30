# Section 3: Method

> Status: draft sessão #33. Revisar na sessão #34 (revisão geral antes de LaTeX).
> Word count target: 700-1000 palavras. Atual: ~850.
> Tom: técnico, reproduzível, sem opinião.

---

## 3.1 Architecture

A arquitetura C3 é uma extensão minimalista do Prototypical Network padrão de Snell et al. [snell2017prototypical], adicionando uma única camada k-WTA esparsa imediatamente após o embedding final do encoder. Mantemos todos os outros componentes idênticos ao baseline pra isolar o efeito da esparsidade.

O encoder é a CNN-4 padrão de Snell et al.: quatro blocos sequenciais, cada um composto por convolução 3×3 com 64 filtros (padding 1), batch normalization, ativação ReLU, e max-pooling 2×2. Para input Omniglot 28×28 (downsampled de 105×105 e com fundo invertido para traços brancos sobre preto, seguindo o protocolo padrão), o output após os quatro max-poolings é um tensor 64×1×1, achatado em um vetor de embedding 64-dimensional $f_\theta(x) \in \mathbb{R}^{64}$.

Aplicamos a operação k-WTA ao embedding $f_\theta(x)$ produzindo embedding esparso $z = \text{kWTA}_k(f_\theta(x)) \in \mathbb{R}^{64}$. A classificação procede como na ProtoNet padrão: para cada classe $c$ no support set, computamos o protótipo $\mathbf{p}_c = \frac{1}{K} \sum_{x_i \in \mathcal{S}_c} z_i$, e classificamos cada query $x_q$ pela classe de protótipo mais próximo em distância euclidiana ao quadrado.

## 3.2 k-WTA layer

A operação winner-take-all com $k$ vencedores [maass2000computational] mantém as $k$ entradas com maior valor de um vetor e zera as demais. Formalmente, dado $z \in \mathbb{R}^n$ e um conjunto de índices $I_k(z) = \arg\text{top-}k(z)$ (os $k$ índices com maiores valores), definimos:

$$\text{kWTA}_k(z)_i = \begin{cases} z_i & \text{se } i \in I_k(z) \\ 0 & \text{caso contrário} \end{cases}$$

A operação é aplicada por exemplo (não em batch global), de forma que cada amostra produz seu próprio padrão esparso. Em PyTorch, a implementação usa `torch.topk(z, k)` para obter os índices, seguido de uma máscara binária aplicada por multiplicação elementar. O gradient flui através dos $k$ canais ativos via backpropagation padrão; canais zerados recebem gradient zero.

Aplicamos k-WTA tanto durante o treino quanto durante avaliação, mantendo a mesma transformação consistente nas duas fases. Isso permite que o encoder $f_\theta$ aprenda a colocar informação discriminativa nas dimensões que tendem a ficar entre os top-$k$ ativações.

## 3.3 Training

Seguimos o protocolo episódico padrão de Prototypical Networks. Cada episódio de treino amostra aleatoriamente $N=5$ classes do training set, $K=1$ exemplo de support por classe, e $Q=5$ exemplos de query por classe. A loss por episódio é cross-entropy categórica computada sobre as distâncias da query aos protótipos:

$$\mathcal{L}(\theta) = -\sum_{x_q \in \mathcal{Q}} \log \frac{\exp(-\|z_q - \mathbf{p}_{y_q}\|^2)}{\sum_{c=1}^{N} \exp(-\|z_q - \mathbf{p}_c\|^2)}$$

onde $z_q$ é o embedding esparso da query (após k-WTA), $\mathbf{p}_c$ é o protótipo da classe $c$ computado das embeddings esparsas do support, e $y_q$ é o rótulo verdadeiro de $x_q$.

Treinamos cada configuração por 5000 episódios totais usando o otimizador Adam [kingma2015adam] com learning rate $10^{-3}$, sem agendamento de learning rate, sem regularização explícita (weight decay = 0), e sem dropout. As classes amostradas em cada episódio vêm do background set padrão de Omniglot (964 caracteres em 30 alfabetos).

## 3.4 Evaluation

Avaliamos cada configuração em 1000 episódios independentemente amostrados do evaluation set padrão de Omniglot (659 caracteres em 20 alfabetos, disjuntos das classes vistas durante o treino). Reportamos resultados em duas condições padrão da literatura few-shot: 5-way 1-shot (5 classes, 1 support por classe, 5 queries por classe) e 20-way 1-shot (20 classes, 1 support por classe, 5 queries por classe). A acurácia é a fração de queries corretamente classificadas, calculada por episódio e depois agregada.

Para quantificar incerteza, computamos intervalos de confiança de 95% via bootstrap percentil sobre as acurácias por episódio: amostramos 1000 reamostragens bootstrap de tamanho 1000 (com reposição) a partir das 1000 acurácias por-episódio observadas, computamos a média de cada amostra, e reportamos os quantis 2.5% e 97.5% como limites do IC95%.

Todos os experimentos usam um único seed determinístico (`seed=42`) tanto para inicialização de pesos do encoder quanto para amostragem de episódios. Isso é uma limitação do estudo — múltiplos seeds dariam ICs mais robustos para a variabilidade total do procedimento, e abordamos isso na discussão de limitações (Seção 5.3).

## 3.5 Sparsity ablation

Testamos três níveis de sparsity, correspondentes a três valores de $k$ aplicados ao embedding 64-dimensional:

| Variante | $k$ | Sparsity | Ativações ativas |
|---|---|---|---|
| C3a | 32 | 50% | 32 de 64 |
| C3b | 16 | 75% | 16 de 64 |
| C3c | 8 | 87.5% | 8 de 64 |

A escolha de potências de 2 facilita comparação e segue convenção comum na literatura de sparse coding (e.g., [ahmad2019dense] discute regimes de 2-4% de unidades ativas como faixa biológica típica, mas em redes muito maiores; nosso embedding 64D limita os pontos de operação testáveis).

## 3.6 Random encoder validation

Para distinguir o efeito de treinar sob restrição de sparsity do efeito da própria estrutura k-WTA, executamos uma validação obrigatória usando um encoder com pesos aleatórios não treinados. Especificamente, instanciamos a CNN-4 com inicialização padrão Kaiming, mantemos os pesos congelados (sem treino), aplicamos k-WTA com $k=16$ (75% sparsity) ao embedding produzido, e avaliamos no mesmo protocolo (1000 episódios, 5-way 1-shot e 20-way 1-shot, mesmo seed).

Esta condição responde diretamente: "quanto da performance C3b vem do treino sob restrição vs. da estrutura k-WTA aplicada a features arbitrárias?" Se random encoder + k-WTA atinge accuracy comparável a C3b treinado, a contribuição da otimização sob sparsity é nula. Se atinge accuracy próxima da chance (20% em 5-way, 5% em 20-way), o resultado C3b é atribuível primariamente ao treino, não à estrutura.

---

## Notas de revisão (pra sessão #34)

- [ ] Verificar word count (atual ~850, target 700-1000) ✓
- [ ] Conferir notação matemática consistente (LaTeX inline `$...$` — converter pra MathJax/equation block na conversão LaTeX final)
- [ ] Adicionar ref Kingma & Ba 2014 (Adam) ao refs.bib se ainda não está
- [ ] Conferir que descrição de bootstrap CI bate com implementação real em c3_protonet_sparse.py
- [ ] Decidir se Tabela em 3.5 (sparsity ablation) fica em Methods ou move pra Experiments
- [ ] Considerar se 3.6 (random validation) deveria estar em Experiments em vez de Methods — argumentável
