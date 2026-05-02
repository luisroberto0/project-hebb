# Section 3: Method

> Status: drafted sessão #59.
> Word count target: 500-700 words. Atual: ~640.

---

\section{Method}
\label{sec:method}

\subsection{Architecture}
\label{subsec:arch}

We use the standard CNN-4 ProtoEncoder of \citet{snell2017prototypical}: four sequential Conv-BN-ReLU-MaxPool blocks with 64 filters and $3 \times 3$ kernels, padding 1. For an Omniglot input $(1, 28, 28)$, four max-poolings of $2 \times 2$ reduce the spatial output to $1 \times 1$, and we flatten into a 64-dimensional embedding $f_\theta(x) \in \mathbb{R}^{64}$. The k-WTA operation is applied to this embedding; classification then proceeds as in standard Prototypical Networks. The architecture is identical to the C3 encoder of \citet{pinho2026kwta}, enabling direct cross-domain transfer of source-trained weights.

\subsection{k-WTA Layer}
\label{subsec:kwta}

The $k$-winner-take-all operation~\citep{maass2000computational} retains the $k$ largest entries of an activation vector and zeros the rest:
\begin{equation}
\text{kWTA}_k(z)_i = \begin{cases} z_i & \text{if } i \in \arg\text{top-}k(z) \\ 0 & \text{otherwise} \end{cases}
\label{eq:kwta}
\end{equation}
The operation is applied per-example (not over batches), and the gradient flows through the $k$ active channels via standard backpropagation. We apply k-WTA at both training and evaluation time. When $k \geq \dim(z) = 64$, the operation is a no-op and the encoder is equivalent to the vanilla ProtoNet baseline; we use this as the $k=64$ control.

\subsection{Source Training: Omniglot}
\label{subsec:source}

We train each $(k$-WTA configuration, seed$)$ pair episodically on the Omniglot~\citep{lake2015human} background split (964 characters from 30 alphabets). Each episode samples $N=5$ classes, $K=1$ support example, and $Q=5$ query examples. Training uses cross-entropy on query-to-prototype squared Euclidean distances, optimized with Adam~\citep{kingma2015adam} (learning rate $10^{-3}$, no scheduling, no regularization) for 5000 episodes, following the protocol of \citet{pinho2026kwta}. Sparsity sweep covers $k \in \{8, 16, 32, 64\}$.

\subsection{Cross-Domain Protocol}
\label{subsec:xdomain}

Our primary protocol resizes all images, regardless of source, to $28 \times 28$ and converts them to grayscale, preserving the source-trained encoders' input shape $(1, 28, 28)$. The cost of this choice is severe: $500 \times 500$ RGB photographs reduced to $28 \times 28$ grayscale lose nearly all texture and color information that conventionally distinguishes bird species. We report the impact of this resolution choice in Section~\ref{sec:experiments} via a parallel $84 \times 84$ RGB retrained baseline. The visual distance between Omniglot and CUB-200 is large in any reasonable embedding metric, and we treat the pair as an instance of the ``extreme task differences'' regime~\citep{phoo2021self}.

After source training, encoder weights are frozen (\texttt{requires\_grad=False}) and evaluated on the CUB-200 test split (5794 images, 200 classes, official \texttt{train\_test\_split.txt}). Each (encoder, seed) pair runs 1000 episodes 5-way 1-shot 5-query, sampling deterministically from the test split via per-seed random number generators.

\subsection{Sparsity Sweep}
\label{subsec:sweep}

The four-point sweep ($k \in \{8, 16, 32, 64\}$, sparsity $\{87.5\%, 75\%, 50\%, 0\%\}$) requires 20 source-trained checkpoints across 5 seeds. The $k=64$ configuration, being a no-op on a 64-D embedding, has gradient flow identical to the vanilla ProtoNet baseline. We materialize $k=64$ checkpoints by copying the trained ProtoNet baseline state dictionary with key prefix adjustment (the sparse encoder wraps the base encoder under \texttt{self.encoder.}, so each key is renamed accordingly). Sanity verification confirms that $k=64$ cross-domain accuracy reproduces the ProtoNet baseline result exactly.

\subsection{Sanity Floors}
\label{subsec:floors}

Two control conditions probe the source of the residual cross-domain signal. \textbf{Pixel kNN}: nearest-neighbor classification over raw pixels, with each $(1, 28, 28)$ image flattened to $\mathbb{R}^{784}$ and class prototypes computed as per-class pixel means. No encoder, no learning. \textbf{Random encoder + k-WTA}: a $\text{ProtoEncoderSparse}(k=16)$ initialized with PyTorch's default Kaiming-uniform scheme, weights frozen, no training. Different seeds produce different random encoders.

\subsection{Re-trained Baseline}
\label{subsec:retrained}

To upper-bound what is achievable on this benchmark with target-domain training, we retrain ProtoNet directly on the CUB-200 train split (5994 images, 200 classes) at two resolutions. \textbf{$28 \times 28$ grayscale}: identical CNN-4 architecture as the source-trained encoders; same hyperparameters (5000 episodes, Adam $10^{-3}$). \textbf{$84 \times 84$ RGB}: a CNN-4 variant in which the first layer accepts three input channels and a final \texttt{AdaptiveAvgPool2d}((1,1)) collapses the $5 \times 5$ feature map to a 64-D embedding consistent with the source-trained models. We train 5 seeds per resolution.

\subsection{Evaluation}
\label{subsec:eval}

For each condition, we report mean accuracy across 5 seeds with two confidence intervals: the inter-seed mean $\pm$ standard deviation, and a 95\% bootstrap CI computed by resampling the five per-seed means 1000 times with replacement. Per-seed numbers also include 95\% bootstrap CIs over the 1000 episodes.

---

## Notas pra #62 (slim revision)

- Word count atual ~640, dentro do target 500-700
- Verificar se §3.4 + §3.5 não soam redundantes (overlap em "k-WTA aplicada")
- Decidir se mantém detalhe sobre prefix adjustment em §3.5 (nicho — pode mover pra footnote ou appendix)
- Confirmar que `\citet{pinho2026kwta}` vs `\citep{pinho2026kwta}` está consistente com convenção do paper C3 (use \citet quando subject da frase)

## Anotações arquivadas (movidas pra cá em #58.5, agora incorporadas)

Conteúdo de §2.4 background original foi movido em #58.5 e agora incorporado em §3.4 (resize 28x28 grayscale + extreme task differences framing). Ajuste 5 da #58.5 fechado.
