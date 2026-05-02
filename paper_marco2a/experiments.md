# Section 4: Experiments

> Status: drafted sessão #59.
> Word count target: 800-1000 words. Atual: ~890.

---

\section{Experiments}
\label{sec:experiments}

\subsection{Setup}
\label{subsec:setup}

All experiments run on a single workstation with an Intel Core i9 CPU and an NVIDIA RTX 4070 Laptop GPU (compute capability 8.9, 8\,GB VRAM), using PyTorch 2.6 with CUDA 12.4. Source training is approximately 70 seconds per (sparsity, seed) checkpoint; the 20 source-trained checkpoints plus 10 retrained-on-CUB checkpoints total roughly 30 minutes of GPU time. Each evaluation run (1000 episodes for one configuration on one seed) takes between 2 and 6 seconds depending on the encoder. CUB images are pre-processed once and cached on disk: 37\,MB for the $28 \times 28$ grayscale tensor and 280\,MB for the $84 \times 84$ RGB tensor.

\subsection{Cross-Domain Results}
\label{subsec:xdomain-results}

Table~\ref{tab:main} reports 5-way 1-shot accuracy on the CUB-200 test split for the seven characterized conditions, with a chance baseline of 20.00\%.

\begin{table}[h]
\centering
\small
\caption{Cross-domain 5-way 1-shot accuracy on CUB-200 test split, 5 seeds $\times$ 1000 episodes each. ``Inter-seed std'' is the standard deviation of the five per-seed means. CIs are 95\% bootstrap intervals on the per-seed means. ``Frozen'' means the encoder weights are not updated for CUB.}
\label{tab:main}
\begin{tabular}{lrrrr}
\toprule
Model & Input shape & Mean ACC & Inter-seed std & 95\% CI \\
\midrule
ProtoNet retrained on CUB ($84 \times 84$ RGB)            & $(3, 84, 84)$  & 49.84\% & 0.82\% & [49.38, 50.59] \\
ProtoNet retrained on CUB ($28 \times 28$ gray)           & $(1, 28, 28)$  & 34.31\% & 0.31\% & [34.06, 34.55] \\
Pixel kNN cross-domain                                    & $(1, 28, 28)$ raw & 22.81\% & 0.18\% & [22.69, 22.97] \\
C3 $k=32$ (50\% sparse, Omniglot frozen)                  & $(1, 28, 28)$  & 22.20\% & 0.53\% & [21.77, 22.57] \\
C3 $k=64$ (vanilla, Omniglot frozen)                      & $(1, 28, 28)$  & 22.13\% & 0.30\% & [21.90, 22.36] \\
C3 $k=16$ (75\% sparse, Omniglot frozen)                  & $(1, 28, 28)$  & 22.09\% & 0.32\% & [21.84, 22.34] \\
Random encoder + k-WTA $k=16$                             & $(1, 28, 28)$  & 21.91\% & 0.17\% & [21.76, 22.03] \\
C3 $k=8$ (87.5\% sparse, Omniglot frozen)                 & $(1, 28, 28)$  & 21.68\% & 0.44\% & [21.34, 22.04] \\
\midrule
Chance                                                    & ---            & 20.00\% & ---    & --- \\
\bottomrule
\end{tabular}
\end{table}

The five conditions using the $(1, 28, 28)$ input through some form of CNN-4 forward pass (the four source-trained sparsities plus the random-encoder control) cluster tightly between 21.68\% and 22.20\%, with overlapping 95\% CIs. Pixel kNN, with no encoder at all, reaches 22.81\%, with a CI that does not overlap any of the encoder-based conditions. ProtoNet retrained directly on CUB at the same $28 \times 28$ grayscale resolution reaches 34.31\%, an absolute improvement of $+12.22$ percentage points over the best source-trained encoder ($k=32$) under the same input pipeline. Increasing the input to $84 \times 84$ RGB raises the retrained baseline to 49.84\%, a further $+15.53$ p.p.

\subsection{In-Domain vs.\ Cross-Domain Effect}
\label{subsec:effect-collapse}

Table~\ref{tab:effect-collapse} compares the same four sparsity levels on Omniglot (in-domain numbers from \citet{pinho2026kwta}) and on CUB-200 cross-domain. The drop in accuracy from in-domain to cross-domain is approximately 70 percentage points across all $k$. Critically, the spread \emph{across} sparsity levels collapses from 3.78 p.p.\ in-domain to 0.52 p.p.\ cross-domain.

\begin{table}[h]
\centering
\small
\caption{In-domain (Omniglot) vs.\ cross-domain (CUB-200) 5-way 1-shot accuracy at four sparsity levels. In-domain numbers from \citet{pinho2026kwta}; cross-domain from this work. The bottom row reports the spread (max $-$ min) across the four $k$ values within each domain.}
\label{tab:effect-collapse}
\begin{tabular}{rrrrr}
\toprule
$k$ & Sparsity & Omniglot 5w1s & CUB 5w1s & $\Delta$ in $\to$ cross \\
\midrule
8   & 87.5\%        & 90.77\%         & 21.68\%       & $-69.09$ p.p. \\
16  & 75\%          & 93.10\%         & 22.09\%       & $-71.01$ p.p. \\
32  & 50\%          & 93.35\%         & 22.20\%       & $-71.15$ p.p. \\
64  & 0\% (vanilla) & 94.55\%         & 22.13\%       & $-72.42$ p.p. \\
\midrule
\multicolumn{2}{l}{Spread $k=8$ vs $k=64$} & 3.78 p.p. & 0.52 p.p. & --- \\
\bottomrule
\end{tabular}
\end{table}

The in-domain spread of 3.78 p.p.\ between $k=8$ and $k=64$ is the empirical signature of k-WTA's effect on ProtoNet documented in \citet{pinho2026kwta}: the cost of imposing 87.5\% sparsity is a measurable accuracy drop of approximately 4 p.p. relative to the unconstrained encoder. Cross-domain, the analogous spread is 0.52 p.p., entirely contained within the overlap of per-condition 95\% confidence intervals. The effect that mattered in-domain becomes statistical noise under transfer.

\subsection{Bottleneck Decomposition}
\label{subsec:bottleneck}

Successive comparisons isolate where the gap between random-weight and high-performance conditions originates:

\begin{itemize}
    \item \textbf{Random encoder} (21.91\%) $\to$ \textbf{$k=16$ source-trained} (22.09\%): $+0.18$ p.p. The contribution of training on Omniglot, holding the input pipeline fixed, is statistically negligible.
    \item \textbf{$k=16$ source-trained} (22.09\%) $\to$ \textbf{ProtoNet retrained on CUB at $28 \times 28$} (34.31\%): $+12.22$ p.p. With the same input pipeline, switching the training distribution from source to target accounts for over twelve percentage points.
    \item \textbf{Retrained at $28 \times 28$} (34.31\%) $\to$ \textbf{retrained at $84 \times 84$ RGB} (49.84\%): $+15.53$ p.p. With training on the target held fixed, increasing input resolution and adding color contributes another fifteen-and-a-half points.
\end{itemize}

The two large contributors---target-domain training and adequate input fidelity---are of comparable magnitude and roughly additive. Sparsity contributes none.

\subsection{Anti-Transfer Evidence}
\label{subsec:anti-transfer}

The 0.18 p.p.\ gap between random and source-trained encoders is statistically indistinguishable from zero: the 95\% CI of the random encoder ([21.76, 22.03]) is contained within the CI of the source-trained $k=16$ encoder ([21.84, 22.34]). Five thousand episodes of episodic training on Omniglot produce no detectable transferable signal in this regime. We refer to this pattern as \emph{anti-transfer}: an encoder trained on a sufficiently distant source is at best equivalent to an untrained one, and---as the next subsection shows---may underperform a representation that involves no encoder at all. The pattern is consistent with the diagnoses of \citet{phoo2021self}: source-domain training, however thorough, is insufficient under extreme task differences, and self-training on the target domain is required to bridge the gap.

\subsection{Pixel kNN Dominates Encoded Representations}
\label{subsec:pixel-dominates}

Pixel kNN (22.81\%, CI [22.69, 22.97]) outperforms all four source-trained sparsities and the random encoder, with a CI that does not overlap any encoder-based condition (next-highest, $k=32$ at 22.20\%, ends at 22.57\%). To our knowledge, this configuration---an Omniglot single-source CNN-4 transferred to CUB-200 at $28 \times 28$ grayscale---has not been characterized in the cross-domain few-shot literature. The pattern is consistent with \citet{chen2019closer}'s observation that simple baselines often outperform meta-learning under domain shift; the finding here is a stronger version of the same effect, in which a baseline that does not even use a learned representation outperforms multiple variants of one. We discuss likely mechanisms in Section~\ref{sec:discussion}.

---

## Notas pra #62 (slim revision)

- Word count atual ~890, dentro do target 800-1000
- Verificar se §4.5 + §4.6 podem ser consolidados (overlap pequeno)
- Decidir se mantém ranking decrescente em Tabela 1 ou ordena por categoria (retrained / cross-domain / pixel)
- Confirmar formato da Table 2 (booktabs `\toprule \midrule \bottomrule` consistente com paper C3)
- Considerar reduzir bullet list em §4.4 pra prosa contínua se reviewer comentar
