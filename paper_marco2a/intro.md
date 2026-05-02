# Section 1: Introduction

> Status: draft sessão #58.
> Word count target: 600-800 words. Atual: ~720.
> Tom: honesto, direto, sem overclaim. Workshop-scope.

---

**Author:** Luis Roberto Pinho da Silva Junior (Independent research)

> *"Don't try to build the mind. Build a neuron that works differently."*
> — Luis Roberto Pinho da Silva Junior, Project Hebb (2026)

---

Sparse coding is a hallmark of cortical computation: only a small fraction of neurons respond actively to each input, and this sparsity is functionally relevant for energy efficiency and representation separation~\citep{olshausen1996emergence,ahmad2019dense}. In mainstream deep learning, however, learned representations remain predominantly dense---post-ReLU activations typically have 30--50\% density, and no architectural constraint enforces additional sparsity. Recent work has begun to characterize when sparsity-inspired mechanisms are compatible with modern methods. \citet{pinho2026kwta}\footnote{Self-citation; see CITATION.cff in the project repository.} showed that explicit k-WTA sparse coding applied to the embedding of a Prototypical Network~\citep{snell2017prototypical} preserves few-shot accuracy on Omniglot~\citep{lake2015human}: 75\% of activations zeroed costs only 1.45 percentage points (93.10\% vs.\ 94.55\% for the full ProtoNet baseline), and even 87.5\% sparsity retains 90.77\%. That result establishes a positive in-domain claim about k-WTA tolerance.

A natural next question follows immediately: \emph{does the effect persist cross-domain?} Few-shot learning research has increasingly emphasized cross-domain robustness~\citep{triantafillou2020meta,tseng2020cross,chen2019closer}, motivated by the observation that meta-learning methods often degrade sharply when source and target domains differ~\citep{phoo2021self}. If sparse coding is a general principle of efficient representation, the expectation is that k-WTA should preserve at least \emph{some} of its in-domain benefit when an Omniglot-trained encoder is applied to a different visual domain. If, conversely, the effect of k-WTA is tied to in-domain regularities, the sparsity-accuracy curve documented on Omniglot should flatten or disappear under domain shift.

This paper presents a controlled empirical study addressing that question. We take the same CNN-4 ProtoNet encoder studied in~\citet{pinho2026kwta}, train it episodically on Omniglot at four sparsity levels ($k = 8, 16, 32, 64$, corresponding to 87.5\%, 75\%, 50\%, and 0\% sparsity over the 64-D embedding), freeze the weights, and evaluate cross-domain on CUB-200-2011~\citep{wah2011caltech}. The Omniglot $\rightarrow$ CUB transfer is intentionally adversarial: binary character strokes against natural RGB textures of bird species, a setting that fits the ``extreme task differences'' regime described by~\citet{phoo2021self}. To distinguish hypotheses about the source of any residual cross-domain signal, we include a Pixel kNN sanity floor (no encoder, raw pixel distances) and a Random encoder + k-WTA control (CNN-4 architecture, untrained Kaiming initialization, k-WTA applied). To establish a baseline for what is achievable with target-domain training, we also retrain ProtoNet directly on the CUB-200 train split at two resolutions: $28 \times 28$ grayscale (architecturally comparable with the source-trained encoders) and $84 \times 84$ RGB (with adaptive pooling preserving the 64-D embedding; this matches resolutions standard in cross-domain few-shot literature).

\textbf{Main empirical findings.} (i) The four sparsity levels tested cross-domain converge tightly within a 0.52 percentage point range (k=8: 21.68\%; k=16: 22.09\%; k=32: 22.20\%; k=64 / vanilla: 22.13\%), with all 95\% confidence intervals overlapping. The 3.78 p.p.\ in-domain spread documented in~\citet{pinho2026kwta} \emph{collapses} cross-domain. (ii) An encoder with random untrained weights followed by k-WTA $k=16$ reaches 21.91\%---statistically indistinguishable from the trained encoders, indicating that source-domain training contributes essentially no transferable signal in this setting. (iii) Pixel kNN, with no encoder at all, reaches 22.81\%, with a confidence interval that does not overlap any of the trained encoders, indicating that successive max-poolings in CNN-4 destroy more cross-domain information than they extract. (iv) ProtoNet retrained directly on CUB at $28 \times 28$ grayscale reaches 34.31\%, and at $84 \times 84$ RGB reaches 49.84\%---establishing that target-domain training (+12.22 p.p.\ over source-trained) and adequate input resolution (+15.53 p.p.\ over the 28-pixel grayscale baseline) are the operative bottlenecks, neither of which sparsity addresses.

The contribution of this work is deliberately empirical and characterizational, not methodological:

\begin{enumerate}
    \item We \textbf{quantify} the cross-domain behavior of k-WTA at four sparsity levels with bootstrap 95\% CIs over five seeds and 1000 episodes per seed, using a single source$\to$target pair (Omniglot$\to$CUB-200) chosen for its extreme distance.
    \item We \textbf{document a k-WTA effect collapse}: the in-domain sparsity-accuracy spread of 3.78 p.p.\ is reduced to 0.52 p.p.\ cross-domain, with overlapping CIs.
    \item We \textbf{decompose the cross-domain bottleneck} quantitatively, showing that target-domain training and input resolution each contribute roughly comparable improvements (+12.2 and +15.5 p.p.), while sparsity contributes none.
    \item We \textbf{characterize anti-transfer}: source-trained encoders perform statistically indistinguishably from random-weight controls in this regime.
\end{enumerate}

We do not claim a new method, biological realism, or a breakthrough mechanism. We claim a precise empirical statement: under extreme domain shift between visually unrelated source and target, an effect that mattered in-domain disappears, and bio-inspired sparsity neither helps nor hurts.

\textbf{Roadmap.} Section~\ref{sec:background} surveys cross-domain few-shot learning and prior work on sparse coding. Section~\ref{sec:method} details the encoders, the k-WTA layer, and the training and evaluation protocols. Section~\ref{sec:experiments} reports the seven characterized conditions and the in-domain comparison. Section~\ref{sec:discussion} discusses likely mechanisms and limitations. Section~\ref{sec:conclusion} concludes.

---

## Notas de revisão (pra sessão #62)

- [ ] Verificar word count (atual ~720, dentro do alvo 600-800)
- [ ] Confirmar que self-citation (`\citep{pinho2026kwta}`) tem entrada em refs.bib
- [ ] Conferir consistência com Tabela principal #59 (números 21.68, 22.09, 22.20, 22.13 etc)
- [ ] Decidir se mantém epígrafe inicial ou só na primeira página de paper_c3 (estilo)
- [ ] Considerar mover "We do not claim..." mais cedo no doc se tom soar defensivo
