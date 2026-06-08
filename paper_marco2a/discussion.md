# Section 5: Discussion

> Status: drafted sessão #61.
> Word count target: 700-900 words. Atual: ~840.

---

\section{Discussion}
\label{sec:discussion}

\subsection{Why the k-WTA Effect Collapses Cross-Domain}
\label{subsec:why-collapse}

The in-domain result of \citet{pinho2026kwta} is that constraining a trained 64-D embedding to its $k$ largest coordinates costs little accuracy because the encoder, optimized on Omniglot under that constraint, learns to concentrate class-discriminative information in the surviving coordinates. Cross-domain, two conditions that made the operation meaningful are absent. First, the encoder is never optimized on CUB, so there is no reason for bird-discriminative information to occupy the top-$k$ coordinates of the embedding it produces. Second, and more fundamentally, our results suggest that little such information reaches the embedding at all: Pixel kNN (22.81\%) outperforms every CNN-4 forward pass (21.68--22.20\%) with a non-overlapping confidence interval. The four max-pooling stages that reduce a $28 \times 28$ input to a $1 \times 1$ feature map ($28 \to 14 \to 7 \to 3 \to 1$) appear to discard more cross-domain-relevant signal than they preserve, leaving an embedding that is already near-degenerate before k-WTA is applied. Selecting the top-$k$ of an uninformative vector is itself uninformative, regardless of $k$; the 0.52 p.p.\ spread is what that degeneracy looks like quantitatively.

\subsection{Anti-Transfer Rather Than Neutral Transfer}
\label{subsec:anti-transfer-disc}

The 0.18 p.p.\ gap between source-trained and random encoders (Section~\ref{subsec:anti-transfer}) is the sharper of our findings. Episodic training on Omniglot does not merely fail to help on CUB---it fails to produce any measurable advantage over Kaiming-initialized random weights. We describe this as \emph{anti-transfer} rather than neutral transfer because the trained encoder also does not match the simplest representation-free baseline: Pixel kNN exceeds it. An encoder optimized to separate binary character strokes evidently learns filters whose inductive bias is actively unhelpful for natural textures, to the point that raw pixel distances are preferable to the features it computes. This is consistent with the central premise of STARTUP~\citep{phoo2021self}: under extreme task differences, source-domain training alone is insufficient, and adaptation on the target domain---which our frozen-encoder protocol deliberately forbids---is the operative ingredient. Our setting can be read as an empirical lower bound on what frozen source transfer achieves when the source is maximally distant.

\subsection{Implications for Bio-Inspired Sparsity}
\label{subsec:implications}

Read together with \citet{pinho2026kwta}, these results bound the scope of a bio-inspired sparsity claim rather than refute it. In-domain, k-WTA is \emph{compatible}: 75\% sparsity is essentially free. Cross-domain under extreme shift, k-WTA is \emph{invisible}: it neither recovers nor degrades accuracy because the representation it operates on carries no transferable signal. Sparsity is therefore neither a universal benefit nor a liability; its effect is contingent on the encoder having learned a relevant representation in the first place. For arguments that sparse coding is a general principle of efficient cortical computation, this is a useful boundary condition: the principle's measurable benefit, at least in this prototype-based setting, is inherited from---not independent of---the quality of the underlying features.

\subsection{Relation to Cross-Domain Few-Shot Literature}
\label{subsec:lit-comparison}

Our numbers extend the trend reported across the cross-domain literature to a more extreme point. \citet{tseng2020cross} report a vanilla ProtoNet dropping from 49.4\% in-domain to 38.0\% on \emph{mini}-ImageNet$\to$CUB; our Omniglot$\to$CUB transfer drives the same architecture to 22\%, within 2 p.p.\ of chance. The monotonic relationship between source--target distance and degradation is intuitive, but the magnitude here is notable: the transfer is weak enough that the meta-learned encoder loses to a representation-free baseline. This is a strengthened form of the observation by \citet{chen2019closer} that simple baselines often beat meta-learning under domain shift---in our regime, a baseline that uses no learned representation beats multiple variants of one. We read the Omniglot$\to$CUB pair as a useful stress-test anchor at the far end of the ``extreme task differences'' axis defined by \citet{phoo2021self}.

\subsection{Limitations}
\label{subsec:limitations}

The study is deliberately narrow and its conclusions should not be over-generalized. We use a single source dataset (Omniglot), a single target dataset (CUB-200), and a single backbone (CNN-4); we did not test ResNet or transformer encoders, which have different pooling structure and might preserve more cross-domain signal. Our primary input pipeline is $28 \times 28$ grayscale, chosen for architectural comparability with the source-trained encoders; the $84 \times 84$ RGB result enters only through the retrained baseline. We apply k-WTA only at the final embedding, not at intermediate layers, and we do not test any target-domain adaptation. Each of these is a plausible reason the collapse might be less complete in a different configuration.

\subsection{Future Work}
\label{subsec:future-work}

Three directions follow directly. First, varying source--target distance---e.g.\ \emph{mini}-ImageNet$\to$CUB versus Omniglot$\to$CUB under an identical k-WTA sweep---would test whether the effect collapse is gradual or threshold-like. Second, applying k-WTA at intermediate convolutional layers, rather than only the embedding, would test whether sparsity interacts with cross-domain features before pooling destroys them. Third, integrating target-domain self-training (STARTUP-style)~\citep{phoo2021self} would test whether the anti-transfer pattern reverses once the encoder is allowed to adapt. Extending to multiple targets (Cars, Places, Plantae) would establish whether the collapse generalizes beyond the single Omniglot$\to$CUB pair characterized here.

---

## Notas pra #62 (slim revision)

- Word count atual ~840, dentro do target 700-900
- Confirmar `\ref{subsec:anti-transfer}` e `\ref{subsec:retrained}` resolvem pros labels reais de experiments.md/methods.md
- §5.4 reusa o número 38% já citado em background §2.1 — verificar se não soa repetitivo; pode encurtar
- Decidir se §5.5 + §5.6 ficam separadas ou consolidam num parágrafo "Limitations and future work" (estilo paper C3 §5.3-5.4)
- Tom de hipótese preservado ("suggests", "appear to", "consistent with") — não afirmar mecanismo como certeza
