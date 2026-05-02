# Section 2: Background

> Status: draft sessão #58.
> Word count target: 700-900 words. Atual: ~810.

---

\section{Background}
\label{sec:background}

\subsection{Cross-Domain Few-Shot Learning}

Few-shot classification has historically been studied in single-domain settings, where the same dataset provides training and evaluation episodes split by class~\citep{lake2015human,vinyals2016matching,snell2017prototypical,finn2017model}. The cross-domain variant---where training and test episodes are drawn from different visual domains---was systematically defined by~\citet{triantafillou2020meta} (Meta-Dataset), which combined 10 sources (ImageNet, Omniglot, Aircraft, CUB-200-2011~\citep{wah2011caltech}, Quick Draw, Fungi, VGG Flower, Traffic Signs, MSCOCO, MNIST) and demonstrated that no single method dominates across domains. \citet{tseng2020cross} introduced feature-wise transformation layers to simulate distribution shift during meta-training, reporting that a vanilla ProtoNet trained on \emph{mini}-ImageNet drops from 49.4\% in-domain accuracy to 38.0\% on CUB-200 in the 5-way 1-shot setting. \citet{chen2019closer} compared baselines and meta-learning methods systematically on \emph{mini}-ImageNet$\to$CUB and showed that simpler baselines (pretrain plus cosine-similarity classifier) often \emph{outperform} sophisticated meta-learning under domain shift. \citet{phoo2021self} pushed further with the BSCD-FSL benchmark, examining ``extreme task differences'' (ImageNet$\to$ChestX, ISIC, EuroSAT, CropDiseases) and proposed STARTUP, which uses unlabeled target-domain self-training to bridge the gap, reporting that without target-domain adaptation ProtoNet baselines fall to the 22--40\% range in 5-way 5-shot accuracy.

The Omniglot$\to$CUB-200 single-source transfer studied here is, to our knowledge, not directly precedented in this literature. Most cross-domain studies use a natural-image source (ImageNet variants) and natural-image targets, with the source domain typically richer than the target. We deliberately invert this: a binary-character source ($28 \times 28$ grayscale strokes) against a natural-texture target (RGB bird photographs at native $\sim 500 \times 500$ resolution, here resized for compatibility). The resulting domain gap is wider than any pair previously characterized, and consequently makes it possible to test whether properties documented in-domain (e.g., k-WTA tolerance) survive at the limit of transferability.

\subsection{k-WTA and Sparse Coding}

The k-winner-take-all (k-WTA) operation is a generalization of classical winner-take-all competition~\citep{maass2000computational}: given a vector of activations, only the $k$ largest values are preserved and the remaining $n-k$ are zeroed. The operation has direct biological motivation: cortical circuits exhibit sparse activity patterns~\citep{olshausen1996emergence}, and lateral inhibition produces approximate winner-take-all dynamics in many neural systems. In machine learning, k-WTA has appeared in Hierarchical Temporal Memory~\citep{ahmad2019dense} and in Modern Hopfield Networks~\citep{ramsauer2021hopfield}, where sparsity supports exponential capacity and pattern separation.

Within Prototypical Networks specifically, \citet{pinho2026kwta} applied k-WTA to the 64-dimensional final embedding of a CNN-4 ProtoNet, varying $k$ across $\{32, 16, 8\}$ (corresponding to 50\%, 75\%, 87.5\% sparsity). On Omniglot 5-way 1-shot, the accuracy curve was nearly flat between 0\% and 75\% sparsity (94.55\% to 93.10\%, a drop of 1.45 p.p.), with a steeper decline at 87.5\% sparsity (90.77\%). A control with random encoder weights and the same k-WTA operation reached 37.60\%, confirming that the in-domain result depends on training under the sparsity constraint, not on the k-WTA structure applied to arbitrary features. The present paper takes those exact source-trained encoders as inputs and asks whether the in-domain effect survives transfer.

\subsection{Bio-Plausible Learning and Domain Transfer}

Bio-plausible learning research has largely focused on demonstrating capabilities (sparse coding, local plasticity, episodic memory) within the domains where they are documented to occur. Cross-domain robustness of bio-inspired mechanisms has received less direct empirical attention. The implicit hope---occasionally explicit in motivational text---is that biology-inspired regularizers (sparsity, local rules, distributed representations) will produce more general features than purely backprop-driven training, because biology must operate across a single substrate that handles all sensory modalities.

The empirical record on this hope is mixed. \citet{chen2019closer} showed that simple baselines can outperform meta-learning under domain shift, suggesting that complex inductive biases learned in-domain may not transfer well. \citet{phoo2021self} showed that source-domain training, however sophisticated, is insufficient when the target is sufficiently different: target-domain self-training is required. These results raise the possibility that bio-inspired sparsity, like meta-learned representations, is domain-locked in the sense that what helps in one domain does not necessarily help in another.

\subsection{Datasets: Omniglot and CUB-200-2011}

\textbf{Omniglot}~\citep{lake2015human} contains 1623 hand-drawn characters from 50 alphabets, 20 instances per character, originally provided as $28 \times 28$ binary images suitable for one-shot benchmarks. We use the standard background/evaluation split (30 alphabets / 20 alphabets) for source training. Following~\citet{pinho2026kwta}, we report 5-way 1-shot performance.

\textbf{CUB-200-2011}~\citep{wah2011caltech} contains 11{,}788 photographs of 200 bird species, with 30--60 images per class at native resolution near $500 \times 500$ RGB. The dataset includes bounding boxes, part locations, and 312 attribute labels (we do not use these annotations). The official \texttt{train\_test\_split.txt} provides a per-image binary split that we adopt without modification: 5994 images for the train split and 5794 images for the test split, with all 200 classes appearing in both. For cross-domain evaluation we use only the test split. For the retrained baseline (Section~\ref{sec:method}.\ref{subsec:retrained}), we use the train split.

In our primary protocol all images, regardless of source, are resized to $28 \times 28$ and converted to grayscale. This preserves architectural compatibility with the source-trained encoders' input shape $(1, 28, 28)$. The cost is severe: $500 \times 500$ RGB photographs reduced to $28 \times 28$ grayscale lose nearly all texture and color information that conventionally distinguishes bird species. We report the impact of this resolution choice in Section~\ref{sec:experiments} via a parallel $84 \times 84$ RGB retrained baseline.

The visual distance between Omniglot and CUB-200-2011 in any reasonable embedding metric is large. We treat this as an instance of the ``extreme task differences'' regime~\citep{phoo2021self} and characterize what survives transfer in that regime.

---

## Notas de revisão (pra sessão #62)

- [ ] Verificar word count (~810, dentro de 700-900)
- [ ] Confirmar todas citações em refs.bib
- [ ] Conferir labels `\ref{sec:method}.\ref{subsec:retrained}` corretos quando #59 escrever methods
- [ ] Considerar consolidar §2.1 + §2.3 se reviewer comentar redundância
- [ ] Decidir se "Bio-Plausible Learning and Domain Transfer" §2.3 é necessária ou sobra
