# Abstract

> Status: drafted sessão #62.
> Word count target: 150-180 words. Atual: ~170.

---

\begin{abstract}
The k-winner-take-all (k-WTA) operation imposes sparsity on learned representations and has been shown to preserve few-shot accuracy in-domain: on Omniglot, a Prototypical Network tolerates 75\% embedding sparsity at a cost of only 1.45 percentage points. We ask whether this tolerance survives cross-domain transfer. Taking CNN-4 ProtoNet encoders trained on Omniglot at four sparsity levels ($k \in \{8,16,32,64\}$), we freeze their weights and evaluate on CUB-200-2011, an intentionally adversarial transfer from binary character strokes to natural bird textures. Across five seeds and 1000 episodes per condition, the in-domain sparsity-accuracy spread of 3.78 p.p.\ collapses to 0.52 p.p.\ cross-domain, with fully overlapping confidence intervals. Source-trained encoders are statistically indistinguishable from random-weight controls, and both are outperformed by a representation-free Pixel kNN baseline. A bottleneck decomposition attributes the recoverable gap to target-domain training (+12.22 p.p.) and input resolution (+15.53 p.p.), not sparsity. Under extreme domain shift, bio-inspired sparsity is neither beneficial nor harmful---it is invisible.
\end{abstract}
