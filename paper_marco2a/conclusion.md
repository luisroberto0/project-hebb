# Section 6: Conclusion

> Status: drafted sessão #62.
> Word count target: 150-200 words. Atual: ~185.

---

\section{Conclusion}
\label{sec:conclusion}

We characterized the cross-domain behavior of k-winner-take-all sparsity by transferring CNN-4 Prototypical Network encoders trained on Omniglot, at four sparsity levels, to CUB-200-2011, evaluated over five seeds and 1000 episodes each with bootstrap confidence intervals. The in-domain sparsity-accuracy spread of 3.78 percentage points reported by \citet{pinho2026kwta} collapses to 0.52 p.p.\ under this extreme domain shift, entirely within the overlap of per-condition confidence intervals. A decomposition of the residual gap shows that target-domain training (+12.22 p.p.) and adequate input resolution (+15.53 p.p.) are the operative bottlenecks, while sparsity contributes none; a source-trained encoder is statistically indistinguishable from a random-weight control, and both are outperformed by a representation-free Pixel kNN baseline.

The practical implication is narrow but precise: bio-inspired sparsity that is demonstrably compatible in-domain neither helps nor hurts under extreme domain shift, because the operation is applied to a representation that carries no transferable signal. Whether the collapse softens with a closer source, intermediate-layer sparsity, or target-domain self-training remains open.

---

## Notas pra #62 (slim revision)

- Word count ~185, dentro do target 150-200
- Não introduz claim novo (todos os números já em Abstract/Experiments)
- Confirmar `\citet{pinho2026kwta}` consistente
