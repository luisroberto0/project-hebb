# Section 1: Introduction

> Status: draft no kickoff. Target 500-700 words. Atual: ~640.

---

**Author:** Luis Roberto Pinho da Silva Junior (Independent research)

---

\section{Introduction}
\label{sec:intro}

Spiking neural networks (SNNs) process information as discrete events in time, and a recurring claim is that they exploit the \emph{timing} of those events---that the moment a unit fires, not merely whether it fires, carries information~\citep{maass1997networks}. On temporal benchmarks such as the Spiking Heidelberg Digits (SHD)~\citep{cramer2020heidelberg}, recurrent SNNs reach accuracies well above non-temporal baselines, and this gap is often attributed to timing. We argue that the attribution is, as usually measured, confounded.

The advantage of a recurrent SNN over a non-temporal baseline mixes two distinct effects. The first is information genuinely carried by spike timing: if the same network is given finer temporal resolution and improves, that improvement is attributable to timing. The second is architectural: an SNN uses LIF dynamics, often batch normalization, and a different readout than a plain multilayer perceptron (MLP), and these differences can move accuracy regardless of any temporal structure. A network given only a single time step---which destroys all timing---can still differ from an MLP for purely architectural reasons. Reporting the raw gap therefore \emph{overstates} how much timing helps.

This paper characterizes a recurrent LIF network on SHD with that confound controlled. Our baseline is deliberately \emph{timing-blind}: we sum each input channel's spikes over the entire trial (a per-channel histogram) and classify the resulting vector with an MLP. This preserves \emph{which} channels fired---the average spectral content---while destroying \emph{when}. The difference between the recurrent SNN and this baseline isolates the contribution of timing, controlling for the spectral information both representations share.

\textbf{Main findings.} (i) On SHD (5 seeds, bootstrap 95\% CIs), the recurrent SNN reaches 71.27\% versus 51.56\% for the timing-blind baseline---a raw gap of 19.7 p.p. (ii) Sweeping the number of time bins from 1 to 100 shows accuracy rising monotonically; the \emph{same} network improves by 10.18 p.p.\ as resolution increases, which we read as the \textbf{genuine timing contribution}. The residual (a single-bin SNN still beats the MLP by $\sim$9.5 p.p.) is architectural, not temporal. (iii) Under latency coding (one spike per channel, at the time of its first event), the network still reaches 50.68\%---far above chance and above a presence-only baseline---showing it reads onset timing, not merely which channels activate. (iv) Temporal k-WTA, which keeps only the $k$ most-active hidden units per time step, tolerates 75\% sparsity at a cost of 1.50 p.p.; this nearly matches the 1.45 p.p.\ tolerance of \emph{spatial} k-WTA reported in-domain for Prototypical Networks~\citep{pinho2026kwta}, and collapses only under extreme ($>$96\%) sparsity. (v) The effect is dataset-specific: on the larger, harder Spiking Speech Commands the timing margin is weak but stable ($+4$--$5$ p.p., not an artifact of undertraining).

The contribution of this work is characterizational, not methodological:

\begin{enumerate}
    \item We \textbf{disentangle timing from architecture} by sweeping temporal resolution, attributing +10.18 p.p.\ to genuine timing and the rest to the SNN architecture.
    \item We corroborate the timing contribution with \textbf{latency coding} (50.68\% from onsets alone) and a monotonic resolution--accuracy curve.
    \item We show \textbf{temporal k-WTA sparsity is tolerated to 75\%}, a quantitative parallel to spatial k-WTA tolerance in metric learning.
    \item We \textbf{measure the generalization limit}: the effect is strong on SHD and weak-but-stable on SSC.
\end{enumerate}

We make no claim of superiority over conventional methods---a convolutional or transformer model on a spectrogram exceeds 90\% on these benchmarks, well above the SNN. We make no claim of a new learning mechanism: training is by surrogate-gradient backpropagation. The value is a controlled, honest measurement of how much spike timing contributes, and where that contribution ends.

---

## Notas de revisão
- [ ] Confirmar word count (~640)
- [ ] Garantir `\citep{pinho2026kwta}` em refs.bib (reusar de paper_marco2a)
- [ ] Conferir números contra `experiment_05_temporal/WEEKLY-1.md` (#72–#77b)
