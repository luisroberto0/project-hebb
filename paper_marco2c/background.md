# Section 2: Background

> Status: draft. Target 500-700 words. Atual: ~620.

---

\section{Background}
\label{sec:background}

\subsection{Spiking Networks and Surrogate-Gradient Training}

A spiking neuron integrates input over time and emits a discrete spike when its membrane potential crosses a threshold, after which the potential resets. We use the leaky integrate-and-fire (LIF) neuron: $u_t = \beta u_{t-1} + I_t$, with a spike when $u_t > \vartheta$ and a reset thereafter. The spike function is non-differentiable, so end-to-end training uses a \emph{surrogate gradient}---a smooth function substituted for the derivative of the threshold during backpropagation through time~\citep{neftci2019surrogate}. We use the fast-sigmoid surrogate. We emphasize at the outset that surrogate-gradient training \emph{is} backpropagation; this work characterizes what the temporal dynamics compute, not a backprop-free learning rule.

A recurrent SNN adds lateral connections in the hidden layer, so the state at one time step feeds the next, giving the network an explicit mechanism to integrate information across time. We contrast a feedforward SNN (temporal input, no recurrence) and a recurrent SNN against a non-spiking baseline.

\subsection{The Spiking Heidelberg Datasets}

The Spiking Heidelberg Digits (SHD) and Spiking Speech Commands (SSC) datasets~\citep{cramer2020heidelberg} convert spoken audio into spike trains through a model of the inner ear: each utterance becomes events across 700 input channels (frequency-like cochlear bands) over roughly one second. SHD contains spoken digits 0--9 in English and German (20 classes, $\sim$10{,}000 samples, speaker-held-out test split). SSC is larger and harder: 35 command words, $\sim$100{,}000 samples. Because the information is intrinsically temporal, these benchmarks are standard for evaluating whether a model exploits timing. Reported recurrent-SNN accuracies on SHD lie in the 70--83\% range; feedforward SNNs are lower (48--66\%).

\subsection{k-WTA and Sparse Coding}

The $k$-winner-take-all (k-WTA) operation keeps the $k$ largest entries of an activation vector and zeros the rest~\citep{maass2000computational}. It has direct biological motivation---cortical activity is sparse---and appears across cortex-inspired models. In a companion study on Prototypical Networks~\citep{pinho2026kwta}, k-WTA applied to a \emph{spatial} embedding was shown to be tolerated in-domain: 75\% sparsity cost only 1.45 percentage points on Omniglot few-shot classification. Here we apply k-WTA along the \emph{temporal} axis (limiting how many hidden units may fire per time step) and ask whether the same tolerance holds. The parallel matters: it tests whether the compatibility of k-WTA sparsity with a trained representation is a property of the axis (spatial) or a more general phenomenon.

\subsection{Why a Timing-Blind Baseline}

To isolate the contribution of timing we need a baseline that has access to everything \emph{except} timing. We use the per-channel spike histogram: for each of the 700 channels, the total number of spikes over the trial, classified by an MLP. This representation preserves \emph{which} channels were active and how much---the average spectral envelope, which is itself discriminative for speech---while discarding \emph{when} each spike occurred. The gap between a temporal model and this baseline therefore measures the value of timing \emph{controlling for} the spectral information both share. This is a stronger control than comparing against chance: on SHD the timing-blind baseline already reaches $\sim$51\%, far above the 5\% chance level, precisely because the spectral envelope carries much of the signal. What the SNN must add, to justify its temporal machinery, is information that survives only in the timing.
