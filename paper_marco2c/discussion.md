# Section 5: Discussion

> Status: draft. Target 600-800 words. Atual: ~700.

---

\section{Discussion}
\label{sec:discussion}

\subsection{Why Disentangling Timing from Architecture Matters}

The headline number for a temporal model is usually its gap over a non-temporal baseline, and that gap is read as the value of timing. Our resolution sweep shows the reading is inflated: half of the 19.7 p.p.\ gap on SHD survives even when the network is given a \emph{single} time bin, i.e.\ no timing at all. That residual is the LIF+normalization architecture out-performing an MLP on the same static histogram. The genuinely temporal contribution---the part that appears only as resolution increases---is +10.18 p.p., about half the raw figure. The control is cheap (vary the number of bins) and we suggest it as a default when claiming a timing benefit: report the single-bin accuracy alongside the full-resolution one, so the architectural and temporal components are visible separately. Two further probes corroborate the temporal component rather than relying on it alone: accuracy rises monotonically with resolution, and onset-only (latency) coding already recovers 50.68\%.

\subsection{k-WTA Is Compatible In-Domain on Either Axis}

The temporal k-WTA result lines up almost exactly with the spatial k-WTA result from the companion Prototypical-Network study: 75\% sparsity costs 1.50 p.p.\ here and 1.45 p.p.\ there. Despite acting on entirely different quantities---which hidden units may fire \emph{per time step} versus which embedding coordinates survive---the tolerance is the same to within noise, and both degrade sharply only at extreme sparsity. This suggests that the compatibility of k-WTA with a trained representation is not a peculiarity of the spatial embedding but a more general in-domain property: a network trained under a sparsity constraint learns to concentrate the signal into the surviving units, on whichever axis the constraint is imposed. (In the companion work, the same spatial k-WTA \emph{collapses} under domain shift; whether temporal k-WTA is similarly fragile across, say, SHD$\to$SSC is left open.)

\subsection{Why the Effect Is Weaker on SSC}

The timing margin shrinks from $\sim$20 p.p.\ on SHD to $\sim$5 p.p.\ on SSC, and this is not undertraining: more epochs plateau the margin. Several factors plausibly contribute---SSC has 35 classes versus 20, more speaker variability, and a relatively stronger spectral baseline---but our setup also reaches only $\sim$30\% absolute, well below literature pipelines. We therefore frame this as a measured \emph{limit}, not a mechanism: the timing benefit demonstrated cleanly on SHD does not transfer at the same magnitude to a harder benchmark under a simple pipeline. Reporting it is the honest counterweight to the positive SHD result.

\subsection{Limitations}

The study is deliberately narrow. (i) Two datasets from one source (SHD/SSC); no claim about other temporal benchmarks. (ii) Training is surrogate-gradient backpropagation through time---this characterizes the temporal \emph{dynamics}, not a local or backprop-free rule. (iii) We do not compare against conventional sequence models (RNNs, transformers); on a spectrogram those exceed 90\% on these tasks, so the SNN is not competitive in absolute accuracy, and we make no efficiency claim here. (iv) Everything runs on a GPU emulating the spiking dynamics with a Python time loop; the energy advantage that motivates spiking hardware is not measured and would require neuromorphic silicon. (v) A single recurrent architecture and hidden size; no architecture search.

\subsection{What We Do Not Conclude}

We do not conclude that spiking networks are a better way to process audio, nor that timing is the dominant cue (rate coding beats latency by 18 p.p.), nor that this is a path to capabilities conventional networks lack. We conclude only what we measured: spike timing carries information recoverable by a recurrent SNN, worth about +10 p.p.\ over architecture on SHD, corroborated by latency coding and the resolution curve, compatible with up to 75\% temporal sparsity, and modest and dataset-specific in magnitude.
