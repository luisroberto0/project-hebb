# Abstract

> Status: draft no kickoff. Target 150-180 words. Atual: ~185.

---

\begin{abstract}
Recurrent spiking neural networks (SNNs) are routinely reported to exploit temporal structure, but the measured advantage over a non-temporal baseline conflates two effects: information genuinely carried by spike \emph{timing}, and architectural differences (LIF dynamics, normalization) unrelated to timing. We characterize a recurrent LIF network on the Spiking Heidelberg Digits (SHD) dataset, separating the two. The recurrent SNN reaches 71.27\% versus 51.56\% for a timing-blind baseline (a per-channel spike histogram followed by an MLP), a gap of 19.7 percentage points. By sweeping the temporal resolution, we attribute \textbf{+10.18 p.p.\ to genuine timing} (the same network as it goes from 1 to 100 time bins) and \textasciitilde+9.5 p.p.\ to architecture (a single-bin SNN still beats the MLP). The network recovers 50.68\% from onset timing alone (one spike per channel). Temporal k-WTA sparsity is tolerated up to 75\% at a cost of 1.50 p.p.---nearly identical to the 1.45 p.p.\ tolerance reported for \emph{spatial} k-WTA in-domain on Prototypical Networks. The effect is dataset-specific: on the harder Spiking Speech Commands it is weak but stable (+4--5 p.p.). We claim no superiority over conventional methods and no new mechanism; we quantify, with controls, how much timing helps and where its benefit ends.
\end{abstract}
