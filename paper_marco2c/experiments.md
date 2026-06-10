# Section 4: Experiments

> Status: draft. Target 900-1100 words. Atual: ~960.

---

\section{Experiments}
\label{sec:experiments}

\subsection{Main Result on SHD}
\label{subsec:main}

Table~\ref{tab:shd} reports 5-seed accuracy on the SHD test split.

\begin{table}[tbp]
\centering\small
\caption{SHD test accuracy, 5 seeds, 95\% bootstrap CIs. Chance is 5\%.}
\label{tab:shd}
\begin{tabular}{lrr}
\toprule
Model & Accuracy & 95\% CI \\
\midrule
Timing-blind baseline (histogram + MLP) & 51.56\% $\pm$0.77 & [50.78, 52.09] \\
SNN feedforward                         & 61.02\% $\pm$0.85 & [60.27, 61.77] \\
SNN recurrent                           & 71.27\% $\pm$0.45 & [70.91, 71.70] \\
\bottomrule
\end{tabular}
\end{table}

The recurrent SNN exceeds the timing-blind baseline by \textbf{19.71 p.p.}, and the feedforward SNN by 10.26 p.p.; the three per-condition CIs do not overlap. Taken at face value, this 19.7 p.p.\ gap is the ``timing advantage.'' The rest of this section shows that roughly half of it is not timing.

\subsection{Disentangling Timing from Architecture}
\label{subsec:bins}

We sweep the number of time bins $T$ for the recurrent SNN, holding everything else fixed. At $T=1$ the input collapses to the per-channel histogram---the same information the blind baseline sees---so any advantage at $T=1$ is \emph{architectural}, not temporal. As $T$ grows, the \emph{same network} gains access to finer timing.

\begin{table}[tbp]
\centering\small
\caption{Recurrent SNN accuracy vs.\ temporal resolution (3 seeds). Timing-blind baseline: 49.41\%.}
\label{tab:bins}
\begin{tabular}{rrr}
\toprule
Time bins & Accuracy & vs.\ baseline \\
\midrule
1   & 58.92\% & +9.51 \\
4   & 63.06\% & +13.65 \\
16  & 67.29\% & +17.87 \\
32  & 68.45\% & +19.04 \\
100 & 69.10\% & +19.68 \\
\bottomrule
\end{tabular}
\end{table}

Accuracy rises monotonically with resolution (Table~\ref{tab:bins}). The decomposition is direct: from $T=1$ to $T=100$ the same network improves by \textbf{+10.18 p.p.}, which is attributable to timing alone (architecture is held fixed). The residual \textbf{+9.51 p.p.}\ at $T=1$ is the architectural advantage of the LIF+BatchNorm network over the MLP, with no timing involved. Thus of the 19.7 p.p.\ raw gap, only about half is genuine timing; reporting the raw gap overstates the temporal contribution roughly twofold.

\subsection{Onset Timing Alone Is Informative}
\label{subsec:latency}

Under latency coding, each channel contributes a single spike at its first-event time (Table~\ref{tab:latency}).

\begin{table}[tbp]
\centering\small
\caption{Rate vs.\ latency coding (3 seeds).}
\label{tab:latency}
\begin{tabular}{lrr}
\toprule
Coding & Timing-blind & SNN recurrent \\
\midrule
rate    & 49.41\% & 69.10\% \\
latency & 23.51\% & 50.68\% \\
\bottomrule
\end{tabular}
\end{table}

With one spike per channel---pure onset timing---the recurrent SNN still reaches \textbf{50.68\%}, far above chance (5\%) and above the latency-coded blind baseline (23.51\%, which retains only binary channel presence and is therefore weak). This shows the network reads onset \emph{timing}, not merely which channels activate. At the same time, full rate coding (69.10\%) beats latency (50.68\%) by 18.4 p.p.: the complete spike count carries substantial information beyond the onset. Timing helps, but it is not the whole story.

\subsection{Temporal k-WTA Tolerance Parallels the Spatial Case}
\label{subsec:kwta}

We apply k-WTA per time step on the hidden layer (Table~\ref{tab:kwta}).

\begin{table}[tbp]
\centering\small
\caption{Recurrent SNN under temporal k-WTA (3 seeds). Hidden size 256.}
\label{tab:kwta}
\begin{tabular}{rrr}
\toprule
$k$ & Sparsity & Accuracy (vs.\ dense) \\
\midrule
dense & 0\%    & 69.10\% \\
128 & 50\%     & 69.60\% (+0.50) \\
64  & 75\%     & 67.59\% (\textbf{$-$1.50}) \\
32  & 87.5\%   & 62.71\% ($-$6.39) \\
16  & 93.8\%   & 55.54\% ($-$13.56) \\
8   & 96.9\%   & 48.98\% ($-$20.11) \\
4   & 98.4\%   & 42.64\% ($-$26.46) \\
\bottomrule
\end{tabular}
\end{table}

Temporal k-WTA is tolerated up to 75\% sparsity ($k=64$) at a cost of only \textbf{1.50 p.p.}---almost exactly the 1.45 p.p.\ cost of 75\% \emph{spatial} k-WTA reported in-domain for Prototypical Networks~\citep{pinho2026kwta}. Degradation accelerates past 87.5\% and collapses beyond 96\% ($k \le 8$), where accuracy returns to the timing-blind baseline: with too few active units per step, the temporal representation has insufficient capacity. The quantitative match across two different axes (spatial embedding vs.\ temporal hidden activity) suggests k-WTA compatibility with a trained representation is a general in-domain property, not specific to the spatial case.

\subsection{Generalization to SSC Is Weak but Stable}
\label{subsec:ssc}

We repeat the main comparison on the harder SSC (35 classes; full $\sim$75k training set). The recurrent SNN reaches 30.83\% versus 25.63\% for the blind baseline, a timing margin of \textbf{+5.20 p.p.} (35 epochs). Training longer (60 epochs) does not grow the margin---it plateaus at +3.95 p.p., with the SNN slightly overfitting---so the modest margin is the model's ceiling on SSC, not undertraining. The timing effect therefore \emph{generalizes qualitatively} (positive on both datasets) but is \emph{dataset-specific in magnitude}: strong on SHD ($\sim$+20 p.p.), weak on SSC ($\sim$+5 p.p.). We note the SNN's absolute SSC accuracy ($\sim$30\%) is well below both the SHD result and the literature ($\sim$50--70\% with heavier preprocessing and augmentation), bounding these conclusions to our simple pipeline.

---

## Notas de revisão
- [ ] Conferir todos os números contra `experiment_05_temporal/WEEKLY-1.md` e `results_*.txt`
- [ ] Gerar Fig 1 (barras SHD), Fig 2 (curva bins), Fig 3 (curva k-WTA) — script reusável estilo paper_c3/generate_figures.py
- [ ] Decidir se §4.3 latency vira figura ou fica tabela
