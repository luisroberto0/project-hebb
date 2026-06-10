# Section 3: Method

> Status: draft. Target 500-700 words. Atual: ~640.

---

\section{Method}
\label{sec:method}

\subsection{Models}

All three models map a sample to logits over the classes; they differ only in how they use time.

\textbf{Timing-blind baseline (BlindMLP).} The input spike tensor $(T, 700)$ is summed over time to a 700-dimensional per-channel histogram, then passed through an MLP ($700 \to 256 \to C$, ReLU). It has no access to temporal order.

\textbf{Feedforward SNN.} A LIF layer ($700 \to 256$) processes the input one time step at a time, followed by an integrating readout ($256 \to C$); no recurrence.

\textbf{Recurrent SNN.} As above, with an all-to-all recurrent connection in the hidden layer: $I_t = \mathrm{BN}(W_{\text{in}} x_t) + W_{\text{rec}} s_{t-1}$, where $s_{t-1}$ is the previous hidden spike vector. This is the model that can integrate timing across the trial.

Hidden size 256, LIF decay $\beta = 0.9$, fast-sigmoid surrogate. The readout is an \emph{integrator}: logits are the sum over time of a linear projection of the hidden spikes, $\sum_t W_{\text{out}} s_t$. We found a spiking output layer trained poorly (the output rarely fired, starving the loss of gradient); the linear integrating readout gives continuous, well-conditioned logits.

\subsection{A Reproducible Training Detail: BatchNorm Statistics}

Without normalization the input current sat far below threshold (hidden firing rate $\sim$3.7\%) and training was slow. Adding BatchNorm on the hidden pre-activation fixed the scale, but introduced a subtle failure: with default running statistics, accuracy trained to $>$70\% yet \emph{tested} near chance (13\%). The cause is that a single BatchNorm, applied at every one of the 100 time steps, accumulates running mean/variance that mix all time steps; at evaluation those pooled statistics are wrong for any individual step. Setting \texttt{track\_running\_stats=False} (using batch statistics at test time as well) resolves it, lifting test accuracy from 13\% to 71\%. We report this because it is an easy, silent trap when batch-normalizing a per-time-step SNN.

\subsection{Input Encodings}

\textbf{Rate.} Each $(T,700)$ tensor counts spikes per (time-bin, channel). Time is binned into $T$ bins (default $T=100$) over the $\sim$1.4\,s trial.

\textbf{Latency (time-to-first-spike).} For each channel, a single spike is placed in the bin of its \emph{first} event; later spikes are discarded. This keeps only onset timing (one spike per active channel) and lets us test whether onset timing alone is informative.

\subsection{Temporal k-WTA}

At each time step, after computing hidden spikes, we keep only the $k$ units with the largest membrane potential and zero the rest, enforcing at most $k$ active hidden spikes per step. Sweeping $k \in \{128, 64, 32, 16, 8, 4\}$ over a hidden size of 256 spans 50\%--98.4\% temporal sparsity.

\subsection{Datasets and Evaluation}

We use the official SHD train/test split (speaker-held-out test). Spikes are read from the published HDF5 files. The main result uses 5 seeds with 95\% bootstrap confidence intervals over the per-seed accuracies. The resolution, latency, and k-WTA characterizations use 3 seeds. For the SSC generalization (Section~\ref{sec:experiments}) we train on the full $\sim$75{,}000-sample training set via lazy HDF5 loading. Training uses Adam ($10^{-3}$) and cross-entropy on the integrated logits. All experiments run on a single RTX 4070 laptop GPU; we note that the recurrent loop over 100 time steps leaves the GPU largely idle, a practical signature of how poorly the sequential-temporal computation maps onto dense parallel hardware.
