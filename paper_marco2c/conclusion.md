# Section 6: Conclusion

> Status: draft. Target 150-200 words. Atual: ~175.

---

\section{Conclusion}
\label{sec:conclusion}

We characterized how much a recurrent spiking network exploits spike timing on the Spiking Heidelberg Digits, with the architectural confound controlled. Against a timing-blind baseline (a per-channel spike histogram), the recurrent SNN gains 19.7 percentage points; but by sweeping temporal resolution we attribute only +10.18 p.p.\ to genuine timing and the remainder to the spiking architecture itself---roughly halving the naive figure. Latency coding (50.68\% from onsets alone) and a monotonic resolution--accuracy curve corroborate the temporal contribution. Temporal k-WTA sparsity is tolerated up to 75\% at a cost of 1.50 p.p., almost identical to the spatial k-WTA tolerance reported in-domain for Prototypical Networks---evidence that k-WTA compatibility is a general in-domain property across axes. The effect is strong on SHD and weak-but-stable on the harder Spiking Speech Commands (+4--5 p.p.), bounding its generality.

The contribution is a controlled, honest measurement: how much timing helps, corroborated by independent probes, and where its benefit ends---not a new method, and not superiority over conventional models.
