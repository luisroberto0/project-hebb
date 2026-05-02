# Section 3: Method

> Status: PLACEHOLDER. Draft pra sessão #59.
> Word count target: 500-700 words.

---

\section{Method}
\label{sec:method}

[Draft #59. Estrutura planejada:]

- §3.1 Architecture: CNN-4 ProtoEncoder (Snell 2017), 64-D embedding, k-WTA over embedding
- §3.2 k-WTA layer: definição matemática top-k, gradient flow
- §3.3 Source training: Omniglot 5w1s5q, 5000 episodes, Adam lr=1e-3, mesmos hyperparams paper C3 #20
- §3.4 Cross-domain protocol: encoder weights frozen, eval em CUB test split via splits oficiais train_test_split.txt
- §3.5 Sparsity sweep: k ∈ {8, 16, 32, 64} (87.5%, 75%, 50%, 0%)
- §3.6 Sanity floors: Pixel kNN, Random encoder + k-WTA k=16
- §3.7 Re-trained baseline (label `subsec:retrained`): ProtoNet retreinado em CUB-200 train split, 28×28 grayscale + 84×84 RGB com AdaptiveAvgPool
- §3.8 Evaluation: 5 seeds × 1000 episodes 5w1s5q, IC95% bootstrap

---

## Notas pra #59

- Pode reusar Section 3 do paper C3 main.tex (linhas ~80-140 do main.tex) com modificações
- Importante incluir info sobre ProtoEncoderRGB (CNN-4 + AdaptiveAvgPool2d) que é específico Marco 2-A
- Detalhes pré-processamento: 28×28 grayscale (compat C3) vs 84×84 RGB (literatura)

### A INCLUIR EM #59 §3 (movido de §2.4 background em #58.5)

Conteúdo movido de Section 2 Datasets pra cá (Section 3 Method é onde
protocolo experimental pertence — convenção académica). Incorporar
nestes 2 pontos quando draftar #59:

1. **Primary protocol resize 28x28 grayscale (compat encoders Omniglot).**
   Texto base movido:
   > "In our primary protocol all images, regardless of source, are resized
   > to 28x28 and converted to grayscale. This preserves architectural
   > compatibility with the source-trained encoders' input shape (1, 28, 28).
   > The cost is severe: 500x500 RGB photographs reduced to 28x28 grayscale
   > lose nearly all texture and color information that conventionally
   > distinguishes bird species. We report the impact of this resolution
   > choice in Section 4 via a parallel 84x84 RGB retrained baseline."

2. **"Extreme task differences" framing (Phoo & Hariharan 2021).**
   Texto base movido:
   > "The visual distance between Omniglot and CUB-200-2011 in any
   > reasonable embedding metric is large. We treat this as an instance
   > of the 'extreme task differences' regime~\citep{phoo2021self} and
   > characterize what survives transfer in that regime."

Conectar em §3.4 (cross-domain protocol) ou §3.5 (sparsity sweep) conforme
fluxo do texto na hora de draftar.
