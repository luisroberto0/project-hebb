# Section 5: Discussion

> Status: PLACEHOLDER. Draft pra sessão #61.
> Word count target: 700-900 words.

---

\section{Discussion}
\label{sec:discussion}

[Draft #61. Estrutura planejada:]

- §5.1 Why k-WTA effect collapses cross-domain (hipótese mecanística)
- §5.2 Anti-transfer mechanism (encoder Omniglot é "anti-transfer")
- §5.3 Implications for bio-plausible learning
- §5.4 Comparison with literature (Tseng 2020, Phoo & Hariharan 2021)
- §5.5 Limitations
- §5.6 Future work

---

## Pontos-chave (anotações pra #61)

### §5.1 Why k-WTA collapses

- Encoder Omniglot aprende features hyper-especializadas em traços binários
- Em CUB, sucessivos MaxPools (28→14→7→3→1) destroem informação visual fina
- Pixel direto preserva mais info útil que CNN-4 forwarding (achado contraintuitivo!)
- Quando representação é tão degradada que pixel direto é melhor, sparsity vira invisível

### §5.2 Anti-transfer

- Encoder treinado ≈ random encoder cross-domain (delta +0.18 p.p., ICs sobrepostos)
- Pattern consistente com Phoo & Hariharan 2021 (STARTUP)
- Treino em fonte muito distante INTRODUZ viés que não generaliza
- Não é "neutral transfer", é "anti-transfer" — pode degradar abaixo de baseline pixel

### §5.3 Implications bio-plausible

- Sparsity é compatível in-domain (paper C3) — fato estabelecido
- Sparsity é neutra cross-domain extreme — fato deste paper
- Não é tóxica, não é benéfica — é INVISÍVEL
- Não invalida bio-plausible learning; refina o escopo de aplicabilidade

### §5.4 Literature comparison

- Tseng 2020: ProtoNet baseline mini-ImageNet→CUB = 38% (5w1s)
- Marco 2-A: ProtoNet baseline Omniglot→CUB = 22% (5w1s)
- Setup mais extremo (binary chars vs RGB textures), sinal mais fraco
- Confirma escala de "extreme task differences" (Phoo & Hariharan 2021)

### §5.5 Limitations

- Single source dataset (Omniglot)
- Single target dataset (CUB-200)
- CNN-4 architecture (não testou ResNet, ViT)
- 28×28 grayscale como input principal (84×84 RGB usado apenas no baseline retreinado)
- Não testou self-training na target (STARTUP-style)
- Não testou k-WTA em camadas intermediárias

### §5.6 Future work

- Source domain mais próxima (mini-ImageNet→CUB)
- k-WTA em camadas intermediárias (não só embedding final)
- Self-training na target (STARTUP integration)
- Architecture variants (ResNet, ViT)
- Multi-target (CUB, Cars, Places, Plantae) para confirmar generalidade do collapse
- Comparison com explicit feature-wise transformations (Tseng 2020)

## Tom recomendado

- Direto, técnico
- Explicações mecanísticas por hipótese (não certeza)
- "may explain", "consistent with", "suggests" — não "proves", "demonstrates"
- Reconhecer limitações antes de reviewer apontar
