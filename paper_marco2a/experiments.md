# Section 4: Experiments

> Status: PLACEHOLDER. Draft pra sessão #59-#60.
> Word count target: 800-1000 words.

---

\section{Experiments}
\label{sec:experiments}

[Draft #59-#60. Estrutura planejada:]

- §4.1 Setup: hardware (RTX 4070), wall-clock total
- §4.2 Cross-domain sweep table (7 condições) + análise estatística (ICs sobrepostos)
- §4.3 In-domain vs cross-domain comparison (k-WTA effect collapse)
- §4.4 Bottleneck decomposition (random→C3→retreinado 28×28→retreinado 84×84)
- §4.5 Anti-transfer evidence (encoder treinado ≈ random)
- §4.6 Pixel kNN dominates encoded representations

---

## Tabelas a incluir (de WEEKLY-1.md)

**Tabela 1 (principal):** 7 condições cross-domain × ACC × IC95%
- ProtoNet retreinado CUB 84×84 RGB: 49.84% [49.38, 50.59]
- ProtoNet retreinado CUB 28×28 gray: 34.31% [34.06, 34.55]
- Pixel kNN: 22.81% [22.69, 22.97]
- C3 k=32: 22.20% [21.77, 22.57]
- C3 k=64 (=ProtoNet baseline): 22.13% [21.90, 22.36]
- C3 k=16: 22.09% [21.84, 22.34]
- Random encoder + k-WTA k=16: 21.91% [21.76, 22.03]
- C3 k=8: 21.68% [21.34, 22.04]
- chance: 20.00%

**Tabela 2:** in-domain (paper C3) vs cross-domain
- k=8: 90.77% / 21.68%
- k=16: 93.10% / 22.09%
- k=32: 93.35% / 22.20%
- k=64: 94.55% / 22.13%
- spread k=8 vs k=64: 3.78 in / 0.52 cross (k-WTA effect collapse)

**Tabela 3 (opcional):** decomposição bottleneck
- Random encoder → C3: +0.18 p.p. (treino fonte distante: irrelevante)
- C3 → ProtoNet retreinado 28×28: +12.22 p.p. (treino na target)
- Retreinado 28×28 → 84×84 RGB: +15.53 p.p. (resolução + canais)

## Figuras a gerar (a fazer #60)

- Fig 1: bar chart 7 condições + chance line + IC95% error bars
- Fig 2: dual-panel in-domain vs cross-domain (visual collapse)
- Fig 3 (opcional): waterfall chart bottleneck decomposition

## Notas pra #59

- Tom: descritivo + análise concisa. Não repetir interpretação (vai em §5 Discussion).
- Incluir z-scores nas tabelas (já tem em WEEKLY-1.md)
- Confirmar todos números batem com WEEKLY-1.md
