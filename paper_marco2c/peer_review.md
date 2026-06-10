# Peer review interno — paper Marco 2-C

Revisão adversarial multi-agente (workflow `marco2c-peer-review`, 4 lentes + síntese), verificando os números contra `experiment_05_temporal/` e o código `temporal_bench.py`. Resultado consolidado abaixo.

## Veredicto

**Publicável como workshop paper, contingente a uma reframe de honestidade** (o controle bins=1) — que foi **aplicada**. Fact-check: todos os números corretos; zero erros numéricos.

## Achado dominante (3 reviewers convergiram, verificado no código) — APLICADO

**O controle bins=1 não isola "timing vs arquitetura" como o draft afirmava.** Em `temporal_bench.py` (`SNN_Rec.forward`), em T=1 o termo recorrente `self.rec(s1)` é identicamente zero (`s1` inicializado em zeros, loop de 1 iteração). Logo a SNN-rec em T=1 é uma rede feedforward de passo único degenerada — a recorrência (feature definidora) é **inerte**. Consequências corrigidas no paper:
- "architecture is held fixed" **retratado** (era falso).
- +10.18 p.p. reframado como **upper bound** que mistura timing com a recorrência ficando operativa.
- +9.51 p.p. reframado como resíduo da arquitetura **feedforward** (não recorrente).
- Citado o gap ff-vs-rec (+10.26 p.p., Tabela 1) como bound frouxo da contribuição da recorrência.

## Safe fixes aplicados (8)

1. Linha §4.1: desambiguada a frase elidida ("exceeds the feedforward SNN by 10.26") — o número estava certo (rec−ff), só a leitura arriscava confusão.
2. Fronteira 3-seed/5-seed explicitada na decomposição (19.68 reproduz 19.71 a 0.03 p.p.) + footnote na Tabela 2 (budget 3 seeds/8 epochs vs 5 seeds/15 epochs).
3. Figura 1 agora referenciada no texto (`\ref{fig:conditions}` — antes órfã).
4. "percentage points" → "p.p." unificado (abstract + conclusion).
5. `\textasciitilde` → `$\sim$` (abstract).
6. Caption da Tabela 1: `$\pm$` = SD entre seeds; brackets = IC95% bootstrap (assimétricos).
7. refs.bib: TODO do Cramer 2022 resolvido.
8. SSC single-seed disclosed (§3.5, §4.5, Limitations vi).

## Judgment calls — todos aplicados como suavização de honestidade

(Tom/claim; alinham ao ethos do projeto, então aplicados sem mudar escopo:)
- **SSC "weak but stable" → "weak and training-dependent"** (single seed; +5.20→+3.95 erode); "ceiling not undertraining" → "suggests but does not establish".
- **k-WTA "general in-domain property" → "consistent with, não estabelece"**; `pinho2026kwta` explicitamente marcado como trabalho próprio não-publicado; "to within noise"/"quantitative match" → "suggestive coincidence".
- **"reads onset timing" → "exploits information beyond binary channel presence, consistent with using onset timing".**
- **Baseline recorrente não-spiking (GRU/RNN) adicionado às Limitations** como o controle que mais falta (timing spiking-específico vs genérico de recorrência).
- **Redundância da intro** cortada (enumerate de contribuições → 1 frase; mantida a lista "Main findings" (i)-(v)).

## O que fica pendente (decisão do autor, não de execução)

- **Venue:** workshop (NeurIPS/ICLR bio-plausible/neuromorphic) vs LinkedIn-PT (como C3) vs arquivar (como 2-A).
- **Opcional (novo experimento):** 3 seeds no SSC + 1 GRU baseline endureceriam os dois pontos mais fracos — mas não são necessários para o claim atual (que já está honestamente delimitado).

Validação estrutural pós-edições: **TUDO OK** (5 cites, 9 refs→labels, 13/13 ambientes, 193/193 braces, 3 figuras).
