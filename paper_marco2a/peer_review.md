# Peer review interno — paper Marco 2-A (#64–65)

> Rodado em #65 (2026-06-08) via painel adversarial multi-agente: 4 reviewers independentes (verificação numérica, overclaim, copyediting, program-committee) + síntese. Fonte revisada: `main.tex` contra `experiment_03_crossdomain/WEEKLY-1.md`.

## Verdict

**Pequenos ajustes.** O paper está sólido e honesto; **todos os números das Tabelas 1 e 2 conferem exatamente com a fonte** (verificação exaustiva). Nenhum erro invalida os resultados centrais. Foram encontrados ~10 defeitos de **consistência interna** (seguros, já corrigidos) e ~16 **decisões de julgamento** (escopo/tom/claim) que cabem ao autor.

## Status #66 — ajustes incorporados

Luis (#66) optou por incorporar os ajustes de **alta + média prioridade**, mantendo o título, e **encerrar o Marco 2-A** (arquivar sem publicar). Aplicados em `main.tex` + `.md` fonte (commit do #66):

- ✅ **A (escopo):** abstract, intro (contrib + parágrafo final) e conclusão escopados para o par único "(Omniglot→CUB, CNN-4, embedding-level k-WTA)". Título mantido.
- ✅ **B (estatística):** "statistically indistinguishable/negligible" → "within overlapping 95% bootstrap CIs / within seed-to-seed variation" (abstract, intro, §4.3, §4.5, conclusão).
- ✅ **C (Phoo):** §5.4 ganhou caveat de shot-count (Phoo é 5w5s; nosso 5w1s é mais difícil → collapse mais severo).
- ✅ **D (max-pool):** intro suavizada para casar com §5.1; §5.1 concede o confounder do random-encoder e hedga "near-degenerate".
- ✅ **E (termos):** "effect collapse" e "anti-transfer" definidos operacionalmente na 1ª ocorrência (sem implicar mecanismo causal).
- ✅ **F (STARTUP):** §5.2 explicita que o +12.22 é retreino **supervisionado**, não self-training; framing STARTUP movido para escopo de future work. §4.5 suavizada ("may be required").
- ✅ **G (input fidelity):** "input resolution" → "input fidelity (resolution and color)" em abstract/§4.3/conclusão.

**Não aplicados (baixa prioridade, item H/I):** merge §5.5/§5.6, poda de redundância numérica, "maximally distant" (já suavizado em F), frase sobre self-citation não-publicada, defesa explícita de n=5 seeds, framing "is this just a negative result". Ficam registrados aqui caso o paper seja retomado.

---

## ✅ Correções seguras aplicadas (#65)

Aplicadas no `main.tex` e replicadas nos `.md` fonte. Factual/consistência/notação, alta confiança, não mudam escopo/claims/tom:

1. **§4.2** — `+12.22 p.p. over the best source-trained encoder (k=32)`: o delta +12.22 é vs **k=16** (34.31−22.09), não k=32 (que daria +12.11). Reformulado para `over the k=16 source-trained encoder` (remove o "best" ambíguo e alinha com a decomposição §4.4). *[erro de rótulo confirmado contra a fonte]*
2. **§4.5** — "the random CI **is contained within** the k=16 CI": geometricamente falso (21.76 < 21.84). Trocado por **"overlaps"** + "well within seed-to-seed variation". Conclusão de indistinguibilidade preservada.
3. **§4.1** — "the **seven** characterized conditions" → **"eight"** (Tabela 1 tem 8 linhas-modelo; Fig 1 já dizia "eight"). *[a própria fonte escrevia "7" mas tabulava 8]*
4. **Caption Tabela 1** — definia coluna "Frozen" inexistente → `"Omniglot frozen" marks encoders whose weights are not updated`.
5. **Tabela 2** — rótulo `Spread k=8 vs k=64` estava aritmeticamente errado para a coluna cross (k=8 vs k=64 = 0.45, não 0.52). Trocado por **`Spread (max − min)`**, casando com a caption.
6. **§2.2** — k-set listado `{32,16,8}` mas citava 94.55% (k=64) → corrigido para `{64,32,16,8}` (0/50/75/87.5%); de quebra inclui o **93.35% (k=32)** que era órfão na Tabela 2.
7. **Eq. (1)** — operador `\text{kWTA}` → `\text{k-WTA}` (casa com a prosa/título).
8. **Floats** — `[h]` → `[tbp]` nos 5 floats (robustez LaTeX; idealmente confirmar no Overleaf).
9. **Notação** — `percentage points` → `p.p.` consistente (mantida a forma completa só na 1ª ocorrência, no abstract).

---

## ⏳ Decisões de julgamento (suas — não apliquei)

Nenhuma exige novo experimento (exceto onde indicado). Agrupadas por tema, com a recomendação convergente dos reviewers. **Prioridade alta** = vários reviewers convergiram e/ou risco de rejeição.

### A. Escopo / generalização single-pair — **prioridade alta** (R2 + R4)
Título ("When Sparsity Stops Mattering"), abstract, intro e conclusão generalizam para "extreme domain shift" e "bio-inspired sparsity" em geral, mas a evidência é **1 par** (Omniglot→CUB), 1 backbone (CNN-4), 1 mecanismo (k-WTA final), 1 setting (5w1s). §5.5 já restringe; as manchetes derrubam a restrição.
→ *Rec.:* escopar abstract/intro/conclusão para "this single-pair regime / Omniglot→CUB, CNN-4". Mexer no **título** é decisão sua.

### B. Linguagem estatística sem teste formal — **prioridade alta** (R2 + R4)
Todo "statistically indistinguishable/negligible" repousa em **sobreposição de IC visual** (n=5 seeds); nenhum teste formal foi rodado. ICs sobrepostos não implicam não-significância.
→ *Rec.:* (a) downgrade de linguagem para "within overlapping 95% bootstrap CIs"; **ou** (b) rodar um teste pareado (Welch/Mann-Whitney) sobre os 5 means por seed já coletados — zero runs novos, endurece o claim central. *(b) exige recuperar os per-seed means dos logs.*

### C. Comparação Phoo 5w5s vs nosso 1-shot — **prioridade alta** (R4)
O paper é todo 5w1s, mas ancora no "22–40% range" de Phoo que é **5w5s**. A coincidência numérica (22%) é enganosa (5-shot é mais fácil). A comparação Tseng 38% é corretamente 5w1s (justa).
→ *Rec.:* adicionar 1 cláusula onde Phoo é citado: "(Phoo reporta 5w5s; nosso 5w1s é mais difícil, logo um piso comparável aqui representa um collapse mais severo)" — converte a inconsistência em claim fortalecido.

### D. Claim mecanístico max-pooling — **prioridade média** (R2 + R4)
Intro (sem hedge) "successive max-poolings... destroy" vs §5.1 (com hedge) "appear to discard" → inconsistência de tom. Além disso, o controle random-encoder mostra que o déficit é do **pipeline CNN-4 inteiro**, não especificamente do max-pooling. O embedding "near-degenerate" é afirmado sem medição.
→ *Rec.:* suavizar a intro para casar com §5.1; conceder o confounder do random-encoder; hedgar "near-degenerate".

### E. Termos de marca: "effect collapse" / "anti-transfer" — **prioridade média** (R2 + R4)
"effect collapse" implica "efeito existiu e foi destruído"; o par único não distingue isso de "nunca se manifestou numa representação nula". "anti-transfer" sugere transfer negativo, mas a evidência é treinado ≈ random (+0.18, ICs sobrepostos).
→ *Rec.:* manter os termos mas **defini-los operacionalmente** na 1ª ocorrência (collapse = redução do spread 3.78→0.52; anti-transfer = ausência de transfer positivo) sem implicar mecanismo causal.

### F. STARTUP citado 4× mas nunca rodado — **prioridade média** (R4)
STARTUP é invocado como "o que bridge the gap", mas os baselines retreinados usam treino **supervisionado**, não self-training. O +12.22 é de retreino supervisionado.
→ *Rec.:* escopar que +12.22 é supervisionado, que self-training não foi avaliado, e mover o framing STARTUP estritamente para Future Work.

### G. Fator "+15.53 input resolution" confunde resolução + cor — **prioridade média** (R2 + R4)
O passo 28×28-gray → 84×84-RGB muda **duas** variáveis (resolução E cor) + AdaptiveAvgPool. Abstract/§1/conclusão rotulam como "resolution" sozinho.
→ *Rec.:* usar "input fidelity (resolution + color)" consistente (Fig 3 e §4.3 já usam), ou caveat de confound.

### H. Editorial / budget de palavras — **prioridade baixa** (R3)
- Fundir §5.5 Limitations + §5.6 Future Work (espelham-se quase verbatim; ~80–120 palavras).
- Os 4 valores per-k aparecem em abstract + intro + §4 + conclusão; manter só nas tabelas (~50–70 palavras).
- §4.5 faz trabalho de Discussion (interpretação anti-transfer/STARTUP); deixar §4.5 report-only.

### I. Outras (baixa) 
- "maximally distant" / "empirical lower bound" (§5.2) — suavizar (não estabelecido).
- Self-citation `pinho2026kwta` não-publicada sustenta o anchor in-domain — frase reconhecendo dependência.
- Defesa de n=5 seeds — frase notando que cada seed agrega 1000 episódios (std inter-seed <1 p.p.).
- Framing "is this just a negative result?" — liderar contribuição com os achados não-triviais (anti-transfer perde pra pixel kNN).

---

## Estado pós-#65

Paper draft-completo, números verificados, consistência interna corrigida. **Pronto para #66 (admin)** — a decisão sobre quais judgment calls incorporar e onde/se publicar é do autor.
