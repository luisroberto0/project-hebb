# Experimento 03 — Cross-domain few-shot (Marco 2-A)

> **Status:** kickoff em sessão #52 (2026-05-14). Sem código rodado nesta sessão.
> **Limite hard:** 15 sessões (#52-#66). Admin obrigatória em #66.

---

## Posicionamento na missão pós-LLM

CONTEXT.md §1 lista 4 capacidades pós-LLM como missão central do Project Hebb. Marco 2-A ataca especificamente uma sub-capacidade implícita: **one-shot inédito** — formar representação durável a partir de 1 exemplo em **domínio visual radicalmente diferente** do treino.

One-shot em Omniglot (Marco 1, sessão #20: C3b 93.10% 5w1s) é caso degenerado — todos os caracteres são da mesma família visual (binary strokes 28×28). Generalizar para um domínio diferente (RGB textures naturais, alta resolução) é o teste real de "one-shot inédito".

LLMs não fazem isso bem out-of-the-box. Project Hebb tem motivo legítimo para atacar essa capacidade.

---

## Pergunta científica

> **"C3 (CNN-4 + k-WTA k=16, treinado em Omniglot na sessão #20, weights congelados) aplicado em CUB-200-2011 cross-domain few-shot — bate ProtoNet retreinado em CUB-200 por ≥5 p.p. em ACC 5-way 1-shot? Caso contrário, qual o gap absoluto e o que ele revela sobre transferência de features bio-inspiradas entre domínios visuais drasticamente diferentes (binary chars 28×28 → RGB textures ~500×500)?"**

### Outcomes informativos

1. **Critério atingido (≥+5 p.p.):** evidência surpreendente de que k-WTA aprende features de baixo nível mais universais que ProtoNet denso. Justifica paper de exploração positiva.
2. **Critério não atingido:** caracterização do gap. Achado negativo defensável e útil — informa limites de transfer learning bio-inspirado pra missão pós-LLM.

---

## Predição provisional (registrada antes do experimento)

Baseada em literatura cross-domain few-shot (ver `PAPERS.md`):

| Modelo | ACC 5w1s CUB esperado |
|---|---|
| Pixel kNN cross-domain (sanity floor) | 22-28% |
| C3 (encoder #20 Omniglot, weights congelados) | **20-40%** |
| ProtoNet baseline (encoder Omniglot, weights congelados) | 25-45% |
| ProtoNet **retreinado** em CUB-200 (baseline a bater) | 60-75% |

**Gap esperado:** C3 cross-domain está **−25 a −50 p.p. ABAIXO** de ProtoNet retreinado, **não acima**.

Critério literal "C3 ≥ ProtoNet retreinado + 5 p.p." → predição: **vai falhar**.

Razões:
- Omniglot é binary 28×28 com estatística visual completamente distinta de RGB natural; distance shift é maior que setups típicos da literatura (mini-ImageNet→CUB já produz 35-50% típico).
- Resize CUB pra 28×28 destrói informação visual fina (textures, pattern, color) que distingue bird species.
- C3 encoder otimizado pra Omniglot pode ter features hyper-especializadas em traços binários, com transfer ainda pior que ProtoNet baseline.

---

## Dataset: CUB-200-2011

**Fonte oficial:** Caltech-UCSD Birds-200-2011 (Wah, Branson, Welinder, Perona, Belongie 2011).

| Característica | Valor |
|---|---|
| Classes | 200 (bird species) |
| Imagens totais | ~11.788 |
| Imagens por classe | ~60 (variável) |
| Resolução nativa | ~500×500 (variável) |
| Canais | RGB |
| Annotations disponíveis | bounding boxes, part locations, attribute labels (não usados nesta passada) |
| Splits cross-domain few-shot padrão | 100 train / 50 val / 50 test (Hilliard 2018, Chen 2019) |

### Disponibilidade

- `torchvision.datasets` **NÃO tem CUB-200 nativo** (até cutoff jan/2026). Precisará download manual.
- Site oficial: http://www.vision.caltech.edu/datasets/cub_200_2011/ — historicamente offline ocasionalmente.
- Mirrors comuns: Hugging Face datasets (`Multimodal-Fatima/CUB_train`, `efekankavalci/CUB_200_2011`), Kaggle, repos GitHub. **Confirmar legitimidade do mirror antes de usar.**
- Fallback: dataset original tem ~1.1 GB (tar). Cache local em `experiment_03_crossdomain/data/CUB_200_2011/`.

### Decisão de pré-processamento (preservar comparabilidade com C3)

**Primeira passada (sessões #53-#58):**
- Resize para **28×28**
- Convert para **grayscale** (mesma dimensão de input que C3 Omniglot, `(B, 1, 28, 28)`)
- Normalize igual ao Omniglot pipeline (`config.py` defaults)
- Splits cross-domain padrão (Chen 2019 split — 100/50/50, eval no test split)

**Limitação registrada explicitamente:** resize 500×500→28×28 grayscale destrói praticamente toda informação visual fina que distingue bird species. Esperado que ProtoNet retreinado também sofra (provavelmente cai para 40-55% em vez de 60-75% típico de literatura). Manter comparabilidade arquitetural com C3 é prioridade nesta passada; resolução maior fica pra Marco 2-A extensão (#61-#62 se justificável).

**Segunda passada (#61-#62 condicional):**
- Se #58 mostrar gap massivo (>30 p.p.), testar resolução 84×84 RGB (resolução padrão few-shot literature) para checar se gap diminui.
- Implica adaptar primeira layer do CNN-4 (`Conv2d(1, 64)` → `Conv2d(3, 64)`) e retreinar — viola comparabilidade direta com C3 mas pode informar mecanismo.
- Decidir em #60 se vale.

---

## Métrica primária

- **ACC 5-way 1-shot** em CUB-200 test split, IC95% bootstrap
- **ACC 5-way 5-shot** como métrica secundária (caracteriza curva data-efficiency)
- 1000 episódios independentes amostrados do test split
- 5 seeds (consistente com baselines anteriores)

### Comparações (todas em CUB-200 test split)

| Modelo | Encoder weights | Treinado onde |
|---|---|---|
| **Pixel kNN cross-domain** | — | — (sanity floor) |
| **ProtoNet baseline cross-domain** | encoder ProtoNet Omniglot (sessão #20, `baselines.py`), congelado | Omniglot |
| **C3 cross-domain (foco)** | encoder C3b Omniglot (sessão #20, k=16, 75% sparsity), congelado | Omniglot |
| **ProtoNet retreinado em CUB-200** | encoder ProtoNet, treinado from scratch em CUB-200 train split | CUB-200 |

**Baseline a bater:** ProtoNet retreinado em CUB-200. Critério: C3 cross-domain ≥ ProtoNet retreinado + 5 p.p. em ACC 5w1s.

---

## Critério literal de fechamento

| Resultado (5 seeds, IC95% bootstrap) | Decisão |
|---|---|
| C3 cross-domain ≥ ProtoNet retreinado + 5 p.p. ACC 5w1s | **Sucesso** → paper de exploração positiva (workshop-scope) |
| C3 dentro de ProtoNet retreinado ± 5 p.p. (qualquer direção) | **Mediano** — admin obrigatória em #66 decide entre paper de caracterização ou encerramento |
| C3 < ProtoNet retreinado − 5 p.p. (predição confirmada) | **Achado negativo defensável** → paper de exploração negativa ou apêndice. Marco 2-A encerrado |

Critério não-negociável. Mudanças de scope (encoder maior, resolução maior, dataset diferente) viram **Marco 2-A.2** (extensão), não substituição.

---

## Plano de sessões #52-#66

| # | Tipo | Output esperado |
|---|---|---|
| **52 (esta)** | Admin + lit review | STRATEGY/CONTEXT update + PLAN+PAPERS criados |
| 53 | Code | Download CUB-200 + dataloader + smoke test (1 episódio) |
| 54 | Code | C3 cross-domain eval + ProtoNet baseline cross-domain (encoders Omniglot congelados → CUB) |
| 55 | Code | Pixel kNN cross-domain |
| 56 | Code | ProtoNet retreinado em CUB-200 (treino + eval) — baseline a bater |
| 57 | Analysis | Comparação cabeça-a-cabeça com IC95% bootstrap. Critério literal? |
| 58 | Analysis | Verificação multi-seed + reprodutibilidade |
| 59 | Code | Caracterização: 5w1s vs 5w5s, sweep n_shots ∈ {1,3,5,10} |
| 60 | Analysis | Análise por classe — quais bird species C3 acerta vs erra? Decisão sobre #61-#62 |
| 61-62 | Code (condicional) | Se #60 justificar: resolução 84×84 RGB ou ResNet backbone (Marco 2-A.2) |
| 63-65 | Writing | Análise final + paper de exploração draft |
| **66** | **Admin obrigatória** | Critério atingido? → paper. Não? → Marco 2-A encerrado, decidir Marco 2-B/C ou encerrar projeto |

Cancelable em qualquer ponto se evidência for clara antes de #66 (ex: #57 já mostra C3 < 30% e ProtoNet retreinado > 70% com IC apertado — não precisa esperar 9 sessões).

---

## Restrições

- **Não modifica scripts existentes** (`c3_protonet_sparse.py`, `baselines.py` congelados como referência reproduzível pra paper C3)
- **Não modifica `paper_c3/`** (paper finalizado pendente publicação separada via LinkedIn)
- **Encoder C3 da sessão #20 = referência canônica.** NÃO retreinar C3 em CUB nesta versão do marco.
- **Resolução de input fixa em 28×28 grayscale** na primeira passada (preserva comparabilidade arquitetural com C3).
- **Sem cadência fixa** (igual decisão (c) Confirmação Pós-#21).

---

## Riscos e como mitigar

| Risco | Mitigação |
|---|---|
| CUB-200 site oficial offline no momento | Mirror Hugging Face/Kaggle. Validar checksum se possível. |
| Checkpoint da sessão #20 (C3b) não estar reproduzível | Verificar `experiment_01_oneshot/checkpoints/` em #53. Se ausente, retreinar via `c3_protonet_sparse.py --device cuda` (~6 min) antes do experimento cross-domain |
| Eval em 28×28 grayscale destruir tanta informação que ProtoNet retreinado fica em chance (20%) | Documentar como achado: "even retraining ProtoNet in CUB-28×28-gray fails — visual texture is fundamentally lost". Justifica passada 84×84 condicional em #61-#62. |
| Setup adversarial Omniglot→CUB não tem precedente claro na literatura | Documentar como contribuição metodológica. Comparar com setups mais próximos (mini-ImageNet→CUB de Chen 2019, ImageNet→CUB de Tseng 2020). |

---

## Referências

Lit review estruturada em `PAPERS.md` (5 papers core):
- Triantafillou et al. 2020 — Meta-Dataset
- Tseng et al. 2020 — Cross-Domain Few-Shot via FWT
- Chen et al. 2019 — A Closer Look at Few-Shot Classification
- Phoo & Hariharan 2021 — STARTUP (extreme task differences)
- Wah et al. 2011 — CUB-200-2011 dataset paper
