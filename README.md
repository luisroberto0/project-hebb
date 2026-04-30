# Project Hebb

Pesquisa em arquiteturas neurais bio-inspiradas: plasticidade local diferenciável, codificação esparsa, e few-shot learning. Estudo empírico em direção a arquiteturas cognitivas pós-LLM, usando Omniglot como benchmark. Workshop paper em preparação.

> *"Não tente construir a mente. Construa um neurônio que funcione diferente."*
> — Luis Roberto Pinho da Silva Junior, Project Hebb (2026)

---

## Status (atualizado em 2026-04-29, pós-sessão #30)

**Fase 1 — Fundação concluída** (30 sessões). Project Hebb teve 2 experimentos:
- **Experimento 01 (one-shot Omniglot):** ✅ atingiu metas numéricas via Caminho C (ProtoNet + k-WTA esparso). Vira paper de workshop.
- **Experimento 02 (continual learning sem replay):** ❌ Marco 1 encerrado pelo critério literal (sessão #29). 4 abordagens testadas, todas ≤ baseline naive. Achado mecanístico documentado.

### Headline result (Omniglot, 1-shot)

Resultado central do paper em preparação:

| Modelo | 5-way 1-shot | 20-way 1-shot | Sparsity |
|---|---|---|---|
| ProtoNet baseline (Snell 2017, replicado) | **94.55%** | — | 0% (denso) |
| **C3a — ProtoNet + k-WTA k=32** | **93.35%** | **81.87%** | 50% |
| **C3b — ProtoNet + k-WTA k=16** | **93.10%** | **80.72%** | **75%** |
| **C3c — ProtoNet + k-WTA k=8** | **90.77%** | **75.44%** | 87.5% |
| Random encoder + k-WTA (validação) | 37.60% | 16.73% | 75% (sem treino) |

Sparsity de 75% custa apenas −1.45 p.p. vs baseline ProtoNet completo. Validação random+kWTA confirma que o ganho vem do treino sob restrição, não da estrutura k-WTA estática.

### Contexto experimental completo (Omniglot 5-way 1-shot)

Trajetória de 30 sessões pra chegar ao C3b:

| Modelo | ACC 5w1s | Sessão | Notas |
|---|---|---|---|
| Pixel kNN (baseline trivial) | 45.76% | #7 | nearest-neighbor sobre pixels |
| Iter 1 STDP saturado (melhor de 13 sessões) | 35.98% | #9 | barreira estrutural documentada (#1-#13) |
| C1b — PCA-32 + Hopfield Memory (sem treino) | 56.28% | #15 | baseline arquitetural surpreendente |
| C2-simplified — plasticidade meta-aprendida | 64.08% | #19 | melhor pré-CNN |
| **C3b — ProtoNet + k-WTA 75% sparsity** | **93.10%** | **#20** | **target paper** |
| ProtoNet baseline (denso) | 94.55% | #20 | upper bound |

C3b atinge as duas metas numéricas do `CONTEXT.md` §4 (≥90% 5w1s, ≥70% 20w1s) com mecanismo neural-inspirado defensável (esparsidade biológica). Não cumpre a restrição mecanística "sem backprop end-to-end" — registrado honestamente em `CONTEXT.md` §1.2 e na discussão do paper.

### Próximo passo

**Paper C3:** "k-WTA Sparsity Preserves Prototypical Network Performance in Few-Shot Learning". Submissão alvo NeurIPS Bio-Plausible Learning Workshop ~setembro 2026. Cronograma: 5 sessões de paper writing (~10-15h reais). Ver `STRATEGY.md` "Plano paper C3".

**Status paper writing:** sessão #36 (2026-04-30) — **decisão final: NÃO submeter NeurIPS Workshop, publicar via LinkedIn em PT.** Razão: founder Rytora sem tempo pra rebuttals/revisões/registration; LinkedIn alcança parte do que peer review faria sem overhead institucional. Paper draft completo preservado: abstract + 6 seções (3789 palavras) + 2 figuras 300 DPI + main.tex compilável + refs.bib (14 entradas) + LinkedIn post drafts (longo + curto). PDF gerável via Overleaf em ~10 min. Ver `STRATEGY.md` "Decisão pós-#35: LinkedIn em vez de NeurIPS" e `paper_c3/README.md` "Final state". Project Hebb entra em estado de manutenção pós-#36.

---

## Quick start

**Hardware testado:** Notebook i9 + RTX 4070 (compute capability 8.9, Ada Lovelace), CUDA 12.4, Windows 11. Roda em CPU também (mais lento), basta omitir `--device cuda`.

```powershell
# 1. Venv + PyTorch CUDA (ordem importa: PyTorch primeiro, com index URL)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. Resto das dependências core (causalnex/dowhy/pyro são opcionais, podem
#    falhar em Python 3.13 — ignorar warnings e seguir)
pip install scipy matplotlib seaborn pandas scikit-learn tqdm einops snntorch brian2 brian2tools

# 3. Validar ambiente
python validate_environment.py
# Esperado: PyTorch 2.6+, CUDA disponível, RTX 4070 detectada
```

### Demonstração — replica o resultado central em ~6 min

```powershell
# No Windows, setar PYTHONIOENCODING=utf-8 evita UnicodeEncodeError no console cp1252
$env:PYTHONIOENCODING = "utf-8"
cd experiment_01_oneshot

# C3 — 3 níveis de sparsity, 5w1s + 20w1s, 1000 episódios eval, IC95% bootstrap
python c3_protonet_sparse.py --device cuda --train-episodes 5000 --eval-eps 1000 --seed 42
# Esperado: C3a k=32 → 93.35% / C3b k=16 → 93.10% / C3c k=8 → 90.77% (5w1s)
#           Random+kWTA validation → 37.60%
```

Para baselines de referência:

```powershell
python baselines.py --baseline pixel_knn --ways 5 --shots 1 --episodes 1000
# Esperado: 45.76%

python baselines.py --baseline proto_net --ways 5 --shots 1 --episodes 1000 --train-episodes 5000
# Esperado: 94.55% (com 500 train_episodes vai pra ~85.88%, smoke)
```

---

## Arquitetura — Pipeline C3 (target paper)

Pipeline publicável (`experiment_01_oneshot/c3_protonet_sparse.py`):

```
Imagem 28×28 (Omniglot, downsample de 105×105, fundo invertido)
    ↓
ProtoEncoder CNN-4 (Snell et al. 2017)
  4 blocos × Conv-BN-ReLU-MaxPool, 64 filtros, kernel 3
    ↓
Embedding 64D (após 4 maxpools 2×2)
    ↓
k-WTA top-k (k=16 → 75% sparsity; treino e eval consistentes)
    ↓
Prototype-based classifier
  Centróide por classe sobre support set
  cdist² → softmax → cross-entropy
    ↓
Predição N-way K-shot
```

**Características:**
- Encoder treinado end-to-end via SGD (Adam lr=1e-3, 5000 episodes).
- k-WTA aplicado em training E eval — gradient flui pelos top-k channels, encoder aprende a colocar info nos k dimensões dominantes.
- Validação com encoder random + k-WTA sem treino (37.60%) confirma contribuição do treino sob restrição de sparsity.

### Família STDP (exploração documentada, sessões #1-#13)

Pipeline original baseado em STDP biofísico convolucional (`experiment_01_oneshot/model.py:STDPHopfieldModel`) atingiu teto em 35.98% (one-shot), com barreira estrutural caracterizada (matched filter trivial). Mantido no repo como referência. Detalhes em `experiment_01_oneshot/WEEKLY-1.md` e `WEEKLY-2.md`. Caminho C (#15-#20) substituiu STDP biofísico por CNN-4 + plasticidade meta-aprendida + k-WTA esparso.

---

## Documentos principais

Ordem de leitura para entender o estado em <10 minutos:

| Doc | O que tem |
|-----|-----------|
| [`CONTEXT.md`](CONTEXT.md) | Mission, princípios operacionais, stack, baselines, papers. §1.1 Refino #21 (plasticidade local diferenciável), §1.2 Refino #30 (Marco 1 encerrado). **Sempre primeiro.** |
| [`PLAN.md`](PLAN.md) | Plano operacional vivo. Notas de iteração (30+ sessões), Decisões arquiteturais permanentes. |
| [`STRATEGY.md`](STRATEGY.md) | Estratégia de pesquisa: decisões pós-#10, #13, #15, #20, #25, #27, **#30 (Fechamento Marco 1 + Plano paper C3)**. |
| [`experiment_01_oneshot/`](experiment_01_oneshot/) | Experimento 01 (one-shot). PLAN.md + WEEKLY-1.md (sanity MNIST), WEEKLY-2.md (família C, sessões #15-#20). C3b é o resultado publicável. |
| [`experiment_02_continual/`](experiment_02_continual/) | Experimento 02 (continual learning). PLAN.md marcado ENCERRADO. WEEKLY-1.md (sessões #22-#29). PAPERS.md (lit review CL). |
| [`CLAUDE.md`](CLAUDE.md) | Guia operacional pra futuras sessões Claude Code. |

---

## Estrutura do repositório

```
project-hebb/
├── CONTEXT.md                # Briefing conceitual (mission, princípios, stack)
├── PLAN.md                   # Plano operacional vivo (notas de iteração, decisões)
├── STRATEGY.md               # Estratégia de pesquisa (decisões pós-#10/#13/#15/#20/#25/#27/#30)
├── CLAUDE.md                 # Guia para Claude Code CLI
├── README.md                 # Este arquivo
├── LICENSE                   # MIT
├── requirements.txt
├── environment.yml
├── setup_neuromorfa.md       # Notas de setup inicial (hardware, ambiente)
├── validate_environment.py
├── validate_snn_minimal.py
├── validate_brian2_stdp.py
├── archive/                  # Histórico fechado (BLOCKED.md, SESSION_SUMMARY)
├── experiment_01_oneshot/    # One-shot Omniglot. C3b é resultado publicável (93.10%)
│   ├── PLAN.md, WEEKLY-1.md, WEEKLY-2.md
│   ├── config.py, data.py, model.py
│   ├── train.py, evaluate.py
│   ├── baselines.py                  # Pixel kNN + Prototypical Networks
│   ├── sanity_mnist.py               # Reprodução Diehl & Cook 2015
│   ├── c1_hopfield_baselines.py      # C1: Hopfield + features triviais (#15)
│   ├── c1d_autoencoder_baseline.py   # C1d: Hopfield + AE (#16, descartado)
│   ├── c2_meta_hebbian.py            # C2: plasticidade meta-aprendida (#17)
│   ├── c2_ablations.py               # C2 ablações (#18)
│   ├── c2_simplified.py              # C2 simplificado (#19)
│   ├── c3_protonet_sparse.py         # C3: ProtoNet + k-WTA (#20) ← TARGET PAPER
│   ├── tests/, utils/, figs/sessao_11/
│   └── run_all.ps1
├── experiment_02_continual/  # Continual learning (Marco 1 ENCERRADO em #30)
│   ├── PLAN.md (marcado ENCERRADO no topo)
│   ├── PAPERS.md (lit review EWC/SI/GEM/A-GEM/Hadsell)
│   ├── WEEKLY-1.md (sessões #22-#29)
│   ├── baseline_naive.py             # Naive sequential ProtoNet (#23, #25)
│   ├── c2_continual_arch_a.py        # Scaffold (não implementado)
│   ├── c2_continual_arch_b.py        # Possibilidade B linear (#27)
│   ├── c2_continual_arch_c.py        # Scaffold (não implementado)
│   └── c5e_combined.py               # Caminho 5e kitchen sink (#28-#29)
└── paper_c3/                 # Workshop paper draft (NeurIPS Bio-Plausible ~set/2026)
    ├── README.md             # Overview, target venue, status
    ├── outline.md            # Estrutura detalhada de cada seção
    ├── intro.md              # Section 1: Introduction (draft #31)
    ├── background.md         # Section 2: Background (draft #31)
    ├── abstract.md           # Abstract (158 palavras)
    ├── methods.md            # Section 3 (slim, 503 palavras)
    ├── experiments.md        # Section 4 (969 palavras)
    ├── discussion.md         # Section 5 (964 palavras)
    ├── conclusion.md         # Section 6 (slim, 180 palavras)
    ├── appendix.md           # Marco 1 supplementary (500 palavras)
    ├── refs.bib              # Bibliography (14 entradas)
    ├── main.tex              # LaTeX consolidado (compilável via Overleaf)
    ├── latex_status.md       # Status compilação + plano Overleaf
    ├── linkedin_post.md      # Post LinkedIn versão longa PT-BR (~1900 chars)
    ├── linkedin_post_short.md # Post LinkedIn versão curta PT-BR (~750 chars)
    ├── generate_figures.py   # Script reusable das figuras
    └── figs/                 # fig1_sparsity_curve, fig2_validation (PNG+PDF 300 DPI)
```

Pastas em `.gitignore`: `data/`, `checkpoints/`, `logs/`, `wandb/`, `.venv/`.

---

## Decisões arquiteturais (timeline 30 sessões)

Documentadas em `PLAN.md` § Decisões arquiteturais e em `STRATEGY.md` por seção. Resumo das principais:

1. **STDP em PyTorch puro** (sessão #1) — GPU-first via vetorização. Brian2 como referência. Validada após 13 sessões (não revertida).
2. **Stack Python como scaffold** (sessão #1) — port Julia adiado pra Fase 2.
3. **k-WTA na dinâmica LIF** (sessão #2) — hard masking de spikes por posição.
4. **Pivot Semana 1 → Semana 2** (sessão #6) — MNIST kernel=28 caso patológico.
5. **CONTEXT.md §1.1 Refino #21** (sessão #21) — "STDP biofísico fiel" → "plasticidade local diferenciável" como framing operacional.
6. **Caminho C escolhido** (sessão #15) — pivot pra ProtoNet + features esparsas após família STDP saturar em ~36%. Produziu C3b (93.10%).
7. **Marco 1 (continual learning) escolhido** (sessão #21) — Split-Omniglot 50-tasks, sem replay, plasticidade meta-aprendida.
8. **Reformulação Pós-#23** (sessão #24) — alphabets + skip warmup. Insuficiente conforme #25.
9. **Caminho 5e** (sessão #28) — kitchen sink CNN+plasticidade+trace+k-WTA. Falhou em #29.
10. **CONTEXT.md §1.2 Refino #30** (sessão #30) — Marco 1 encerrado formalmente. Caminho 4 ativado: publicar só C3.

---

## Reproduzir resultados conhecidos

```powershell
$env:PYTHONIOENCODING = "utf-8"
cd experiment_01_oneshot

# Baselines (rápido, ~3 min total na 4070)
python baselines.py --baseline pixel_knn --ways 5 --shots 1 --episodes 1000
# Esperado: 45.76% (sessão #7)

python baselines.py --baseline proto_net --ways 5 --shots 1 --episodes 1000 --train-episodes 5000
# Esperado: 94.55% ACC (sessão #20). Com 500 train_episodes (smoke) dá ~85.88%.

# Resultado publicável C3 (sessão #20) — 3 níveis de sparsity, ~6 min total
python c3_protonet_sparse.py --device cuda --train-episodes 5000 --eval-eps 1000
# Esperado: C3a k=32 → 93.35% / C3b k=16 → 93.10% / C3c k=8 → 90.77%

# Família C completa (one-shot, sem treino) — sessão #15
python c1_hopfield_baselines.py --device cuda --episodes 1000
# Esperado: C1a Pixels+L2 50.17%, C1b PCA-32 56.28%, C1c RandomProj 41.23%

# Continual learning Marco 1 (encerrado em #30, mantido como referência)
cd ../experiment_02_continual
python baseline_naive.py --device cuda --seeds 5
# Esperado: ACC 80.65%, BWT -9.26 (sessão #25)
```

---

## Refs essenciais

Centrais para o paper C3:

- **Snell et al. (2017)** — *Prototypical Networks for Few-shot Learning.* NeurIPS.
- **Lake et al. (2015)** — *Human-level concept learning through probabilistic program induction.* Science. (Omniglot)
- **Maass (2000)** — *On the computational power of winner-take-all.* Neural Computation. (k-WTA)
- **Olshausen & Field (1996)** — *Emergence of simple-cell receptive field properties by learning a sparse code for natural images.* Nature.
- **Ahmad & Scheinkman (2019)** — *How Can We Be So Dense? The Benefits of Using Highly Sparse Representations.* arXiv:1903.11257.
- **Ramsauer et al. (2020)** — *Hopfield Networks Is All You Need.* ICLR 2021.

Centrais para a família STDP (exploração documentada):

- **Diehl & Cook (2015)** — *Unsupervised learning of digit recognition using spike-timing-dependent plasticity.* Frontiers in Computational Neuroscience.
- **Kheradpisheh et al. (2018)** — *STDP-based spiking deep convolutional neural networks for object recognition.* Neural Networks.

Bibliography completa do paper em `paper_c3/refs.bib` (BibTeX, ~14 entradas). Lista expandida em `CONTEXT.md` §6.

---

## Princípio operacional

Esse projeto é pesquisa, não produto. Métrica de progresso é **insight verificável**, não acurácia bruta. **Falhas bem-documentadas valem mais que sucessos superficiais** — todo experimento (mesmo os que falham) entra em `WEEKLY-N.md` com hipóteses ranqueadas e diagnóstico mecânico.

90% do tempo é leitura, hipótese, refinamento conceitual. 10% é experimento. Não otimizar código antes de validar ideia. Disciplina > entusiasmo.

## Author

**Luis Roberto Pinho da Silva Junior** — Independent research
[LinkedIn](https://www.linkedin.com/in/luisroberto0/)

**Project:** Project Hebb (2026)
**Affiliation:** Independent research

## Licença

MIT. Copyright (c) 2026 Luis Roberto Pinho da Silva Junior. Ver `LICENSE`.

## How to cite

Ver `CITATION.cff` na raiz do repositório (formato GitHub padrão). Citação rápida:

```
Pinho da Silva Junior, L. R. (2026). Project Hebb: Bio-inspired Neural Architectures Research. https://github.com/luisroberto0/project-hebb
```
