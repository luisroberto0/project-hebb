# Project Hebb

Pesquisa em arquiteturas neurais bio-inspiradas: plasticidade local, codificação esparsa (k-WTA), dinâmica spiking. Estudo empírico **dos limites** da bio-inspiração como caminho pós-LLM, ao longo de 4 capacidades (one-shot, continual learning, eficiência radical, raciocínio temporal) e múltiplos benchmarks (Omniglot, CUB-200, Fashion-MNIST, SHD/SSC). Síntese honesta da jornada em [`SYNTHESIS.md`](SYNTHESIS.md).

> *"Não tente construir a mente. Construa um neurônio que funcione diferente."*
> — Luis Roberto Pinho da Silva Junior, Project Hebb (2026)

---

## Status (atualizado em 2026-06-19, pós-jornada SoftHebb — Marcos 3-5 + eficiência)

**A tese do projeto, fundamentada:** a plasticidade local Hebbiana (sem backprop) **não entrega capacidades pós-LLM que o backprop não tem — mas entrega as mesmas features úteis e robustas por ~1/21 do custo.** É eficiência, não superioridade. Provada (Marco 3), escalada (Marco 5), e quantificada (21× mais rápido), com a contribuição honestamente isolada a cada passo. **Publicando a jornada** ([`writeups/linkedin-jornada-pt.md`](writeups/linkedin-jornada-pt.md), arco de 3 atos; capstone técnico em [`SYNTHESIS.md`](SYNTHESIS.md)).

- **Marco 4 — continual learning com plasticidade local (MEDIANO).** O SoftHebb sequencial **não esquece** (BWT +0,34) e supera o backprop supervisionado (que sofre catastrophic forgetting, −16,78). **Mas o controle decisivo** (autoencoder backprop *não-supervisionado*) também não esquece (+4,59): a resistência vem do **não-supervisionado**, não da localidade Hebbiana. Mesmo padrão do GRU no Marco 2-C — o controle adversarial desinfla a narrativa fácil. `experiment_07_continual_local/`.
- **Marco 5 — escala (positivo).** Tiny-ImageNet (200 classes): a margem sobre random **persiste** (+9,31 p.p.; softhebb 31,67% ≈ 63× chance). O sinal do Marco 3 é robusto a escala, não específico de CIFAR-10. `experiment_08_scale/`.
- **Eficiência — quantificada.** SoftHebb treina features **21× mais rápido** que o backprop (33s vs 686s, sem labels/backprop, a −7 p.p.). A contribuição ortogonal medida. `experiment_06_plasticity/efficiency.py`.
- **Marco 6 (futuro) — hardware neuromórfico.** Medir a *energia real* em silício (EBRAINS/Akida) — validação física da tese de eficiência. Aguarda acesso a hardware.

---


**Marco 3 — plasticidade local SEM backprop (eixo SoftHebb): 1º POSITIVO LIMPO da premissa-mãe.** Depois de 4 capacidades testadas (3 negativos + 1 desinflado), o projeto finalmente atacou a *premissa-mãe* (CONTEXT §1, linha 16): pode uma regra de plasticidade **local, sem backprop**, aprender uma representação útil? Eixo escolhido após mapear 7 famílias backprop-free ([`writeups/plasticidade-landscape.md`](writeups/plasticidade-landscape.md)): **SoftHebb** (Hebbiano competitivo, ICLR 2023) — o teste mais limpo (regra local fechada, zero autograd na pilha de features). **Critério literal fixado antes.** Resultado (3 seeds, CIFAR-10, linear-probe): pilha conv treinada **só** por Hebbiano competitivo local atinge **80,27%** — **+11,67 p.p. sobre pesos-random** (IC95 [+11,2, +12,0]: sinal **real**, não-arquitetural, o oposto do STDP), com a **competição essencial** (desligá-la colapsa para 43%, abaixo do random — o oposto do termo Hebbiano dispensável do C2), e a só **−6,84 p.p. do backprop** (mesma arquitetura). **Veredicto:** SUCESSO em 2 das 3 sub-condições (acc ≥75% ✓, ≤15 p.p. do backprop ✓); a margem sobre random (+11,67) ficou 3,3 p.p. abaixo do limiar de SUCESSO pleno (+15) — e 2 vias principled de empurrá-la (mais treino, ZCA whitening) falharam honestamente. **Mas é, sem ambiguidade, o primeiro mecanismo bio-inspirado do projeto que carrega sinal genuíno** — a prova de conceito da tese, que faltava. Ressalvas honestas: o classificador final usa backprop (a *pilha* é 100% local); margem abaixo do limiar; gap de 6,8 p.p. para o backprop. `experiment_06_plasticity/`.

---

**Marco 2-C — raciocínio temporal (#72–#78): timing genuíno, mas NÃO uma vantagem do spiking.** A 4ª e última capacidade pós-LLM, atacada no domínio nativo da SNN (SHD, Cramer/Zenke 2020): SNN recorrente vs **baseline cego ao timing** (histograma de spikes por canal → MLP). **#72 (sweep formal, 5 seeds, IC95%):** cego 51.56% / SNN-ff 61.02% / **SNN-rec 71.27%** — timing (rec−cego) **+19.71 p.p.**, critério literal atingido; parecia o 1º marco positivo. Caracterizado em 6 frentes: decomposição timing-genuíno/arquitetura (#73, +10.18 p.p. — mas *upper bound*, pois em bins=1 a recorrência é inerte), latency coding (#74, 50.68% só do onset), k-WTA temporal tolerante a 75% (#75, −1.50 p.p., paralelo ao C3 espacial), generalização SSC fraca/dataset-específica (#76–77b, +4–5 p.p.). **A virada (#78, controle pedido pelo peer review):** um **GRU não-spiking** no mesmo input atinge **79.64% — supera a SNN (69.10%) por +10.5 p.p.** O timing é **genérico de recorrência, não do spiking**, e a SNN é uma forma *inferior* de explorá-lo. **O único positivo aparente foi desinflado:** a SNN explora timing genuíno, mas não há vantagem competitiva da bio-inspiração — o 2-C junta-se de fato aos 3 negativos. **Status das 4 capacidades pós-LLM: nenhuma deu vantagem competitiva à abordagem neuro-inspirada no regime testável.** Paper `paper_marco2c/` (draft-completo, peer-reviewed, GRU integrado); decisão de enquadramento/venue + rumo do projeto em aberto (do Luis). Ver `SYNTHESIS.md` (capstone da jornada) + `experiment_05_temporal/`.

---

**Marco 2-B — eficiência radical (encerrado #70, achado negativo).** Reaberto em #67 (decisão do Luis via /goal) para atacar a **3ª capacidade pós-LLM — eficiência radical** (CONTEXT §1: rodar em CPU comum). Eixo: inferência event-driven — SNN-LIF + k-WTA temporal vs MLP denso em Fashion-MNIST, métrica dupla **SynOps** (teórico) + **latência CPU** (real). **Resultado (#70, sweep 5 seeds, IC95% bootstrap):** Falha decisiva — nenhuma config SNN atinge acc dentro de −2 p.p. (denso 87.16% vs melhor SNN 84.82%) **E** SynOps ≥5× menores **E** latência CPU ≤ denso. Achados: k-WTA só ataca SynOps na entrada (não na hidden); trade-off acc↔SynOps íngreme (4.79× menos SynOps custa −14.6 p.p.); **latência CPU 80–327× pior sempre**; inferência event-driven (sparse) é até mais lenta que o runtime denso (overhead de indexação > matmul BLAS). **Conclusão:** eficiência radical via SNN **não se realiza em CPU von Neumann** — é co-design hardware-algoritmo (silício neuromórfico), consistente com Merolla/Davies. Arquivado em `experiment_04_efficiency/`. **3 das 4 capacidades pós-LLM agora têm achado negativo documentado; 1 (raciocínio temporal) não atacada.** Próximo marco em aberto (decisão do Luis).

---

**Marco 2-A — cross-domain few-shot (encerrado #66, paper arquivado).** Após o fechamento do Marco 1 (#30) e a decisão pós-#36 de publicar C3 via LinkedIn, o projeto reabriu em #52 (2026-05-14) para atacar uma sub-capacidade pós-LLM específica: **one-shot inédito** em domínio visual radicalmente diferente do treino. Pergunta: o encoder C3 (CNN-4 + k-WTA k=16) treinado em Omniglot e **congelado** bate ProtoNet retreinado em CUB-200 por ≥5 p.p. cross-domain? Critério literal, limite hard de 15 sessões (#52-#66).

**Resultado (#52-#57, 5 seeds, IC95% bootstrap):** critério **refutado conforme previsto**. As cinco condições CNN-forward (4 sparsities C3 + random encoder) colapsam num cluster de 21.68–22.20% — apenas ~2 p.p. acima de chance (20%), com ICs sobrepostos. ProtoNet **retreinado** em CUB chega a 34.31% (28×28 gray) / 49.84% (84×84 RGB). Três achados centrais:

1. **k-WTA effect collapse:** o spread entre k=8 e k=64 cai de **3.78 p.p. in-domain (Omniglot)** para **0.52 p.p. cross-domain (CUB)** — ruído. A esparsidade que importava in-domain some sob transfer extremo.
2. **Anti-transfer:** encoder treinado em fonte distante é estatisticamente indistinguível de random encoder cross-domain (+0.18 p.p., ICs sobrepostos). Consistente com STARTUP (Phoo & Hariharan 2021).
3. **Bottleneck decomposition:** treino na target (+12.22 p.p.) e resolução/cor adequadas (+15.53 p.p.) são os dois gargalos reais e ~aditivos; sparsity contribui zero. Contraintuitivo: Pixel kNN (22.81%) supera todos os encoders.

**Paper Marco 2-A** ("When Sparsity Stops Mattering: k-WTA Effect Collapse Under Extreme Domain Shift", `paper_marco2a/`) — **draft-completo**: 6 seções + abstract (#58–#62), 3 figuras 300 DPI (#60), `main.tex` consolidado + bib final (#63, validação estrutural OK; compila via Overleaf), e **peer review interno adversarial** (#64–65, verdict "pequenos ajustes" — todos os números das tabelas conferem; 10 correções de consistência aplicadas; 16 decisões de escopo/tom registradas em `paper_marco2a/peer_review.md`). **#66 (admin, 2026-06-08): Marco 2-A encerrado** — critério refutado conforme previsto; ajustes de rigor do peer review (alta+média) incorporados; paper **arquivado como achado documentado, sem publicar** (decisão do Luis — nem workshop, nem LinkedIn). Project Hebb volta a estado de manutenção; próximo marco (2-B eficiência / 2-C temporal / encerrar projeto) **em aberto**.

---

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

**Status paper writing:** sessão #36 (2026-04-30) — **decisão final: NÃO submeter NeurIPS Workshop, publicar via LinkedIn em PT.** Razão: tempo limitado de side project não absorve rebuttals/revisões/registration; LinkedIn alcança parte do que peer review faria sem overhead institucional. Paper draft completo preservado: abstract + 6 seções (3789 palavras) + 2 figuras 300 DPI + main.tex compilável + refs.bib (14 entradas) + LinkedIn post drafts (longo + curto). PDF gerável via Overleaf em ~10 min. Ver `STRATEGY.md` "Decisão pós-#35: LinkedIn em vez de NeurIPS" e `paper_c3/README.md` "Final state". Project Hebb entra em estado de manutenção pós-#36.

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
| [`SYNTHESIS.md`](SYNTHESIS.md) | **Capstone da jornada pós-LLM:** 5 marcos (3 ❌ + 1 ✅ modesto), narrativa transversal de k-WTA, avaliação honesta. Leia para o panorama completo. |
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
├── experiment_03_crossdomain/ # Cross-domain few-shot Omniglot→CUB-200 (Marco 2-A, #52-#57)
│   ├── PLAN.md               # Pergunta científica, critério literal, plano #52-#66
│   ├── PAPERS.md             # Lit review cross-domain (5 papers core)
│   ├── WEEKLY-1.md           # 7+ condições caracterizadas, IC95% bootstrap (#52-#57)
│   ├── cub_data.py           # Dataloader CUB-200 + cache 28×28 gray / 84×84 RGB
│   ├── episodes.py           # Sampler N-way K-shot cross-domain
│   ├── eval_crossdomain.py   # Eval C3/ProtoNet Omniglot-frozen → CUB
│   ├── eval_pixel_knn.py     # Pixel kNN cross-domain (sanity floor)
│   ├── eval_random_encoder.py# Random encoder + k-WTA (sanity floor)
│   ├── train_cub_protonet.py # ProtoNet retreinado em CUB (baseline a bater)
│   ├── train_encoders.py     # Sweep k-WTA source-trained (k=8,16,32,64)
│   └── smoke_test.py         # Smoke test do pipeline
└── paper_marco2a/            # Workshop paper Marco 2-A (cross-domain k-WTA collapse)
    ├── README.md, outline.md # Overview + estrutura detalhada
    ├── intro.md, background.md          # Sections 1-2 (draft #58)
    ├── methods.md, experiments.md       # Sections 3-4 + Table 1/2 (draft #59)
    ├── discussion.md, conclusion.md     # Sections 5-6 (placeholders #61-#62)
    ├── refs.bib             # Bibliography (~18 entradas)
    ├── generate_figures.py  # Script reproduzível das 3 figuras (#60)
    └── figs/                # fig1_crossdomain_bars, fig2_effect_collapse, fig3_bottleneck_waterfall (PNG+PDF 300 DPI)
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
