# Project Hebb

Pesquisa de longo prazo em arquiteturas de IA biologicamente inspiradas. Foco: spiking neural networks (SNNs) com plasticidade local sináptica (STDP), com objetivo de oferecer capacidades que LLMs não têm — aprendizado contínuo sem catastrophic forgetting, one-shot real, eficiência radical em hardware consumer (CPU / GPU laptop), raciocínio temporal via timing de spikes.

> *"Não tente construir a mente. Construa um neurônio que funcione diferente."*

---

## Status (atualizado em 2026-04-29, pós-sessão #30)

**Fase 1 — Fundação concluída** (30 sessões). Project Hebb teve 2 experimentos:
- **Experimento 01 (one-shot Omniglot):** ✅ atingiu metas numéricas via Caminho C (ProtoNet + k-WTA esparso). Vira paper de workshop.
- **Experimento 02 (continual learning sem replay):** ❌ Marco 1 encerrado pelo critério literal (sessão #29). 4 abordagens testadas, todas ≤ baseline naive. Achado mecanístico documentado.

### Resultados principais (Omniglot 5-way 1-shot)

| Modelo | ACC 5w1s | ACC 20w1s | Sessão | Status |
|---|---|---|---|---|
| Pixel kNN | 45.76% | — | #7 | baseline trivial |
| Iter 1 STDP saturado (melhor de 13 sessões) | 35.98% | 9.80% | #9 | barreira estrutural identificada (#1-#13) |
| C1b PCA-32 + Hopfield (sem treino) | 56.28% | 35.37% | #15 | baseline arquitetural |
| C2-simplified (plasticidade meta-aprendida) | 64.08% | — | #19 | melhor pré-CNN |
| **C3b ProtoNet + k-WTA k=16 (75% sparsity)** | **93.10%** | **80.72%** | #20 | **target paper** |
| ProtoNet baseline | 94.55% | — | #20 | upper bound |

C3b atinge as duas metas numéricas do `CONTEXT.md` §4 (≥90% 5w1s, ≥70% 20w1s) com mecanismo neural-inspirado defensável (esparsidade biológica). Não cumpre restrição mecanística "sem backprop end-to-end" — registrado honestamente em CONTEXT.md §1.2.

### Próximo passo

**Paper C3:** "k-WTA Sparsity Preserves Prototypical Network Performance in Few-Shot Learning". Submissão alvo NeurIPS Bio-Plausible Learning Workshop ~setembro 2026. Cronograma: 5 sessões de paper writing (~10-15h reais). Ver `STRATEGY.md` "Plano paper C3".

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

### Primeiro comando útil — smoke test do pipeline Omniglot

```powershell
# No Windows, setar PYTHONIOENCODING=utf-8 evita UnicodeEncodeError no console cp1252
$env:PYTHONIOENCODING = "utf-8"
cd experiment_01_oneshot

# Baseline random (sem checkpoint) — confirma que pipeline fecha (~chance)
python evaluate.py --device cuda --ways 5 --shots 1 --episodes 100

# Baselines clássicos
python baselines.py --baseline pixel_knn --ways 5 --shots 1 --episodes 100
python baselines.py --baseline proto_net --ways 5 --shots 1 --episodes 100 --train-episodes 500

# Pretreino STDP curto (smoke, ~30s) + eval — atualmente não passa de chance,
# bloqueio ativo documentado em WEEKLY-2.md
python train.py --device cuda --n-images 500 --epochs 1
python evaluate.py --device cuda --checkpoint checkpoints/stdp_model.pt --ways 5 --shots 1 --episodes 100
```

---

## Arquitetura

Pipeline completo (em `experiment_01_oneshot/model.py:STDPHopfieldModel`):

```
Imagem 28×28 (Omniglot, downsample de 105×105, fundo invertido)
    ↓
Codificação Poisson (T=100 timesteps, max_rate=100Hz)
    ↓
Conv-STDP layer 1 (1 → 8 filtros, kernel 5, padding 2)
    + LIF integrator + k-WTA (k=1) por posição espacial
    + adaptive threshold homeostático (Diehl & Cook §2.3)
    ↓
MaxPool 2×2
    ↓
Conv-STDP layer 2 (8 → 16 filtros, kernel 5, mesma dinâmica)
    ↓
MaxPool 2×2 → flatten → projeção ortogonal (embedding_dim=64)
    ↓
Modern Hopfield Memory (Ramsauer et al. 2020)
    armazena protótipos do support, recupera via β-softmax
    ↓
argmin distância → predição N-way K-shot
```

**Características:**
- Pesos STDP têm `requires_grad=False` — atualizados via regra local de spike timing, **não backprop**.
- STDP convolucional vetorizado em PyTorch puro (`F.unfold` + `einsum`). Decisão arquitetural permanente registrada em `PLAN.md`.
- Inibição lateral via k-WTA hard masking na membrana LIF (não decay de pesos pós-STDP).
- Memória episódica Hopfield Moderna armazena suporte do episódio, NÃO é treinada.

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
├── BLOCKED.md                # Histórico de bloqueios (Semana 1, fechado)
├── CLAUDE.md                 # Guia para Claude Code CLI
├── README.md                 # Este arquivo
├── LICENSE                   # MIT
├── requirements.txt
├── environment.yml
├── validate_environment.py
├── validate_snn_minimal.py
├── validate_brian2_stdp.py
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
└── experiment_02_continual/  # Continual learning (Marco 1 ENCERRADO em #30)
    ├── PLAN.md (marcado ENCERRADO no topo)
    ├── PAPERS.md (lit review EWC/SI/GEM/A-GEM/Hadsell)
    ├── WEEKLY-1.md (sessões #22-#29)
    ├── baseline_naive.py             # Naive sequential ProtoNet (#23, #25)
    ├── c2_continual_arch_a.py        # Scaffold (não implementado)
    ├── c2_continual_arch_b.py        # Possibilidade B linear (#27)
    ├── c2_continual_arch_c.py        # Scaffold (não implementado)
    └── c5e_combined.py               # Caminho 5e kitchen sink (#28-#29)
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

- **Diehl & Cook (2015)** — *Unsupervised learning of digit recognition using spike-timing-dependent plasticity.* Frontiers in Computational Neuroscience.
- **Ramsauer et al. (2020)** — *Hopfield Networks Is All You Need.* ICLR 2021.
- **Lake et al. (2015)** — *Human-level concept learning through probabilistic program induction.* Science. (Omniglot)
- **Snell et al. (2017)** — *Prototypical Networks for Few-shot Learning.* NeurIPS.
- **Maass (1997)** — *Networks of Spiking Neurons: The Third Generation of Neural Network Models.*
- **Kheradpisheh et al. (2018)** — *STDP-based spiking deep convolutional neural networks for object recognition.* Neural Networks.

Lista completa em `CONTEXT.md` §6 e `experiment_01_oneshot/PLAN.md` §12.

---

## Princípio operacional

Esse projeto é pesquisa, não produto. Métrica de progresso é **insight verificável**, não acurácia bruta. **Falhas bem-documentadas valem mais que sucessos superficiais** — todo experimento (mesmo os que falham) entra em `WEEKLY-N.md` com hipóteses ranqueadas e diagnóstico mecânico.

90% do tempo é leitura, hipótese, refinamento conceitual. 10% é experimento. Não otimizar código antes de validar ideia. Disciplina > entusiasmo.

## Autor

Luis Roberto — [LinkedIn](https://www.linkedin.com/in/luisroberto0/)

## Licença

MIT. Ver `LICENSE`.
