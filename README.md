# Project Hebb

Pesquisa de longo prazo em arquiteturas de IA biologicamente inspiradas. Foco: spiking neural networks (SNNs) com plasticidade local sináptica (STDP), com objetivo de oferecer capacidades que LLMs não têm — aprendizado contínuo sem catastrophic forgetting, one-shot real, eficiência radical em hardware consumer (CPU / GPU laptop), raciocínio temporal via timing de spikes.

> *"Não tente construir a mente. Construa um neurônio que funcione diferente."*

---

## Status

**Fase 1 — Fundação** (meses 1–3). Experimento ativo: one-shot learning em **Omniglot** com pipeline `STDP convolucional → memória Hopfield Moderna`, sem backpropagação end-to-end.

| Marco | Estado |
|-------|--------|
| Stack PyTorch + CUDA + RTX 4070 | Validada |
| Pipeline Omniglot (data + train + eval + baselines) | Funciona end-to-end |
| Sanity check Diehl & Cook 2015 (MNIST) | Concluída com ressalva (caso patológico documentado, 17.76% vs meta 85%) |
| Pretreino STDP em Omniglot | Bloqueio ativo: saturação de pesos em escala média/longa |
| Avaliação 5w1s / 20w1s vs MAML / ProtoNet | Não atingida ainda |

**Baselines internos medidos** (Omniglot 5-way 1-shot, 100 episódios): Pixel kNN 45.76%, ProtoNet 85.88% (com 500 train episodes). Modelo STDP+Hopfield principal ainda não bate baselines — é o que o trabalho atual ataca. Ver `experiment_01_oneshot/WEEKLY-2.md` pra diagnóstico do bloqueio atual e `STRATEGY.md` pra próximas hipóteses.

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
| [`CONTEXT.md`](CONTEXT.md) | Mission, princípios operacionais, stack, baselines, papers. **Sempre primeiro.** |
| [`PLAN.md`](PLAN.md) | Plano operacional vivo. Notas de iteração (1 linha por sessão), Decisões arquiteturais permanentes. |
| [`STRATEGY.md`](STRATEGY.md) | Estratégia de pesquisa: próximas 2-3 sessões, critérios de pivot, fallback Brian2. |
| [`experiment_01_oneshot/PLAN.md`](experiment_01_oneshot/PLAN.md) | Pergunta de pesquisa, hipótese, arquitetura detalhada, roadmap 6 semanas com status. |
| [`experiment_01_oneshot/WEEKLY-1.md`](experiment_01_oneshot/WEEKLY-1.md) | Sanity MNIST (concluída com ressalva, 6 sessões de iteração). |
| [`experiment_01_oneshot/WEEKLY-2.md`](experiment_01_oneshot/WEEKLY-2.md) | Adaptação Omniglot (ativa, bloqueada em saturação). |
| [`experiment_01_oneshot/WEEKLY-2-NEXT.md`](experiment_01_oneshot/WEEKLY-2-NEXT.md) | Auditoria da infra Omniglot e ordem sugerida pra Semana 2. |
| [`BLOCKED.md`](BLOCKED.md) | Histórico do bloqueio Semana 1 (FECHADO desde sessão #6, mantido como artefato). |
| [`CLAUDE.md`](CLAUDE.md) | Guia operacional pra futuras sessões Claude Code. |

---

## Estrutura do repositório

```
project-hebb/
├── CONTEXT.md                # Briefing conceitual (mission, princípios, stack)
├── PLAN.md                   # Plano operacional vivo (notas de iteração, decisões)
├── STRATEGY.md               # Estratégia de pesquisa formalizada
├── BLOCKED.md                # Estado de bloqueios (histórico Semana 1)
├── CLAUDE.md                 # Guia para Claude Code CLI
├── README.md                 # Este arquivo
├── LICENSE                   # MIT
├── requirements.txt          # Dependências Python
├── environment.yml           # Conda (alternativa)
├── validate_environment.py   # Sanity check do setup
├── validate_snn_minimal.py   # SNN mínima end-to-end (snntorch)
├── validate_brian2_stdp.py   # Brian2 + STDP (referência)
└── experiment_01_oneshot/
    ├── PLAN.md               # Roadmap 6 semanas, hipótese, protocolo
    ├── WEEKLY-{0..6}.md      # Progresso por semana
    ├── WEEKLY-2-NEXT.md      # Preparação Semana 2
    ├── config.py             # Hiperparâmetros centrais (editar AQUI)
    ├── data.py               # Omniglot loader + EpisodeSampler + spike encoding
    ├── model.py              # ConvSTDPLayer + HopfieldMemory + STDPHopfieldModel
    ├── train.py              # Loop de pretreino STDP
    ├── evaluate.py           # N-way K-shot com IC bootstrap
    ├── baselines.py          # Pixel kNN + Prototypical Networks
    ├── sanity_mnist.py       # Reprodução Diehl & Cook 2015 (Semana 1)
    ├── analysis.py           # Geração de RESULTS.md a partir de logs
    ├── run_all.ps1           # Pipeline automatizado das 6 semanas (pwsh 7+)
    ├── tests/
    │   ├── test_assignment.py            # Auditoria de assign_labels e evaluate
    │   ├── test_spike_balance.py         # Razão pré:pós em MNIST kernel=28
    │   └── test_spike_balance_omniglot.py # Razão pré:pós em arquitetura conv real
    └── utils/
        └── visualize.py      # Visualização de filtros aprendidos
```

Pastas em `.gitignore`: `data/` (datasets), `checkpoints/` (`.pt` files), `logs/`, `wandb/`, `.venv/`.

---

## Decisões arquiteturais já tomadas

Documentadas em `PLAN.md` § Decisões arquiteturais. Resumo:

1. **STDP em PyTorch puro** (não Brian2) — GPU-first via vetorização `F.unfold` + `einsum`. Custo de reverter: ~1 semana. Brian2 fica como referência canônica.
2. **Stack Python como scaffold inicial** — port pra Julia adiado pra Fase 2 se modelo provar valor. Ecossistema científico maduro pesa mais que performance teórica nesta fase.
3. **k-WTA na dinâmica LIF** (não decay de pesos pós-STDP) — fiel a Diehl & Cook 2015 §Methods. Hard masking de spikes por posição espacial.
4. **Adaptive threshold homeostático** (Diehl & Cook 2015 §2.3) — buffer `theta` por filtro, cresce com spikes próprios e decai com tempo.
5. **Pivot Semana 1 → Semana 2** — MNIST com kernel=28 é caso patológico (k-WTA degenerado sobre output 1×1). Stack validada por outras vias; ROI baixo de continuar em MNIST.

---

## Reproduzir resultados conhecidos

```powershell
$env:PYTHONIOENCODING = "utf-8"
cd experiment_01_oneshot

# Baselines (rápido, ~3 min total na 4070)
python baselines.py --baseline pixel_knn --ways 5 --shots 1 --episodes 100
# Esperado: ~45-50% acc

python baselines.py --baseline proto_net --ways 5 --shots 1 --episodes 100 --train-episodes 500
# Esperado: ~85% acc (com mais train_episodes vai pra ~98% do paper)

# Auditoria do pipeline de classificação (3 testes sintéticos, ~5s)
python tests/test_assignment.py
# Esperado: 3/3 passam

# Diagnóstico de regime de spikes em conv real (~5s)
python tests/test_spike_balance_omniglot.py
# Esperado: R1≈1.5, R2≈0.7
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

## Licença

MIT. Ver `LICENSE`.
