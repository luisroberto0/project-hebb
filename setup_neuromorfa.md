# Setup do Ambiente — Pesquisa Neuromorfa

**Hardware alvo:** Notebook Intel i9 + NVIDIA RTX 4070 (Ada Lovelace) + Windows
**Objetivo:** Laboratório local pra pesquisa em spiking neural networks, aprendizado one-shot com plasticidade local (STDP), memória episódica, e raciocínio causal.

---

## Filosofia da configuração

Pesquisa neuromorfa séria vive em Linux. NEST, simuladores avançados, e a maior parte dos papers que você vai querer reproduzir assumem ambiente Unix. Ao mesmo tempo, sua máquina é Windows — e a 4070 com drivers NVIDIA roda muito bem. A solução: **WSL2 + Ubuntu 24.04** como ambiente principal de pesquisa, com **Windows nativo** como fallback pra ferramentas que rodam bem nele (VS Code, Jupyter no browser, snnTorch).

CUDA na 4070 é exposto pra dentro do WSL2 sem virtualização — o desempenho é praticamente igual ao nativo. É o melhor dos dois mundos.

> **Se você quiser começar mais rápido sem WSL2:** dá pra fazer tudo em Windows nativo usando Miniforge + PyTorch + snnTorch + Norse + Brian2. Você só perde NEST e algumas ferramentas Linux-only. Veja a seção "Caminho rápido (Windows nativo)" no final.

---

## Etapa 1 — Drivers e CUDA no host Windows

Antes de qualquer coisa, garanta que o Windows está com drivers atualizados. WSL2 herda o driver do host.

1. Atualize o **driver NVIDIA** pela GeForce Experience ou direto em https://www.nvidia.com/Download/index.aspx (Game Ready ou Studio — tanto faz pra ML; Studio é mais conservador).
2. Confirme no PowerShell:
   ```powershell
   nvidia-smi
   ```
   Você deve ver a 4070 listada com a versão do driver e a versão de CUDA suportada (provavelmente CUDA 12.6+).
3. **Não instale o CUDA Toolkit no Windows** — você vai instalar dentro do WSL2. Conflitos de toolkit Windows + WSL costumam dar dor de cabeça.

---

## Etapa 2 — Habilitar WSL2 + Ubuntu 24.04

No PowerShell **como administrador**:

```powershell
wsl --install -d Ubuntu-24.04
```

Reinicie a máquina quando pedir. Na primeira inicialização do Ubuntu, crie um usuário (use o mesmo nome curto que você usa no Windows pra simplificar) e uma senha.

Confirme que está em WSL2 (não WSL1):

```powershell
wsl -l -v
```

Deve mostrar `VERSION 2`. Se mostrar 1, converta:

```powershell
wsl --set-version Ubuntu-24.04 2
```

---

## Etapa 3 — Atualizar Ubuntu e instalar dependências base

Dentro do Ubuntu:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget cmake \
    libssl-dev libffi-dev libgsl-dev libltdl-dev \
    libboost-all-dev libgomp1 \
    pkg-config python3-dev
```

Isso cobre compiladores C++ (Brian2 gera código C++ na hora) e bibliotecas que NEST e Brian2 dependem.

---

## Etapa 4 — Instalar Miniforge (gerenciador de ambientes)

Miniforge é o conda da comunidade — sem licença comercial da Anaconda Inc., padrão `conda-forge`, mais leve.

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh
```

Aceite as licenças, deixe instalar em `~/miniforge3`, e diga **yes** pra inicializar o conda no shell. Reabra o terminal.

Confirme:

```bash
conda --version
mamba --version
```

`mamba` é o resolvedor rápido do conda — sempre prefira `mamba install` em vez de `conda install` (10–20x mais rápido).

---

## Etapa 5 — Criar o ambiente de pesquisa

Vou usar Python 3.11 como base. É o sweet spot atual: Brian2, NEST, PyTorch, Norse e snnTorch todos têm wheels prontos pra 3.11. 3.12 ainda tem fricção com algumas libs neurais.

```bash
mamba create -n neuro python=3.11 -y
mamba activate neuro
```

A partir daqui, todo `pip install` cai dentro desse ambiente isolado. Se algo quebrar, você apaga (`mamba env remove -n neuro`) e recria sem afetar nada.

---

## Etapa 6 — PyTorch com CUDA 12.4

A 4070 é Ada Lovelace (compute capability 8.9). Use PyTorch oficial com CUDA 12.4 — vai detectar a GPU dentro do WSL2 automaticamente.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Validação rápida:

```bash
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('Versão CUDA:', torch.version.cuda)"
```

Saída esperada: algo como `CUDA disponível: True`, `GPU: NVIDIA GeForce RTX 4070 ...`, `Versão CUDA: 12.4`.

Se `CUDA disponível: False`, o problema quase sempre é driver Windows desatualizado ou WSL1 em vez de WSL2.

---

## Etapa 7 — Frameworks de spiking neural networks

Os três que importam pra você no curto prazo:

### snnTorch (mais amigável, PyTorch puro)

```bash
pip install snntorch
```

Permite construir SNNs como módulos `torch.nn`. Treina por surrogate gradient (BPTT em spikes). Bom pra protótipos rápidos.

### Norse (PyTorch nativo, mais "neuro" que snnTorch)

```bash
pip install norse
```

Da Hugging Face, foco em modelos LIF/AdEx com mais fidelidade biológica que snnTorch. Suporta plasticidade local.

### Brian2 (simulador científico, código C++ gerado)

```bash
pip install brian2 brian2tools
```

Padrão-ouro pra neurociência computacional. Você descreve neurônios e sinapses em equações diferenciais e ele gera C++ otimizado. Não usa GPU por padrão — roda paralelizado em CPU (seu i9 brilha aqui).

### NEST (opcional, instalação mais pesada)

NEST é o simulador de redes em larga escala usado pelo Human Brain Project. Vale instalar se você for escalar pra centenas de milhares de neurônios.

```bash
mamba install -c conda-forge nest-simulator -y
```

---

## Etapa 8 — Bibliotecas auxiliares

```bash
pip install numpy scipy matplotlib seaborn pandas \
    jupyterlab notebook ipykernel \
    scikit-learn tqdm einops \
    tensorboard wandb \
    pyro-ppl dowhy causalnex
```

Resumo:
- `jupyterlab` + `notebook` — exploração interativa
- `einops` — manipulação tensorial legível (essencial pra SNN)
- `tensorboard` / `wandb` — logging de experimentos
- `pyro-ppl`, `dowhy`, `causalnex` — pra quando atacar o pilar de raciocínio causal

Registre o kernel do Jupyter:

```bash
python -m ipykernel install --user --name neuro --display-name "Python (neuro)"
```

---

## Etapa 9 — Datasets de referência

Pra o primeiro experimento (one-shot learning), o benchmark padrão é **Omniglot** — 1623 caracteres de 50 alfabetos, 20 amostras de cada. É o "MNIST do meta-learning".

```bash
pip install torchmeta learn2learn
```

Os datasets baixam automaticamente na primeira execução. `learn2learn` é mais ativo hoje que `torchmeta`.

---

## Etapa 10 — VS Code + WSL

Instale o VS Code no Windows e a extensão **WSL** (Microsoft). Aí você abre o terminal WSL e roda:

```bash
code .
```

O VS Code abre com o backend rodando dentro do WSL — você edita código Linux como se fosse local, com debugger, terminal, Jupyter inline, tudo integrado.

Extensões recomendadas dentro do WSL:
- Python
- Jupyter
- Pylance
- GitLens
- Ruff (linter rápido)

---

## Etapa 11 — Estrutura de projeto sugerida

```bash
mkdir -p ~/research/neuromorfa
cd ~/research/neuromorfa
git init
mkdir -p experiments notebooks data papers logs
```

Convenção:
- `experiments/` — scripts numerados por experimento (`01_oneshot_omniglot/`)
- `notebooks/` — exploração e visualização
- `data/` — datasets baixados (no `.gitignore`)
- `papers/` — PDFs de referência
- `logs/` — saídas de TensorBoard / wandb

Crie um `.gitignore` mínimo:

```
data/
logs/
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
wandb/
```

---

## Caminho rápido (Windows nativo) — alternativa

Se você não quiser WSL2 agora, dá pra começar 100% em Windows com a maior parte da stack:

1. Instale **Miniforge for Windows**: https://github.com/conda-forge/miniforge
2. Abra o "Miniforge Prompt" e:
   ```
   mamba create -n neuro python=3.11 -y
   mamba activate neuro
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install snntorch norse brian2 brian2tools
   pip install numpy scipy matplotlib jupyterlab learn2learn
   ```
3. Pra Brian2 funcionar, instale **Microsoft Visual C++ Build Tools** (ele compila C++ na hora).

O que você perde no Windows nativo:
- **NEST** — não roda nativo no Windows, precisa de WSL2 ou Docker
- Ferramentas Linux-only de neurociência (PyNN com backends específicos, alguns simuladores de plasticidade)

Pra os primeiros 3–6 meses de pesquisa em one-shot learning + STDP, Windows nativo é suficiente. WSL2 fica como upgrade quando você quiser escalar.

---

## Próximos passos depois do setup

1. Rodar os **scripts de validação** (`validate_*.py`) pra confirmar que tudo está OK.
2. Reproduzir o paper **Diehl & Cook (2015)** — STDP não-supervisionado em MNIST. É o "hello world" sério da área. Tem implementação de referência em Brian2.
3. Mover pra Omniglot one-shot com plasticidade local (sua meta).
4. (Opcional) Quando precisar de Julia: `curl -fsSL https://install.julialang.org | sh` dentro do WSL. Pacotes principais: `Flux.jl`, `WaspNet.jl`, `SpikingNeuralNetworks.jl`.

---

## Troubleshooting comum

**`torch.cuda.is_available()` retorna False no WSL2**
→ Driver NVIDIA do Windows desatualizado, ou WSL ainda em versão 1. Atualize driver, rode `wsl --update` e `wsl --shutdown` antes de reabrir.

**Brian2 reclama de compilador no Linux**
→ `sudo apt install build-essential` (geralmente já feito na Etapa 3).

**`mamba` muito lento mesmo assim**
→ Limpe caches: `conda clean -a -y`. Verifique se `~/.condarc` tem `channels: [conda-forge]` e `channel_priority: strict`.

**VS Code não abre dentro do WSL**
→ Reinstale a extensão WSL e rode `code .` direto no terminal WSL.

**Memória da 4070 estourando em treino de SNN**
→ Reduza batch size; use `torch.cuda.empty_cache()` entre runs; SNNs com BPTT em muitos timesteps são VRAM-hungry. Considere `bptt_truncated` ou treinamento online (forward-only com plasticidade local).
