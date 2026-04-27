# PLAN.md — Plano Operacional

> Plano vivo do projeto. Atualizado a cada fase concluída ou decisão arquitetural.
> Para o briefing conceitual (missão, princípios, stack), veja `CONTEXT.md`.

---

## Fase atual

**Fase 1 — Fundação** (meses 1–3)

Objetivo: estabelecer fundamento experimental sólido em uma das três frentes de pesquisa, com pipeline reproduzível e baseline próprio acima dos triviais. Métrica de fim de fase: critérios listados em `CONTEXT.md` §4.

Frente escolhida: **aprendizado one-shot com plasticidade local (STDP)** — ver `experiment_01_oneshot/PLAN.md` pra detalhes.

---

## Experimento ativo

[`experiment_01_oneshot/`](./experiment_01_oneshot/) — One-shot Omniglot com STDP + Memória Hopfield Moderna.

Status: implementação completa entregue (STDP vetorizado, sanity MNIST, pretreino Omniglot, eval N-way K-shot, baselines, análise automática, run_all.ps1). Pendente apenas execução na 4070 — migrada pro Claude Code CLI no Windows host.

Sequência mandada:

1. `pwsh -File experiment_01_oneshot/run_all.ps1` (com `$env:HEBB_QUICK="1"` pra debug rápido na primeira passada)
2. Ler `experiment_01_oneshot/RESULTS.md` gerado
3. Conforme outcome: paper rascunho / `NEXT.md` / `POSTMORTEM.md`

---

## Experimentos planejados

| ID | Tema | Status | Pasta |
|---|---|---|---|
| 01 | One-shot Omniglot com STDP + Hopfield | **ativo (pendente execução)** | [`experiment_01_oneshot/`](./experiment_01_oneshot/) |
| 02 | Memória episódica (replay-based, hippocampus-inspired) | placeholder | `experiment_02_episodic_memory/` (a criar) |
| 03 | Raciocínio causal explícito (Pearl-style) | placeholder | `experiment_03_causal/` (a criar) |

A ordem reflete maturidade teórica decrescente: 01 tem décadas de fundamentação (STDP, Hebbian), 02 tem trabalho recente sólido (Hopfield moderno, replay neuroscience), 03 é o mais teórico (necessita estudo prévio de Pearl). Não pular ordem — completar 01 antes de iniciar 02.

---

## Notas de iteração

> Registrar aqui falhas, surpresas e ajustes de rumo. Falhas informam mais que sucessos.
> Formato: `YYYY-MM-DD — observação. Causa hipotética. Decisão tomada.`

- **2026-04-27 — Sandbox do agente sem ambiente Python executável.** Cowork sandbox Linux remoto não tem PyTorch nem rede pra instalar; sem GPU; sem datasets. Causa: limitação intencional de segurança da plataforma (allowlist de rede, sem GPU passthrough). Decisão: pivot pra Opção B — agente entrega scaffold completo em uma rodada, execução migra pro Claude Code CLI no Windows host (acesso direto ao shell e à RTX 4070). Ciclo iterativo (Opção A) descartado por inviabilidade prática.
- **2026-04-27 — Filesystem montado entre Cowork e sandbox tem write-back não-determinístico.** Edits via tool persistem só "do lado Cowork"; bash via mount Linux vê arquivos truncados. Causa: provavelmente buffer write-back não-flushed do mount. Decisão: usar `cat > <file> <<'EOF'` via bash pra forçar persistência atômica em arquivos críticos (model.py, data.py, etc).

---

## Decisões arquiteturais

> Registrar aqui decisões irreversíveis ou de alto custo de reverter, com data e raciocínio.
> Não usar pra trade-offs pequenos. Critério: "se eu mudar isso depois, custa horas ou dias?"

### 2026-04-27 — STDP vetorizado em PyTorch puro (em vez de Brian2)

**Decisão:** implementar a regra STDP convolucional em `experiment_01_oneshot/model.py:ConvSTDPLayer.stdp_update` usando PyTorch puro (`F.unfold` + `einsum`), sem Brian2 nem simulador externo.

**Alternativa rejeitada:** usar Brian2 com `prefs.codegen.target = "cython"` pra gerar C++ e chamar via interface Python.

**Raciocínio:**

1. **GPU first-class.** A 4070 do projeto exige que o hot path rode em CUDA. Brian2 é CPU-only nativamente — pra GPU, precisaria de Brian2GeNN ou portar manualmente. PyTorch + CUDA é direto.
2. **Vetorização limpa.** A regra STDP convolucional reduz a dois `einsum` sobre tensores `unfold`-ed, sem loops Python. Custo computacional O(B × C_out × C_in × kH × kW × Hp × Wp) por timestep, que CUDA executa em fração de ms.
3. **Stack coesa.** Tudo (LIF, STDP, Hopfield, Conv) vive em PyTorch nn.Module. Zero impedância: salva/carrega via `state_dict`, integra com TensorBoard, debug com pdb.
4. **Brian2 fica reservado pra prototipagem de regras novas.** Se uma variante de STDP (eligibility traces, modulada por dopamina, etc.) precisar ser explorada, Brian2 é melhor pra prototipagem por sua sintaxe declarativa de equações diferenciais. Após validar, porta-se pra PyTorch.

**Custo de reverter:** ~1 semana (reescrever camada STDP em Brian2 + portar pretreino e eval).

**Marca de teste:** se a Semana 1 (sanity Diehl & Cook em MNIST) atingir ≥ 85%, a decisão se consolida. Se falhar, primeira hipótese a investigar é se a vetorização introduziu bug sutil — comparar contra implementação naive em loop Python.

### 2026-04-27 — Stack Python como scaffold inicial; port pra Julia adiado pra Fase 2

**Decisão:** todo o scaffold do experimento 01 está em Python, em vez de seguir literalmente a stack do `CONTEXT.md` §3 (Julia 1.10+ como linguagem principal).

**Raciocínio:**

1. **Ecossistema científico maduro.** Brian2, snnTorch, Norse, torchvision, PyTorch — anos de papers reproduzíveis em Python. Julia tem `SpikingNeuralNetworks.jl` mas comunidade muito menor.
2. **Princípio de não otimizar prematuramente** (`CONTEXT.md` §8.5). Antes de validar que o modelo conceitual funciona, custo de iterar em Julia (com menos exemplos, menos Stack Overflow, menos integração) é alto demais.
3. **Migração é factível depois.** Quando o modelo provar valor, porta-se pra Julia mantendo a interface (Flux.jl + WaspNet.jl). PyTorch tensor ops mapeiam quase 1-pra-1 em Julia.

**Custo de reverter:** ~3-4 semanas pra port completo da Fase 1 quando ela estiver consolidada.

**Marca de teste:** Fase 2 (meses 3-6) decide port baseado em sinais de tração. Se o experimento 01 falhar ou for inconclusivo, port pra Julia provavelmente não acontece — o tempo vai pra outro pilar.
