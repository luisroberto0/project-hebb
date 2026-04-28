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
- **2026-04-27 — Sanity check MNIST bloqueado após 3 iterações de k-WTA.** Melhor resultado: 17.76% acurácia (k=1 WTA) com distribuição balanceada de labels, vs meta 85%. Causa hipotética: combinação de (1) número de filtros insuficiente (100 vs 400-6400 do paper), (2) pretreino curto (5k imgs × 1 epoch vs 60k × 3), (3) implementação de WTA diverge do paper (masking vs condutância). Decisão: bloqueio temporário, documentado em `BLOCKED.md`. Aguarda input humano para escolher entre: escalar filtros+dados, aumentar max_rate Poisson, ou reimplementar inibição via condutância.
- **2026-04-27 (sessão #3) — Auditoria e rebalance LTP/LTD descartam 3 hipóteses, isolam causa raiz.** Etapa 1: `tests/test_assignment.py` valida `assign_labels` e `evaluate` com 3 casos sintéticos (100% perfeito, 10% chance, 100% sinal fraco) — pipeline de classificação está OK. Etapa 2: `tests/test_spike_balance.py` mede razão pré:pós-spikes = 10.1; tentativas de rebalanceamento (R=10 e R=3) mostram trade-off estrutural: LTP<LTD mata pesos, LTP>LTD colapsa filtros (rich-get-richer). **Causa raiz isolada: falta de adaptive threshold homeostático** (Diehl & Cook 2015 §2.3) que força filtros a disparar igualmente. Próxima sessão: decidir entre implementar homeostasis (H_homeostasis), pular pra Omniglot (H_arch) ou validar contra Brian2 (H_paper_replicability) — detalhes em `BLOCKED.md`.

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

### 2026-04-27 — Inibição lateral em ConvSTDPLayer via k-WTA na dinâmica LIF (não decay de pesos)

**Decisão:** implementar inibição lateral como k-winner-take-all (k=1) durante a dinâmica LIF em `experiment_01_oneshot/model.py:ConvSTDPLayer.forward`, aplicado por patch espacial. Remover completamente o decay de pesos pós-STDP que estava em `stdp_update`.

**Alternativa rejeitada:** manter decay de pesos (`self.conv.weight.data -= 1e-4 * ...`) e apenas aumentar fator de 1e-4 para 1e-2.

**Raciocínio:**

1. **Fidelidade ao paper base.** Diehl & Cook 2015 §Methods descreve inibição lateral como "conexões inibitórias entre neurônios excitadores", implementada via spikes que suprimem membrana de competidores durante a integração LIF — **não** como ajuste de pesos após STDP.
2. **Diagnóstico do colapso.** Semana 1 mostrou que decay de pesos com fator 1e-4 é ineficaz: todos os 100 filtros colapsaram para classe 0 (distribuição [100,0,0,0,0,0,0,0,0,0]). Aumentar o fator seria band-aid sem justificativa teórica.
3. **k-WTA é computacionalmente eficiente.** Por patch espacial (H×W), calcular `argmax` ao longo da dimensão de filtros (C_out) e mascarar spikes tem custo O(C_out × H × W), compatível com GPU.
4. **Generalização para arquiteturas convolucionais.** k-WTA por patch permite que diferentes regiões da imagem ativem filtros diferentes, capturando features locais — essencial para Omniglot onde caracteres têm múltiplas sub-estruturas.

**Implementação:**
```python
# Em ConvSTDPLayer.forward, após calcular mem e antes de gerar spikes:
spikes_raw = (mem >= self.cfg.lif.v_thresh).float()
# k-WTA: só o filtro com maior membrana por posição espacial dispara
max_filter_idx = mem.argmax(dim=1, keepdim=True)  # (B, 1, H, W)
wta_mask = torch.zeros_like(mem).scatter_(1, max_filter_idx, 1.0)
spikes_out = spikes_raw * wta_mask
```

**Custo de reverter:** ~2-3 horas (reverter para decay de pesos e re-tunar fator).

**Marca de teste:** se o sanity check MNIST atingir ≥ 70% com distribuição de labels razoavelmente balanceada (nenhuma classe com 0 filtros, nenhuma com > 40), a decisão se consolida. Se falhar novamente, próxima hipótese é STDP puro sem inibição.
