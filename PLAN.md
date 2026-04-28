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
- **2026-04-27 (sessão #4) — Homeostasis implementada, mecanicamente eficaz mas insuficiente sozinha.** Adaptive threshold de Diehl & Cook §2.3 codificado em `ConvSTDPLayer`. Iteração 1 com theta_plus=0.05 (paper) saturou theta em 267 e silenciou filtros (9.80% acc). Iteração 2 com theta_plus=0.0005 (100× menor, calibrado pro nosso regime de 100 ts) dá theta saudável (mean=2.5) e melhora distribuição de filtros [36,10,11,7,7,7,6,9,3,4] vs antes [24,23,11,9,3,5,7,13,1,4], mas acurácia continua ~16-17%. **Razão**: homeostasis força filtros a se distribuir → cada filtro vence menos → LTP/LTD imbalance re-emerge por filtro. Os dois problemas precisam ser atacados juntos. Próximas opções vivas em `BLOCKED.md`: combinar homeostasis + LTP/LTD ajustado simultaneamente; arquitetura conv real; validação Brian2; ou pular Semana 1.
- **2026-04-27 (sessão #5) — H_combo descartada: ataque simultâneo de homeostasis + LTP/LTD não destrava acurácia.** Testado com `theta_plus=0.0005, A_post=-0.001` (R=10): acurácia 13.76%, PIOR que homeostasis sozinha (16.39%) e que baseline sem nada (17.76%). Padrão sequencial revelador: quanto mais LTP relativo a LTD, mais colapso de filtros (theta std subiu 4×, 16% dos filtros nunca dispararam, distribuição [87,1,1,2,1,0,0,6,2,0]). Conclusão: homeostasis não compensa rich-get-richer no regime esparso PyTorch. Sistema parece ter instabilidade arquitetural irrecuperável via hiperparâmetros. Hipóteses vivas restantes: H_arch (mover pra Omniglot conv real) ou H_paper_replicability (validar contra Brian2). Não houve tuning porque tendência era monotonicamente errada.
- **2026-04-27 (sessão #6) — Semana 1 fechada como caso patológico, pivot pra Semana 2.** Decisão registrada em "Decisões arquiteturais": MNIST com kernel=28 é caso degenerado pra k-WTA (output espacial 1×1, todos filtros disputam mesmo slot). Stack validada via testes sintéticos + 5 sessões de diagnóstico (assignment/evaluate corretos, STDP atualiza pesos, homeostasis funciona). ROI baixo de continuar (~5h investidas, 0pp ganho acima de 17.76%). Semana 2 (Omniglot kernel=5 + pool) tem arquitetura multi-posição que naturalmente diversifica filtros — caminho conceitual mais alinhado com o experimento real. Sessão administrativa: nenhum experimento novo, só fechamento e preparação.
- **2026-04-27 (sessão #7) — Semana 2 Etapas 0-1: pipeline integra mas STDP em conv real satura pesos.** Etapa 0: smoke test confirmou pipeline end-to-end (Pixel kNN 45.76%, ProtoNet 85.88%, evaluate fecha) e expôs bug de pickling em `data.py:build_transforms` (lambda local) — corrigido com função module-level. Pretreino smoke (500 imgs) revelou colapso de layer 1 (w1=μ0.000). Etapa 1: criado `tests/test_spike_balance_omniglot.py`, mediu R1=1.51 e R2=0.69 (próximos do paper, NÃO o problema dominante). Trace step-by-step revelou mecanismo: A_pre=0.01 do paper produz Δw ≈0.86 por param/timestep — pesos saturam em 10 ts, depois oscilam e morrem (theta locked-out em ~2.9). Calibração A_pre=0.0001 (100× menor) resolve transitório curto (500 imgs OK, w1=0.114), mas em 5000 imgs satura novamente (w1=0.999, σ=0.011, todos os pesos = 1.0). Acurácia 5w1s recalibrada: 23.08% (z=0.4) com 500 imgs, volta pra 20.52% (chance) com 5000 imgs (saturação destrói sinal). Bloqueio novo: saturação STDP em escala média/longa. Próximo: testar hipóteses H_norm (normalização de Σw), H_mult (STDP multiplicativo), H_theta_omn (recalibrar theta_plus). Detalhes em `WEEKLY-2.md`.
- **2026-04-27 (sessão #7 encerramento) — Estratégia de pesquisa formalizada em `STRATEGY.md`.** Próximas 2-3 sessões focam em H_theta_omn e H_norm (custo baixo, alto ROI). Se não destravarem, considera pivot pra abordagem adjacente (Hopfield puro, meta-learning bio, prototypical com features esparsas) antes de cair pra Brian2. Cadência: sessão 60-90 min, max 2x/semana, atualizar Notas de iteração após cada uma; revisar STRATEGY.md se 3 sessões consecutivas sem progresso (sinal acima de chance).
- **2026-04-27 (sessão #8) — H_theta_omn descartada empiricamente.** Testado theta_plus em 3 valores (0.0005 baseline, 0.001 médio, 0.005 alto) com mesmo setup (5000 imgs / 1 epoch / k=1 WTA / A_pre=0.0001). Todos eval ≈ chance: 20.52%, 20.32%, 20.00% (este último com IC zero, predição constante porque theta=9 silenciou todos filtros). Causa raiz isolada: **tau_theta=1e7 (do paper) não permite decay efetivo no nosso regime** (500k oportunidades de update por treino vs decay desprezível) → theta cresce monotonicamente, qualquer theta_plus gera trade-off (silenciar todos OU não frear pesos). Restaurado theta_plus=0.0005. Próxima sessão: H_norm (recomendação STRATEGY.md) ou nova hipótese H_tau_theta (derivada deste diagnóstico). Sessões consecutivas sem sinal>chance: 1.
- **2026-04-28 (sessão #9) — H_tau_theta ✅ CONFIRMADA: primeiro sinal acima de chance.** Iter 1 com tau_theta=1e4 (1000× menor que paper) deu 5w1s=35.98% IC95%[35.17,36.79] z≈1.3 (+15.98 p.p. acima de chance). Surpresa metodológica: proxies estruturais (w1 saturado em 0.999 σ=0.001; theta cresceu monotonicamente até 20.68) FALHARAM, mas critério funcional bateu com folga. 3/3 verificações de robustez passaram: V1 eval seed=100 → 36.06%; V2 retrain seed=43 → 35.96%; V3 20w1s → 9.80% IC95%[9.58,10.01] z≈1.4 (chance=5%). Mecanismo conjecturado inicialmente (theta diferenciada carrega sinal) **refutado pelo V2**: seed=43 tem theta range ainda mais apertado [20.78,20.86] que seed=42 [20.55,20.68], mesma acurácia. Sinal real, mecanismo não identificado. Fixado tau_theta_ms=1e4 como decisão arquitetural (ver abaixo). Sessões consecutivas sem sinal>chance: 0 (resetado).
- **2026-04-28 (sessão #10) — Decisão sessão #9 REVERTIDA: sinal era ~80% arquitetural, não STDP.** 4 ablações: A1 (bypass _proj) → 36.12% (irrelevante); A2 (conv=0, theta_iter1) → 20.00% (chance, pipeline ok); **A3 (random U(0,1) sem treino) → 32.89%** (já entrega quase tudo!); A3b (random + theta_iter1) → 32.65% (theta inerte). Decomposição: +12pp = magnitude alta dos pesos (qualquer config que sature dá isso), +3pp = estrutura espacial sutil dos pesos saturados pelo STDP, ~0pp = theta treinada. Revertido tau_theta_ms=1e7 (paper). Lição: rodar A3 random ANTES de fixar decisão arquitetural — virou passo padrão. Sinal arquitetural (35.98%) continua existindo, sem regressão. Sessões consecutivas sem sinal>chance: 0.
- **2026-04-28 (sessão #11) — H_visualize: filtros saturados são "matched filter" pra estatística do Omniglot, não Gabor.** Análise visual e quantitativa sobre ckpts existentes (sem treino novo). Achado central: cosine centered off-diagonal entre filtros = **0.20 em L1, 0.55 em L2** (Iter 1) vs ~0 (random) — Iter 1 tem **estrutura espacial sistemática e compartilhada** entre filtros. Visualmente (delta = Iter1 − média(random)), os 8 filtros L1 têm o mesmo padrão: positivo no centro/superior, negativo no canto inferior direito (estatística do Omniglot — onde caracteres têm mais "tinta"). Conclusão: STDP no regime saturado descobre **1 protótipo médio** do dataset, replicado em todos os filtros — perde diversidade de features. Os 3 p.p. residuais vêm desse "matched filter trivial". H_norm/H_mult expectativa rebaixada: podem preservar variância sem destravar diversidade. Hipótese nova H_filter_diversity emerge (competição explícita entre filtros). Sessões consecutivas sem sinal>chance: 0.
- **2026-04-28 (sessão #12) — H_norm com target=0.3: destrava diversidade (cosine L2 0.55→0.04) mas magnitude 0.3 mata sinal (acc 20.04%, chance).** Adicionada flag `STDPConfig.norm_target_mean` (default None = desativado, backward compat). Treino com target=0.3: pesos preservaram μ=0.3 σ=0.1 (não saturaram), centered cosine off-diag baixou de 0.20→0.04 (L1) e 0.55→0.04 (L2). MAS acurácia 5w1s = 20.04% IC95% [20.01, 20.07] (chance). Controle random U(0, 0.6) com mesma magnitude: 20.33% (chance) — confirma que magnitude 0.3 é insuficiente pra atravessar threshold LIF, independente de treino. Insight: sinal de 35.98% e 32.89% dependia criticamente de magnitude alta (≥0.5) dos pesos. H_norm cumpriu mecanismo de diversidade, falhou por mag baixa. Próximo: H_norm_sweep com target ∈ {0.5, 0.6, 0.8} pra encontrar sweet spot. Restaurado norm_target_mean=None (estado conhecido); código de normalização mantido. Sessões consecutivas sem sinal>chance: 1.
- **2026-04-28 (sessão #13) — H_norm_sweep falha: mag maior não destrava, descoberta é o clamp em w_max.** Treinou com target ∈ {0.6, 0.8}, ambos chance: 20.07% (cosine cent L1=0.08, L2=0.02) e 20.11% (cosine cent L1=0.21, L2=0.02). Controles random U(0, 1.2)/U(0, 1.6) **clampados** [0,1] também chance (20.51%, 20.65%) — mostra que **distribuições com massa em w_max=1.0 matam sinal arquitetural** independente de mean ou treino. Random U(0, 1.0) puro (sessão #10) deu 32.89% porque virtualmente nenhuma amostra atinge 1.0 (massa em w_max desprezível). Conclusão mecanística: sinal arquitetural depende de (a) saturação total uniforme [Iter 1] ou (b) distribuição rica sem mass em w_max [random U(0,1)]. NÃO há sweet spot intermediário com clamp ativo. Implica nova direção H_no_clamp (w_max ≫ 1) ou H_mult (soft bound natural). norm_target_mean=None restaurado. Sessões consecutivas sem sinal>chance: 2.
- **2026-04-28 (sessão #14) — Sessão administrativa: reavaliação estratégica após 13 sessões.** Nenhum experimento. Adicionada seção "Pós-Sessão #13: Reavaliação" em STRATEGY.md documentando: (a) framing pós-#10 ("amplificar sinal arquitetural via STDP") não bate com aprendizados pós-#13 — sweet spot intermediário não existe na parametrização atual; (b) gap até meta CONTEXT.md (90% 5w1s) é de 54 p.p. acima do melhor produzido (35.98%); (c) 3 caminhos honestos com prós/cons documentados — A: mais tuning (H_no_clamp, H_mult, ROI baixo), B: mudança arquitetural ambiciosa (H_filter_diversity ou Brian2, ROI médio custo alto), C: pivot pra abordagem adjacente (Hopfield puro, meta-learning bio, prototypical com features esparsas). **Decisão A/B/C pendente pra próxima sessão** com cabeça fresca. Sessões consecutivas sem sinal>chance: 2 (esta administrativa não conta).
- **2026-04-28 (sessão #15) — Caminho C escolhido. C1 baseline Hopfield estabelece piso 56.28% 5w1s SEM TREINO.** STRATEGY.md atualizado com "Decisão Sessão #15". Novo script `c1_hopfield_baselines.py` (reutiliza HopfieldMemory existente, sem alterar model.py ou config.py). 3 variações: C1a Pixels+L2 → 50.17%/30.30%, **C1b PCA-32 → 56.28%/35.37%** (vencedor, z≈2.8 e 5.0), C1c RandomProj-32 → 41.23%/20.05%. **C1b bate Iter 1 STDP (35.98%) por +20 p.p. sem feature learning algum** — confirma diagnóstico estrutural pós-#13. C1a bate Pixel kNN (45.76%) por +4.4 p.p. via softmax pesado. Critério "Sucesso médio (50-70%)" atingido. C2/C3 podem agregar 10-20 p.p. → resultado defensável (alvo 65-75%). Sessões consecutivas sem sinal>chance: 0 (resetado).
- **2026-04-28 (sessão #16) — C1d (Hopfield + autoencoder MLP) DESCARTADO: features aprendidas via reconstrução não batem PCA-32.** Novo script `c1d_autoencoder_baseline.py`. Arquitetura fixa: 784→128→32→128→784, MSE pixel-wise, 30 epochs Adam lr=1e-3 batch=64 em 5000 imgs do background. AE-32 → 50.57%/30.59% (≈ C1a Pixels 50.17%, dentro do IC). Variação obrigatória bottleneck 64D: AE-64 → 52.64%/32.54% (sobe 2 p.p. mas ainda **3.6 p.p. abaixo de C1b PCA-32 56.28%**, ainda <54%). Pelo critério: ambos cenários "Pior" → C1d descartado. Insight: reconstrução pixel-wise preserva ruído tanto quanto sinal; PCA-32 descarta direções de baixa variância naturalmente. **PCA-32 é fronteira realista sem meta-objetivo.** Pra ultrapassar precisa C2 (meta-learning bio) ou C3 (ProtoNet+esparso). Sessões consecutivas sem sinal>chance: 0.
- **2026-04-28 (sessão #17) — C2 meta-Hebbian (Najarro & Risi 2020-style) BATE C1b PCA-32 por +6.94 p.p. (63.22% 5w1s, MÉDIO).** Novo script `c2_meta_hebbian.py`. Arquitetura fixa: encoder MLP 784→128→32 (tanh), pesos iniciais random fixos, plasticidade Hebbian local com 4 params (A,B,C,D) por peso meta-aprendidos via outer loop (Adam lr=1e-3 nos 417K params de plasticidade). Inner loop n_inner=5 sobre support, classificador prototípico (cosine sim) — substitui Hopfield. 5000 eps meta-train no background em 132s, eval 1000 eps. **C2 5w1s = 63.22% IC[62.41,64.06] z≈3.3** (vs C1b 56.28%); C2 20w1s = 37.30% z≈5.1 (vs C1b 35.37%). **Validação obrigatória passou:** sem inner loop, 38.02% (similar a random encoder); plasticidade contribui +25.21 p.p. real. Critério MÉDIO (60-70%). Gap até ProtoNet (85.88%) ainda 22.66 p.p. Sessões consecutivas sem sinal>chance: 0.
- **2026-04-28 (sessão #18) — Ablações sobre C2 revelam que termo Hebbian "puro" NÃO carrega o sinal; modelo é efetivamente linear.** Novo script `c2_ablations.py` (4 ablações × 5000 eps cada, 6.7 min total). **A1 só termo Hebbian (sem B,C,D): 39.39% (-23.83 p.p.)** — queda catastrófica, próximo de RandomProj. **A2 W iniciais zero: 63.97% (+0.75 p.p.)** — pesos iniciais irrelevantes, plasticidade reconstrói tudo. **A3 inner_loop=1: 53.05% (-10.17 p.p.)** — profundidade agrega ~10 p.p. **A4 encoder linear sem tanh: 64.07% (+0.85 p.p.)** — não-linearidade irrelevante. Insight central: o sinal vem dos termos modulatórios (B·pre, C·post, D bias), NÃO do termo Hebbian A·pre·post. Modelo é melhor descrito como "differentiable plasticity rule learning" (Miconi 2018) que "Hebbian bio-inspired". Sessões consecutivas sem sinal>chance: 0.
- **2026-04-28 (sessão #19) — C2-simplified satura em 64% 5w1s; termo Hebbian confirmado dispensável.** Novo script `c2_simplified.py`. Combinação dos achados da #18 (linear + W=0 + n_inner=10): **C2-simplified = 64.08%** IC[63.24, 64.92] z≈3.3 (vs baseline #17 63.22%). Composição não soma os ganhos marginais isolados — saturou em ~64%. n_inner=10 vs 5 só agrega +0.86 p.p. (vs +10 p.p. de inner=1→5 da #18). **Validação A1-invertida:** C2-simplified-no-A (treina só B,C,D, A=0 fixo) = 64.07% — delta -0.01 p.p. = ruído, **confirma definitivamente que termo Hebbian A é dispensável** (-25% params sem custo). Família C2 saturou na arquitetura testada. Pelo critério literal (60-65% = saturação aparente): **próxima sessão pivota pra C3** (ProtoNet+features esparsas). Sessões consecutivas sem sinal>chance: 0.
- **2026-04-28 (sessão #20) — C3 (ProtoNet + k-WTA esparso) SUCESSO FORTE: 93.10% 5w1s com 75% sparsity, ATINGE METAS DO CONTEXT.md numericamente.** Novo script `c3_protonet_sparse.py`. Reproduzido baseline ProtoNet com 5000 train eps = **94.55%** (vs 85.88% smoke test #7). 3 níveis de k-WTA no embedding 64D: **C3a k=32 (50%)=93.35%/81.87%, C3b k=16 (75%)=93.10%/80.72%, C3c k=8 (87.5%)=90.77%/75.44%** (5w1s/20w1s). Curva sparsity×acc plana até 75%, queda de só -1.45 p.p. com 75% das ativações zeradas. **Validação random+kWTA=37.60%** confirma ganho vem do treino, não da estrutura k-WTA. Critério SUCESSO FORTE atingido (≥80%). Atinge metas numéricas (≥90% 5w1s, ≥70% 20w1s), MAS não atinge restrições mecanísticas originais (C3 usa backprop+SGD, não plasticidade local). C3 é defensável academicamente como "esparsidade biológica compatível com metric learning", mas é incrementalismo sobre ProtoNet, não demonstração de "STDP funciona pra few-shot". Salto cumulativo: STDP 35.98% → C2 64.08% → C3b 93.10% (+57 p.p. desde Iter 1). Sessões consecutivas sem sinal>chance: 0.

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

### 2026-04-27 — Pivot: Semana 1 fechada como caso patológico, ir direto pra Semana 2 (Omniglot conv)

**Decisão:** declarar sanity check MNIST com kernel=28 como caso patológico documentado (não falha de stack), encerrar Semana 1, mover ativamente pra Semana 2 (pretreino STDP em Omniglot com arquitetura conv real, kernel=5 + pool).

**Alternativa rejeitada:** persistir em MNIST sanity até atingir ≥85% via (a) reimplementação em Brian2 (~1 semana), ou (b) tuning adicional de hiperparâmetros (espaço já exaurido em 5 sessões).

**Raciocínio:**

1. **MNIST com kernel=28 é caso degenerado pra k-WTA.** Output espacial (1,1) faz k-WTA competir sobre uma única posição — todos os filtros disputam o mesmo "slot". Isso cria instabilidade arquitetural (rich-get-richer ou colapso simétrico) que homeostasis não compensa, conforme demonstrado nas sessões #4 e #5.
2. **Stack está validada empiricamente.** 5 sessões de iteração descartaram bugs em assignment/evaluate (`tests/test_assignment.py` ✓), confirmaram que STDP atualiza pesos corretamente, validaram homeostasis mecanicamente (theta com variância controlada, distribuição de filtros uniformizável). O pipeline funciona — o que não funciona é o caso degenerado.
3. **Arquitetura Omniglot é fundamentalmente diferente.** kernel=5 + pool gera output espacial multi-posição (~28×28 com padding=2 na primeira layer). k-WTA por posição vira "1 vencedor entre 8/16 filtros por cada posição espacial", o que naturalmente diversifica filtros via specialização espacial — não há único slot disputado.
4. **ROI baixo de continuar em MNIST.** 5 sessões × ~1h cada = ~5h investidas pra ganhar 0pp acima de 17.76%. O experimento real (Omniglot one-shot) foi a tese desde o início — MNIST sanity era validação. Validação produziu sinal misto (stack OK, arch específica não), logo: prosseguir.

**Implicação para Semana 1:** declarada **concluída com ressalva** (`experiment_01_oneshot/PLAN.md` atualizado). Resultado consolidado: 17.76% MNIST, distribuição de filtros melhorada via homeostasis. Não atingiu meta 85% — mas a meta era validação de stack, e validação aconteceu por outras vias (testes sintéticos + diagnóstico mecânico).

**Custo de reverter:** zero. Código permanece. Se Semana 2 falhar de forma que aponte de volta pra MNIST sanity, o trabalho é só rodar `sanity_mnist.py` novamente.

**Marca de teste:** se Semana 2 (Omniglot 5-way 1-shot) atingir > 50% (vs chance 20%), confirma que kernel=28 era de fato o problema. Se Semana 2 também colapsar, o caminho honesto é Brian2 (Opção C documentada em `BLOCKED.md` final da Semana 1).

### 2026-04-27 — Adaptive threshold homeostático (Diehl & Cook 2015 §2.3) adicionado a ConvSTDPLayer

**Decisão:** adicionar buffer `theta` (shape `(out_channels,)`) em `ConvSTDPLayer`, usar `v_thresh_eff = v_thresh + theta` como threshold por filtro, atualizar theta após cada timestep com `theta += theta_plus * spike_per_filter; theta *= exp(-dt/tau_theta)`. Persiste entre imagens.

**Alternativa rejeitada:** depender só de k-WTA + STDP rebalanceado (sessão #3 isolou que isso oscila entre failure modes — pesos morrem ou filtros colapsam, não há ratio LTP/LTD que resolva sozinho).

**Raciocínio:**

1. **Aderência a Diehl & Cook 2015 §2.3.** Adaptive threshold é como o paper-base implementa diversidade de filtros. Sem isso, k-WTA + STDP é provadamente instável.
2. **Mecanicamente eficaz.** Sessão #4 validou que com theta_plus calibrado (0.0005, vs 0.05 do paper — ajustado pra nosso regime de ~100 timesteps), distribuição de filtros fica significativamente mais uniforme: [24,23,11,9,3,5,7,13,1,4] → [36,10,11,7,7,7,6,9,3,4].
3. **Limitação reconhecida.** Acurácia não destrava só com homeostasis (continua ~17%). Próximo bloqueio: combinar homeostasis com LTP/LTD rebalanceado, ou mudar arquitetura, ou validar contra Brian2.

**theta_plus calibrado pra nosso regime:** valor do paper (0.05) saturou theta em 267 (filtros silenciaram). Reduzido pra 0.0005 (100× menor) pra refletir que nosso regime gera ~100× mais spikes por filtro por unidade de theta_plus que o setup de Diehl & Cook (eles usam ~7000 ts/imagem com refractory longo; nós, 100 ts × k=1 WTA).

**Custo de reverter:** ~30 min (remover buffer theta + simplificar forward). Mantém compatibilidade.

**Marca de teste:** estado consolidado 17.76% (igual a sem homeostasis) com distribuição de filtros melhor. Decisão se consolida quando combinada com solução pro LTP/LTD imbalance.

### 2026-04-28 — `tau_theta_ms=1e4` REVERTIDA na sessão #10 (decisão prematura, mecanismo era arquitetural)

**Decisão original (sessão #9):** fixar `tau_theta_ms=1e4` por produzir 35.98% 5w1s (primeiro sinal>chance).

**Reversão (sessão #10, mesmo dia):** restaurado `tau_theta_ms=1e7` (paper). Decisão original era prematura.

**Raciocínio da reversão (3 ablações + 1 controle):**

| Ablação | Setup | Acurácia 5w1s | Implicação |
|---|---|---|---|
| Baseline Iter 1 | tau_theta=1e4 treinado | 35.98% | sinal real |
| A1: bypass `_proj` | flatten 784D direto | 36.12% | _proj é irrelevante |
| A2: conv=0, theta_iter1 | só leakage | 20.00% (chance) | conv é necessária |
| A3: random U(0,1), theta=0 | sem treino algum | 32.89% | **+12pp já sem treino** |
| A3b: random U(0,1) + theta_iter1 | sem pesos treinados | 32.65% | theta inerte |

**Decomposição empírica** dos 16 p.p. acima de chance:
- **+12 p.p.** = magnitude alta dos pesos (Iter 1 satura em 0.999, random U(0,1) é semelhante em magnitude)
- **+3 p.p.** = estrutura espacial sutil dos pesos saturados pelo STDP (Iter 1 vs random U(0,1) controlando theta)
- **~0 p.p.** = theta treinada / homeostasis (A3 vs A3b dentro do IC)

**O que destrava o sinal de 35.98% NÃO é tau_theta=1e4 calibrado.** É: pesos saturando em magnitude alta (qualquer config que produz saturação serve) + Poisson + Hopfield + estrutura esparsa do Omniglot. tau_theta=1e4 só permite que pesos saturem mais cedo no treino; não cria sinal — habilita saturação que outros valores também habilitariam.

**Lição aprendida (registrada aqui pra não repetir):** quando achar uma config que dá sinal acima de chance, **rodar A3 (random sem treino) ANTES de declarar decisão arquitetural**. Custou pouco, e teria evitado a decisão prematura. Vai virar passo padrão de validação.

**Estado pós-reversão:** `tau_theta_ms=1e7` (paper) restaurado em config.py. Sinal arquitetural (35.98%) **continua existindo** — não há regressão (qualquer config que sature produz). Ablações ficam documentadas em WEEKLY-2.md sessão #10.

### 2026-04-28 — Validação de descoberta via random baseline (passo padrão a partir de #10)

**Decisão:** sempre que uma nova config produzir sinal acima de chance pela primeira vez, **antes de fixá-la como decisão arquitetural**, rodar baseline com pesos random sem treino (range similar ao da config) pra isolar contribuição do treino vs arquitetura.

**Motivação:** sessão #10 mostrou que decisão arquitetural sessão #9 foi prematura — A3 random teria mostrado em ~10 min que o sinal era ~80% arquitetural antes de gastar a sessão #9 declarando "primeiro sinal real" e fixando `tau_theta=1e4`.

**Implementação:** `tests/ablate_random_weights.py` reutilizável. Critério: se random sem treino entrega ≥80% do sinal, decisão arquitetural fica em standby até descobrir o que o treino realmente agrega.

**Custo de reverter:** zero (é só protocolo, não código).

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
