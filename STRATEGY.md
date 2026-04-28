# Estratégia da pesquisa STDP+Hopfield

## Decisão tomada em 2026-04-27 (sessão #7, encerramento)

Combinar tuning + opção de pivot:

1. **Próximas 2-3 sessões:** testar H_theta_omn e H_norm (custo baixo, alto ROI)
2. **Se H_theta_omn e H_norm não destravarem:** considerar pivot pra abordagem adjacente (Hopfield puro, meta-learning bio, ou prototypical com features esparsas)
3. **Brian2 (validação contra paper original) fica como último recurso** — alto custo (~1 semana), só justificável se pivot também falhar

## Critério pra cada decisão futura

- Sessão = 60-90 min, max 2x/semana
- Após cada sessão: atualizar PLAN.md "Notas de iteração"
- Se 3 sessões consecutivas sem progresso (sinal acima de chance), revisar STRATEGY.md

---

## Pós-Sessão #9 (2026-04-28): Sinal real, mecanismo a investigar

### O que foi confirmado

`tau_theta_ms=1e4` produz **primeiro sinal acima de chance do projeto**:

- 5w1s = 35.98% IC95% [35.17, 36.79], z≈1.3 (+15.98 p.p. acima de chance)
- 20w1s = 9.80% IC95% [9.58, 10.01], z≈1.4 (+4.80 p.p. acima de chance)

**Robusto em 3 verificações ortogonais:**
- V1 (eval seed=100, mesmo ckpt): 36.06% — não é ruído de seed de eval
- V2 (retrain seed=43, eval seed=42): 35.96% — não é combinação específica de seeds
- V3 (20w1s, escala dificuldade): mantém z≈1.4 — sinal escala

Decisão arquitetural fixada em PLAN.md: `tau_theta_ms=1e4`.

### O que NÃO foi confirmado: mecanismo

Pesos saturam em μ=0.999 σ=0.001 (filtros teoricamente indistinguíveis), theta range é ~0.1 entre filtros, ainda assim sinal aparece. **Não sabemos por quê.**

Conjectura inicial "theta diferenciada carrega sinal" foi **refutada** pela V2: seed=43 produziu theta range ainda mais apertado [20.78, 20.86] que seed=42 [20.55, 20.68], com acc idêntica.

Hipóteses vivas sobre o mecanismo:

- **H_init_bias:** initialization bias preservado na saturação (qual filtro satura primeiro / ordem temporal codifica algo discriminativo, mesmo que valores finais sejam quase iguais)
- **H_lif_dynamics:** spike timing residual via dinâmica LIF (input × peso saturado + reset + ruído gera spike trains discriminativos mesmo com pesos ~iguais)
- **H_pooling_amp:** σ=0.001 amplificado pela não-linearidade do segundo layer + pooling
- **H_artifact:** sinal é artefato de eval/dataset (improvável dado V1/V3, mas precisa ser descartado)

### Próxima sessão: dedicada exclusivamente a investigar mecanismo

**Não escalar, não tunar, não modificar config até mecanismo ser identificado.** A regra é honestidade metodológica: se o sinal vem de artefato, "amplificá-lo" é desperdício.

3 testes de ablação, todos no mesmo checkpoint Iter 1 (`stdp_model_iter1_seed42.pt`):

| Ablação | Como | O que testa | Predição se sinal real |
|---|---|---|---|
| **A1 — `_proj` zerada** | substituir projeção final (embedding_dim=64) por identidade ou zeros antes do Hopfield | sinal vem do feature space pós-conv ou da projeção? | acc cai pra chance se proj é o canal; mantém ~36% se conv carrega |
| **A2 — zero conv** | substituir `layer1.conv.weight` e `layer2.conv.weight` por zeros, manter theta treinada | pesos importam? | acc cai pra chance (zero spikes propagam) |
| **A3 — random conv** | substituir pesos treinados por re-initialization aleatória (`w_init_low/high`), manter theta treinada | pesos *treinados* importam, ou só a presença de qualquer estrutura conv? | acc cai pra chance se pesos treinados carregam sinal; mantém ~36% se theta sozinha basta |

### Critério de decisão pós-ablação

- **Sinal sobrevive aos 3 testes (acc cai apropriadamente em A2; cai ou se mantém conforme predição em A1/A3, mas o resultado é interpretável e consistente):** sinal é real, mecanismo identificado (ou parcialmente identificado), decisão `tau_theta=1e4` se consolida, próxima sessão escala/tuna.

- **Algum teste de ablação atinge ~36% sem o componente ablacionado:** "descoberta" é artefato (a parte ablacionada não era necessária pra produzir o sinal — sinal está em outro canal não-conv, possivelmente eval ou pipeline). Reverte decisão `tau_theta=1e4`, restaura `tau_theta=1e7`, marca H_tau_theta como "sinal medido mas mecanismo artefatual" em vez de descartada-sucesso. Próxima sessão pivota pra H_norm.

- **Resultado intermediário/ambíguo:** documenta padrão observado, **não fixa decisão**, agenda mais ablações antes de seguir.

---

## Pós-Sessão #10 (2026-04-28): Reversão e nova framing

### O que aprendemos

As 3 ablações + A3b mostraram que o sinal de 35.98% (Iter 1) é **dominantemente arquitetural**, não fruto do STDP:

| Componente | Contribuição (5w1s) |
|---|---|
| Magnitude alta dos pesos (saturação ~1.0) | +12 p.p. |
| Estrutura espacial sutil pós-STDP | +3 p.p. |
| Theta treinada / homeostasis | ~0 p.p. |
| **Total acima de chance** | **16 p.p.** |

Decisão sessão #9 (`tau_theta=1e4` como arquitetural) foi revertida. Não foi a calibração de homeostasis que destravou sinal — foi a saturação dos pesos (que qualquer config produzindo saturação habilita) interagindo com a esparsidade do Omniglot e a memória Hopfield.

### Nova framing pra próximas sessões

**Pergunta antiga:** "como fazer STDP aprender features de Omniglot?" (premissa: STDP é o motor)

**Pergunta nova:** "existe um sinal arquitetural baseline (~33% sem treino, ~36% com STDP saturado). Como amplificar esse sinal de 33-36% pra 70%+? STDP precisa contribuir mais, ou o caminho é outro?"

Subperguntas concretas:
1. **O que dá os +12 p.p. de magnitude?** É a magnitude per se, ou a interação magnitude × Hopfield × Omniglot esparso? Testar em FashionMNIST (input denso) pode separar.
2. **O que dá os +3 p.p. residuais do STDP?** Os pesos saturados em 0.999 σ=0.001 ainda têm estrutura espacial discriminativa — qual? Visualizar filtros pode revelar.
3. **Como fazer STDP contribuir com mais que 3 p.p.?** Hipóteses: (a) impedir saturação total via H_norm/H_mult pra preservar variância informativa, (b) treinar com mais imagens quando saturação é evitada, (c) revisar arquitetura (mais filtros, kernel maior, sem pool).

### Roadmap revisado pra Semana 2

Hipóteses vivas (atualizadas pós-#10):

| Hipótese | Custo | Justificativa pós-#10 |
|---|---|---|
| **H_norm** (Σw=1 por filtro) | ~30 min | Impede saturação → preserva variância informativa → STDP pode contribuir além dos 3 p.p. residuais |
| **H_mult** (STDP multiplicativo) | ~1h | Soft bound natural → pesos não saturam em 1.0 → estrutura espacial sobrevive |
| H_visualize (filtros + ativações) | ~1h | Antes de tunar, entender o que os pesos saturados estão realmente codificando — visualização pode mudar prioridades |

**Recomendação:** H_visualize **antes** de H_norm/H_mult. Custa pouco e pode mudar interpretação. Se filtros saturados estão capturando algo Gabor-like apesar de σ=0.001, otimização vale. Se são ruído puro, abordagem precisa mudar mais radicalmente.

### Protocolo atualizado

**Novo passo padrão antes de fixar decisão arquitetural:** rodar ablação random sem treino (`tests/ablate_random_weights.py`) com magnitude similar à da config nova. Se random entregar ≥80% do sinal, decisão arquitetural fica em standby até entender o que o treino agrega.

Custo: ~10 min. Teria evitado a decisão prematura da sessão #9.

### Critério de revisão

Mantém: 3 sessões consecutivas sem sinal>chance dispara revisão de STRATEGY.md. Pós-#10, o contador continua em 0 — sinal arquitetural existe e é reproducível.

---

## Pós-Sessão #13 (2026-04-28): Reavaliação após 13 sessões

### Status do framing "Pós-#10"

A framing pós-#10 propunha: "existe sinal arquitetural baseline (~33%); como amplificar isso de 33% pra 70%+ via STDP?". Recomendou H_visualize → H_norm → H_norm_sweep como cadência barata.

**Após executar essa cadência (#11, #12, #13), a framing precisa ser revisitada.**

O que aprendemos:

- **#11 (H_visualize):** filtros saturados são "matched filter trivial" pra estatística do Omniglot, não Gabor — STDP no regime saturado **converge todos os filtros pra 1 protótipo médio replicado**. Não é seletividade de features; é compressão de dataset.
- **#12 (H_norm 0.3):** normalização destrava diversidade dramaticamente (centered cosine L2: 0.55→0.04) **mas magnitude baixa mata atividade pós-LIF** → chance.
- **#13 (H_norm sweep 0.6, 0.8 + controles random clampados):** magnitudes maiores também dão chance. Diagnóstico via controles: **distribuições com massa em `w_max=1.0` (clamp) matam o sinal arquitetural**, independente de mean ou treino.

**Padrão consolidado das 3 sessões:** sinal arquitetural depende de (a) saturação total uniforme [Iter 1] **OU** (b) distribuição rica sem mass em w_max [random U(0,1)]. **Sweet spot intermediário NÃO existe** com a parametrização padrão (`w_min=0`, `w_max=1.0`, k-WTA por posição).

A framing "amplificar sinal arquitetural via STDP" assumia continuidade entre os dois polos. Não há. STDP padrão **não tem como produzir um regime intermediário discriminativo** dentro dessa parametrização — qualquer configuração que evita ambos os polos cai pra chance.

### Realidade observada após 13 sessões

**Pergunta original do projeto** (CONTEXT.md §4): "como SNN+STDP+Hopfield atinge ≥90% em Omniglot 5w1s sem backprop end-to-end?"

**Realidade observada:**
- Sinal acima de chance: **35.98% (z≈1.3)** — o melhor produzido em 13 sessões.
- Decomposição: ~32 p.p. arquitetural (random U(0,1)) + ~3 p.p. matched filter trivial via STDP saturado + ~0 p.p. de tudo o mais (theta, calibração de tau, normalização de Σw).
- Gap até a meta: **54 p.p.** acima do que a abordagem atual produz (35.98% → 90%).
- Custo de cada experimento: ~60-90 min de sessão; 13 sessões já gastas.
- Padrão das últimas 3 sessões: hipóteses razoáveis a priori → resultado em chance ou em magnitude do sinal arquitetural.

**Diagnóstico estrutural:** a combinação STDP aditivo + k-WTA por posição + clamp hard em [w_min, w_max] tem **barreira estrutural identificada**. Hiperparâmetros dentro dessa parametrização não atravessam a barreira — sessões #4, #5, #8, #9, #12, #13 confirmam.

### Três caminhos honestos (pendente decisão na próxima sessão)

#### A) Mais 1-2 sessões de tuning (H_no_clamp, H_mult)

**O que seria:**
- H_no_clamp: aumentar `w_max` pra ~5.0 e reativar H_norm. Se sinal voltar com mag média sem mass em w_max, confirma diagnóstico do clamp como gargalo único (não estrutural).
- H_mult: STDP multiplicativo (Δw ∝ (w_max − w)). Soft bound natural sem clamp hard.

**Pros:**
- Custo barato (~30-60 min cada), código de normalização já está em `model.py` com flag.
- H_no_clamp testa diretamente a hipótese mecanística de #13 (clamp em w_max é o problema).
- Se H_no_clamp destravar sinal intermediário (ex 45-55%), confirma que a barreira é o clamp, não estrutural.

**Cons:**
- Padrão das últimas 3 sessões (#11, #12, #13) sugere que mais tuning dentro da mesma parametrização tem ROI baixo. Cada hipótese parecia óbvia a priori e não entregou.
- Mesmo se H_no_clamp/H_mult destravarem +10-15 p.p., gap até 90% (atual 35.98% → meta 90%) ainda exige ganhos de 40+ p.p. — improvável dentro do mesmo regime.
- Risco de sunk cost: justificar continuidade só porque "já investimos tanto" em vez de avaliar honestamente o que cada nova sessão acrescenta.

#### B) Mudança arquitetural ambiciosa (H_filter_diversity ou Brian2)

**O que seria:**
- H_filter_diversity: penalidade explícita de similaridade entre filtros no objetivo STDP, ou ortogonalização periódica. Força diversidade independente de magnitude/clamp.
- Brian2 port: reproduzir Diehl & Cook 2015 fielmente em Brian2 (CPU-only, ~1 semana). Resolve "nossa implementação PyTorch tem bug sutil?" e dá baseline de paper validado.

**Pros:**
- Atacam causa raiz (convergência pra mesmo protótipo) em vez de tunar margens.
- Brian2 alinha com Diehl & Cook §2.3 fielmente — inibição via membrana, não weights — o que sessões #4-#5 isolaram como diferencial não replicado.
- Se H_filter_diversity funcionar, é resultado positivo defensável academicamente (mecanismo novo, não só hyperparam tuning).

**Cons:**
- Custo alto. H_filter_diversity é ~1-2h de implementação + experimentos; Brian2 é ~1 semana real.
- Risco de implementação: STDP+homeostasis em Brian2 é não-trivial; mais 1 semana sem garantia de que Brian2 vai entregar 90%.
- Sucesso parcial provável: ganhos +10-20 p.p. mas não os 54 p.p. necessários pra meta.

#### C) Pivot pra abordagem adjacente

**O que seria:**
- Hopfield puro com features pré-extraídas: usar embeddings de modelo simples (ex: PCA de imagens, pixel-kNN sobre patches) + Hopfield Moderno. Aceita que STDP pra feature learning não está funcionando.
- Meta-learning bio-inspirado: MAML-like com regras de plasticidade local em vez de gradient descent (literatura mais recente: Najarro 2020, Pedersen 2023).
- Prototypical Networks com features esparsas: replica ProtoNet (que já dá 85.88% em Omniglot baseline) mas com camada esparsa neuro-inspirada.

**Pros:**
- Aceita honestamente que reproduzir Diehl & Cook 2015 fielmente em PyTorch puro está custando ROI baixo.
- Mantém a missão (CONTEXT.md §1: capacidades que LLMs não têm — one-shot, continual learning) sem prender em STDP específico.
- Hopfield puro é fundamentalmente bem caracterizado (Ramsauer 2020) — provavelmente atinge 70%+ rapidamente com features adequadas.
- Permite avaliar honestamente: o problema é STDP especificamente, ou é a combinação STDP+arquitetura?

**Cons:**
- Pivot perde a tese central original ("STDP é a regra real do cérebro, vamos provar que funciona em ML").
- Curva de aprendizado nova pra abordagem adjacente.
- Risco de virar incrementalismo de baselines existentes (ProtoNet+ruído ≠ contribuição clara).
- Sem garantia que adjacente também não trava em barreira similar.

### Nova framing pra próxima fase

> **STDP+k-WTA+threshold em PyTorch puro tem barreira estrutural identificada após 13 sessões.** O melhor que essa parametrização produz é ~36% (5w1s), dominantemente arquitetural, com STDP contribuindo ~3 p.p. via matched filter trivial. Sweet spot intermediário entre saturação total e distribuição rica sem clamp não existe na parametrização atual.
>
> Decisão estratégica entre A (mais tuning), B (mudança arquitetural ambiciosa) ou C (pivot pra abordagem adjacente) **fica pra próxima sessão com cabeça fresca**, depois de revisar este documento.

### Critério de revisão atualizado

- Sessões consecutivas sem sinal>chance: **2** (#12, #13). #14 é administrativa (não conta). Se a decisão da próxima sessão for caminho A e a sessão seguinte continuar em chance, contador chega a 3 → revisão obrigatória de STRATEGY.md (já estamos fazendo agora preventivamente).
- Recomendação implícita ao Luis: independente do caminho A/B/C escolhido, **definir antes do experimento qual sinal seria suficiente pra continuar** vs. qual sinal motivaria pivot. Critérios numéricos a priori, igual sessões #9/#10/#12/#13.
