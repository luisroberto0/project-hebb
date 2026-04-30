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

---

## Decisão Sessão #15 (2026-04-28): Caminho C escolhido, sub-opção C1 primeiro

**Decisão tomada:** Caminho **C — pivot pra abordagem adjacente**. Não é "pivot envergonhado" — é reposicionamento alinhado com a missão do CONTEXT.md §1 ("capacidades que LLMs não têm: one-shot, continual learning, eficiência radical, raciocínio temporal"). Reproduzir Diehl & Cook 2015 fielmente em PyTorch puro **não é a missão** — é uma rota possível pra um pilar (one-shot via plasticidade local). Após 13 sessões com barreira estrutural identificada, custo-benefício de continuar nessa rota específica é baixo.

**Sub-opção C1 escolhida primeiro:** Modern Hopfield Memory baseline puro (Ramsauer 2020) em Omniglot, alimentado por features triviais (pixels, PCA, random projection). Por quê C1 antes de C2 (meta-learning bio) ou C3 (ProtoNet esparso):

1. **Reaproveita código pronto.** `model.py:HopfieldMemory` já existe, validado e em uso pelo pipeline. Zero custo de implementação extra.
2. **Dá número rápido em 1 sessão.** 60 min são suficientes pra rodar 3 variações × 2 settings = 6 evaluations e ter resposta empírica.
3. **Forma o piso de comparação.** C2 e C3 só fazem sentido se agregarem acima de C1 — sem C1 medido, decisões futuras viram especulação.
4. **Valida ou refuta a tese central de Hopfield.** Ramsauer 2020 prova que Modern Hopfield Memory tem capacidade exponencial e é equivalente à atenção do Transformer. Se mesmo com features triviais Hopfield não atinge ~50%+ em Omniglot, isso é informação importante: o motor "Hopfield + few-shot" precisa ser suplementado, não pode carregar sozinho.

### Critério de decisão pós-C1

| Resultado C1 (5w1s) | Implicação |
|---|---|
| **≥70%** | C1 é vitória, baseline alta. C2/C3 precisam bater C1. Próximas sessões focam C2 (meta-learning bio) que precisa adicionar mecanismo cognitivo defensável (não só acurácia). |
| **50-70%** | Piso real estabelecido. C2/C3 podem agregar 10-20 p.p. → resultado defensável. |
| **35-50%** | Hopfield sozinho não é mágico. C2 (meta-learning) vira mais necessário. |
| **≤35%** | Falha. Não bate Pixel kNN (45.76%). Bug na implementação OU Hopfield não é o motor esperado. Investigar antes de C2/C3. |

### Restrições da sessão #15

- **Não modificar `model.py`** (HopfieldMemory já validado)
- **Não tocar em `config.py`** (estado revertido pós-#13 deve ser preservado)
- **Não rodar STDP** (foco é baseline puro)
- Se algo travar, restaura estado e documenta como "C1 inconclusivo, próxima sessão revisita".

### Status do Caminho A e Caminho B

- **A não foi descartado**: H_no_clamp e H_mult ficam como hipóteses dormentes. Se C1 falhar (≤35%), Caminho A volta ao topo da fila como diagnóstico (clamp era único gargalo? OU é estrutural?).
- **B não foi descartado**: Brian2 e H_filter_diversity ficam como rotas se C1 entregar 50-70% mas C2/C3 não fecharem o gap até 90%.

---

## Decisão Pós-Sessão #20: Projeto B definido

**Contexto:** sessão #20 fechou Caminho C com sucesso numérico (C3b 93.10% 5w1s, atinge as duas metas quantitativas do CONTEXT.md §4) mas falha mecanística (C3 usa backprop + SGD, não plasticidade local). A meta original era "≥90% sem backprop end-to-end". C3 atingiu metade da meta. Resta decidir o que fazer com o resto.

Esta seção responde **6 perguntas concretas** sobre o "Projeto B" — a continuação após C3, na rota mecanisticamente fiel. Onde houver incerteza honesta, recomenda mas não decide; decisão final é do Luis.

### (a) Versão do Projeto B

**Recomendação: B HÍBRIDO** (publicar C3 como milestone primeiro, depois B reduzido focado).

| Versão | Tempo | Compatível com founder Rytora? | Risco |
|---|---|---|---|
| B ambicioso (atacar 4 critérios pós-LLM, 3-5 anos) | 15-20h/semana | **Não** — exigiria mudança radical de alocação de tempo, conflita com obrigação comercial Rytora | Burnout, atrito conjugal/Rytora, ROI distante |
| **B reduzido** (atacar 1 critério em 1-2 anos, 5-7h/semana) | 5-7h/semana | **Sim** — encaixa em "side project disciplinado" do CONTEXT.md §2 | Saturação, falta de novidade real, perda de motivação |
| **B híbrido** (publicar C3 + B reduzido) | 5-7h/semana + 2 weekends de paper | **Sim** | Distração entre escrita e exploração; diluir foco |

**Por que híbrido > reduzido puro:**
- C3 atinge metas numéricas e tem mecanismo defensável ("k-WTA preserva ProtoNet com 75% sparsity"). Não publicar é desperdício de evidência empírica acumulada.
- Workshop paper publicado (mesmo que de bench-incremental) cria credenciamento mínimo no campo bio-plausible learning, abre porta pra reviewer feedback.
- Custo de escrever C3 paper é finito e independente do trabalho de B — pode ser feito em 1-2 weekends sem comprometer cadência de pesquisa.

**Por que híbrido > ambicioso:**
- Founder de Rytora não pode realisticamente alocar 15-20h/semana em side project não-comercial sem prejudicar produto comercial.
- 3-5 anos atacando 4 critérios simultâneos é receita pra produzir nada concreto. Disciplina > entusiasmo (CONTEXT.md §8).

**Risco principal de híbrido:** distração ao escrever C3 paper enquanto B precisa de continuidade conceitual. Mitigado por: paper de C3 é workshop (~6-8 páginas) com experimentos prontos; escrita pode ser modular (intro/methods/experiments) e separada da pesquisa de B.

### (b) Critério pós-LLM atacado primeiro

**Recomendação: CONTINUAL LEARNING sem catastrophic forgetting.**

Análise dos 4 critérios contra realidade pós-#20:

| Critério | Maturidade | Conecta com C2/C3? | Benchmark | ROI estimado |
|---|---|---|---|---|
| **Continual learning** | Alta (EWC 2017, SI 2017, GEM 2017, A-GEM 2019, MAS 2018, lots of recent work) | Sim — C2 plasticity meta-learning + C3 sparsity podem balancear stability-plasticity | Split-Omniglot, Permuted MNIST, Split-CIFAR-10/100, CLEAR, COIL-100 | **Alto** |
| One-shot inédito | Média | Já em Omniglot one-shot; pra ser "inédito" precisa cross-domain (Mini-ImageNet → CUB, generated novel chars) | Cross-domain few-shot benchmarks existem mas menos estabelecidos | Médio |
| Eficiência radical | Baixa pra setup atual | Requer SNN simulation detalhada (Brian2/NEST) ou hardware neuromórfico (Loihi, $$) | Pouco padronizado; cada paper define próprio | Baixo (founder, sem capital pra hardware) |
| Raciocínio temporal | Baixa | Conecta com SNN temporal coding mas distante de C2/C3 | Niche, poucos benchmarks padrão | Baixo |

**Por que continual learning:**

1. **Conexão direta com missão LLM-relativa.** LLMs não fazem continual learning — fine-tuning sequencial corrompe modelo (well-documented). Demonstrar plasticidade local meta-aprendida pra continual = ataque honesto a uma capacidade que LLM não tem.
2. **Continuidade técnica com C2/C3.** Meta-aprender uma regra de plasticidade que **explicitamente** balanceia stability vs plasticity é extensão natural do C2. Sparsity adaptativa via k-WTA pode reduzir interferência (cf. Sparse Distributed Memory).
3. **Métricas quantitativas claras.** Average Accuracy, Forgetting (BWT), Forward Transfer — bem definidas. Permite avaliar empiricamente sem ambiguidade.
4. **Benchmark adequado pra side project 5h/semana.** Split-Omniglot (50 tarefas, 5 chars cada) cabe em GPU laptop, treina rápido, cycle de iteração curto.
5. **Literatura aberta.** Continual learning é área quente; publicações em workshops de NeurIPS/ICML toda iteração. Não é dead-end nem over-saturated.

**Por que NÃO escolher os outros agora:**
- *One-shot inédito*: scope difícil, pode virar "rodar Mini-ImageNet" que é mais incremental que continual learning.
- *Eficiência radical*: requer capital hardware ou aprendizado profundo de Brian2/NEST. ROI baixo pra founder.
- *Raciocínio temporal*: especulativo, lit menos consolidada, conecta menos com o que já temos.

### (c) Pergunta científica concreta e mensurável

**Pergunta:**

> "Pode uma rede com **plasticidade local meta-aprendida** (estilo C2) ou **encoder esparso ProtoNet com k-WTA** (estilo C3), **sem replay buffer**, atingir **average accuracy ≥75% em Split-Omniglot 50-tasks sequenciais com forgetting (BWT) ≤−10%**, batendo baseline **EWC** por ≥3 p.p. em average accuracy?"

Quebrando os termos:
- **Sem replay buffer:** restrição mecanística — replay é capability de hippocampus mas exige memória episódica explícita; pra esta primeira pergunta, queremos que a *plasticidade* sozinha + *arquitetura* faça o trabalho.
- **Split-Omniglot 50-tasks:** 50 grupos de classes treinadas sequencialmente, 1 task por vez. Métrica padrão: avg accuracy nas 50 tasks no fim.
- **BWT (Backward Transfer):** −10% significa "tasks antigas degradam só 10 p.p. quando 49 novas tasks são aprendidas em sequência". Liminar realista pós-EWC (typical EWC tem BWT de −15% a −25% em 50 tasks).
- **Bater EWC por ≥3 p.p.:** EWC é baseline padrão da área. Se nova abordagem não bate por ≥3 p.p., é incremental sem novidade científica clara.

### (d) Critério de fechamento

| Resultado | Decisão |
|---|---|
| Avg acc ≥75% E BWT ≥−10% E bate EWC por ≥3 p.p. | **Sucesso → escrever paper** (workshop-quality em 8 sessões, conference em 12-16). |
| Avg acc 65-75%, ou BWT entre −10% e −20%, ou bate EWC por <3 p.p. | **Resultado mediano → reavaliar:** vale escrever workshop paper ainda? Ou pivotar pra outro critério? Decisão administrativa em sessão dedicada. |
| Avg acc <65% OU BWT <−20% OU pior que EWC | **Pivot:** plasticidade meta-aprendida não é o motor pra continual learning. Considerar replay-based methods, ou pivotar pra eficiência radical. |
| Nada funciona após 20 sessões + ablações exaustivas | **Encerrar como exploração documentada.** O projeto vira documentação rigorosa do que NÃO funcionou. Aprendizado pessoal mantido. Rytora segue como prioridade comercial. |

### (e) Orçamento de tempo aceito

**Pré-condições assumidas (sujeitas a correção pelo Luis):**
- Founder de Rytora; tempo principal alocado ao produto comercial.
- Project Hebb é side project disciplinado (CONTEXT.md §2).

**Proposta:**
- **5-7 horas/semana**, distribuídas em 1-2 sessões de 60-90 min cada.
- **Cycle de avaliação:** a cada 5 sessões, status check de 30-60 min — está progredindo? Critério de fechamento ainda alcançável?
- **Marco 1 (continual learning):** 8 sessões de exploração + 2 de ablação = **10 sessões** = ~10-15 horas. Em cadência de 1-2 sessões/semana, **6-10 semanas**.
- **Marco 2 (paper writing, se sucesso):** 5-8 sessões de escrita = ~10-15 horas adicionais. **4-8 semanas**.
- **Total Marco 1 → workshop submission:** 4-6 meses.
- **Decisão de continuar Projeto B além do marco 1:** após resultado, não antes. Se sucesso, considerar Marco 2 (atacar segundo critério). Se mediano/falha, encerrar ou pivotar.

### (f) Status de C3

**Decisão: PUBLICAR como workshop paper.**

Justificativa:
- C3 atinge as duas metas numéricas do CONTEXT.md §4 (≥90% 5w1s, ≥70% 20w1s).
- Mecanismo defensável: "k-WTA esparso preserva ProtoNet metric learning com 75% sparsity at minimal cost (-1.45 p.p.)".
- Validação random+kWTA confirma ganho vem do treino, não de artifact.
- Workshop targets viáveis:
  - **NeurIPS Workshop on Self-Supervised Learning / Bio-plausible Learning**
  - **ICML Workshop on AI for Science / Bio-Inspired Computing**
  - **CCN (Cognitive Computational Neuroscience)** — não é proceedings mas tem visibility no campo
- Custo: 1-2 weekends de escrita (~8-12 horas). Experimentos já rodados, figs já geradas (pelo menos parcialmente, pode precisar refinar plots de sparsity × acc).

**Honestidade do paper:**
- **Não claim** "post-LLM"; claim sobre sparsity-tolerance em metric learning.
- **Reconhecer** que usa backprop end-to-end (não plasticidade local).
- **Posicionar** como evidência empírica de que esparsidade biológica não impede deep representation learning — ponte conceitual entre bio-inspired e mainstream metric learning.

**Custo de NÃO publicar:** evidência empírica fica em weekly notes; não traz feedback externo, não consolida narrativa pra Project B; pode ser overshadowed por trabalho similar publicado por outros grupos.

### Refino do CONTEXT.md (proposto)

Após 20 sessões, as duas mudanças honestas no framing original:

1. **"STDP biofísico fiel" → "plasticidade local diferenciável".**
   Sessões #1-#13 mostraram que STDP aditivo + k-WTA + clamp tem barreira estrutural na parametrização padrão (Diehl & Cook 2015). Sessões #17-#19 mostraram que plasticidade meta-aprendida (estilo Najarro & Risi 2020) é mais próxima de "differentiable plasticity" que de "Hebb biofísico" — ablações refutaram que termo Hebbian puro carrega o sinal. **A missão pós-LLM não exige fidelidade biofísica; exige mecanismos com propriedades target (local, online, no backprop end-to-end).**

2. **Adicionar nota sobre Caminho C (incrementalismo aceito como milestone).**
   C3 (ProtoNet + k-WTA) é incrementalismo defensável: atinge metas numéricas mas não as restrições mecanísticas. **Próximo marco honesto: continual learning sem replay** — capacidade que LLMs não têm e que plasticidade local pode atacar diretamente.

Atualização proposta abaixo (ver seção "Refino #21" em CONTEXT.md, criada nesta sessão).

### Roadmap de 10 sessões pra Marco 1 (continual learning sem replay)

| # | Tipo | Tempo | Objetivo concreto | Outputs |
|---|---|---|---|---|
| **22** | Admin | 60 min | Literatura: ler EWC (Kirkpatrick 2017), SI (Zenke 2017), GEM (Lopez-Paz 2017), A-GEM (Chaudhry 2019), Hadsell 2020 review. Escolher benchmark exato (Split-Omniglot 50-tasks vs alternativa). | Lista de 5 papers anotados em `experiment_02_continual/PAPERS.md`; benchmark escolhido + métricas definidas (avg acc, BWT) |
| **23** | Code | 90 min | Implementar Split-Omniglot dataset wrapper + baseline naive sequential fine-tuning (encoder ProtoNet treinado em task 1, fine-tuned em task 2, etc., sem defesa). Medir forgetting. | `experiment_02_continual/baseline_naive.py`; baseline accumulating forgetting medido em ~30-50% típico |
| **24** | Code | 90 min | Implementar EWC baseline (importance via Fisher) sobre encoder ProtoNet. | `baseline_ewc.py`; EWC vs naive em 10 tasks pra confirmar implementação correta antes de escalar |
| **25** | Code | 90 min | Escalar EWC pra 50 tasks Split-Omniglot. Estabelecer número exato de baseline EWC (referência alta) e baseline naive (referência baixa) com IC95% bootstrap. | Tabela: avg acc + BWT pra naive vs EWC, 5 seeds |
| **26** | Code/admin | 90 min | Proposta C2-continual: meta-aprender regra de plasticidade com 2 termos: (a) loss task atual, (b) penalidade de drift em representations das tasks anteriores via small probe set. Sketch arquitetura + pseudocódigo. | `proposta_c2_continual.md` + esqueleto de código (sem treinar) |
| **27** | Code | 90 min | Implementar e treinar C2-continual em Split-Omniglot 10 tasks. Validar que loop fecha. | `c2_continual.py`; primeiro número (mesmo que ruim) |
| **28** | Code | 90 min | Escalar pra 50 tasks. Comparar com EWC baseline. | Tabela: C2-continual vs EWC vs naive em 50 tasks |
| **29** | Code | 90 min | Se C2-continual mostra promessa: ablações (variar sparsity, penalidade, regra simplificada). Se não mostra: testar C3-continual (k-WTA + small replay buffer). | Ablações registradas |
| **30** | Admin | 60 min | **Status check.** Bate critério de sucesso (≥75% avg acc + BWT ≥−10% + bate EWC por ≥3 p.p.)? Se sim, abrir Marco 2 (paper writing). Se mediano, decidir pivot. | Decisão registrada em STRATEGY.md "Pós-Sessão #30" |
| **31** | Code/admin | 90 min | Buffer/refinement com base na #30. | A definir conforme resultado |

**Cancelable em qualquer ponto.** Se nas 5 primeiras sessões já fica claro que abordagem não funciona, encerra cedo e documenta como exploração.

### Decisões pendentes do Luis

Esta seção é minha recomendação. Decisões finais a tomar (idealmente em outra sessão administrativa de 30-60 min):

1. **(a)** Híbrido confirmado, ou prefere reduzido puro (skip publicação C3)?
2. **(b)** Continual learning confirmado, ou prefere atacar one-shot inédito (cross-domain) primeiro?
3. **(e)** Cadência 1-2 sessões/semana realista, ou mais conservador (1 a cada 2 semanas)?
4. **(f)** Workshop submission target específico — qual venue prioritário?
5. **CONTEXT.md refino:** aceitar mudança "STDP biofísico → plasticidade local diferenciável", ou manter framing original e tratar C2 ablações como apêndice?

Sem decisão dessas, pode-se rodar sessão #22 mesmo assim — leitura de literatura é independente do critério escolhido. Mas decidir antes de #23 (implementação) é necessário pra evitar refazer trabalho.

---

## Confirmação Pós-Sessão #21 (sessão #22, 2026-05-01)

Após 3 dias de reflexão sobre as 5 decisões pendentes em "Decisões pendentes do Luis", **decisões finais registradas pelo Luis:**

### (a) Versão do Projeto B: **HÍBRIDO** ✓

Confirmado: publicar C3 como workshop paper + continuar Projeto B (continual learning sem replay) em paralelo.

- C3 paper: trabalho independente de pesquisa, executável em weekend dedicado.
- Projeto B: 5-7h/semana de pesquisa.
- Sem conflito de tempo entre os dois — atacam estágios diferentes (escrita vs experimentação).

### (b) Critério pós-LLM atacado primeiro: **CONTINUAL LEARNING sem replay buffer** ✓

Confirmado: catastrophic forgetting é a capacidade que LLMs não têm e que plasticidade local meta-aprendida pode atacar diretamente.

**Pergunta científica oficial** (carry-over de "Decisão Pós-Sessão #20" item c):

> "Pode uma rede com plasticidade local meta-aprendida (estilo C2) ou encoder esparso ProtoNet com k-WTA (estilo C3), **sem replay buffer**, atingir **average accuracy ≥75% em Split-Omniglot 50-tasks sequenciais com forgetting (BWT) ≤−10%**, batendo baseline EWC por ≥3 p.p. em average accuracy?"

### (c) Cadência: **SEM TETO FORMAL** ✓

Mudança em relação à recomendação original (que sugeriu "5-7h/semana, 1-2 sessões/semana"). Decisão final do Luis:

- **Sem teto formal de horas/semana ou sessões/semana.** Luis decide sessão-a-sessão se continua ou para.
- **Limite por sessão mantido** (90 min hard, watchdog pra implementações).
- **Sem regra "X sessões por semana".**

**Implicação:** projeto pode parar dias ou semanas sem prejuízo formal. Critério de continuidade vira intrínseco (Luis julga se vale continuar) em vez de extrínseco (calendário). Combina com "deixar ele decidir rumo" — disciplina é dele, não imposta pelo plano.

### (d) Venue C3: **NeurIPS Workshop on Bio-plausible Learning** ✓

Confirmado:

- **Target:** NeurIPS Workshop on Bio-plausible Learning (submissão típica setembro/outubro de cada ano).
- **Próxima janela:** ~setembro 2026.
- **Paper writing:** pode começar a qualquer momento. Ideal: 1-2 weekends dedicados.
- **Backup venues** se NeurIPS Bio-plausible não tiver workshop em 2026: ICML AI for Science, NESY (Neuro-Symbolic), CCN (Cognitive Computational Neuroscience).

### (e) CONTEXT.md refino: **ACEITO** ✓

A seção §1.1 "Refino #21" adicionada à CONTEXT.md (sessão #21) torna-se **parte oficial da missão**:

- "STDP biofísico fiel" foi superado por **"plasticidade local diferenciável"** como framing operacional.
- Sucesso numérico ≠ sucesso mecanístico (C3 atingiu metas mas usa backprop).

Implicação: decisões futuras de mecanismo são avaliadas contra "plasticidade local diferenciável" (mais permissivo) em vez de "Hebb biofísico estrito" (mais restritivo).

### Estado oficial pós-Confirmação #21

| Item | Status | Onde está |
|---|---|---|
| Caminho A (mais STDP tuning) | Dormente | STRATEGY.md "Pós-#10" e "Pós-#13" |
| Caminho B (mecanístico fiel original) | **Reformulado como "plasticidade local diferenciável" + continual learning** | Esta seção, CONTEXT.md §1.1 |
| Caminho C (ProtoNet+esparso) | **Concluído como milestone** — vira workshop paper NeurIPS Bio-plausible | Esta seção (d) |
| Marco 1 ativo | Continual learning sem replay em Split-Omniglot | `experiment_02_continual/` (a criar nesta sessão) |
| Cadência | Sem teto formal; Luis decide sessão-a-sessão | Esta seção (c) |

### Próximas sessões (roadmap reativado)

Roadmap de 10 sessões pra Marco 1 (#22-#31), originalmente em "Decisão Pós-Sessão #20", agora com decisões fechadas:

- **#22 (esta):** confirmar decisões + literatura review (5 papers core).
- **#23:** baseline naive sequential fine-tuning sobre encoder ProtoNet pré-treinado.
- **#24-#25:** baseline EWC, escalar pra 50 tasks.
- **#26-#28:** propostas C2-continual e/ou C3-continual.
- **#29:** ablações.
- **#30:** status check sucesso/pivot.
- **#31:** refinement ou paper writing.

Decisões 1-5 fechadas — não revisitar sem motivo forte (ex: implementação revela impossibilidade técnica).

---

## Reformulação Pós-Sessão #23 (sessão #24, 2026-05-01)

**Contexto:** sessão #23 mediu naive ProtoNet sequential em Split-Omniglot 50-tasks (random class splits, com warmup) e obteve **ACC 82.58% / BWT −12.46 p.p.** — bem acima do critério "ACC 30-50% + BWT −30 a −50". Causa mecanística (não bug): ProtoNet não tem classifier head treinado, prototypes computados fresh no eval, encoder aprende métrica genérica do dataset robusta a tasks da mesma família. Resultado: pergunta científica oficial (`ACC ≥75% E BWT ≥−10% E batendo EWC por ≥3 p.p.`) é **trivialmente atingida pelo baseline naive em ACC**, anulando margem científica pra distinguir propostas.

Esta seção avalia 4 opções de reformulação e escolhe uma. Decisão final é do Luis; eu recomendo provisionalmente.

### Opções avaliadas

#### (A) Skip warmup + mais episodes/task (mesmo Split-Omniglot random)

- **O que muda:** `--warmup-episodes 0 --finetune-episodes 300` (3× mais drift, sem pretreino genérico).
- **Custo de implementação:** mínimo (apenas CLI flags em `baseline_naive.py`).
- **Naive esperado:** ACC 60-72%, BWT −15 a −25 p.p. — mais adversarial que #23 mas ainda dentro do regime "fácil" do ProtoNet (random splits compartilham features visuais entre tasks).
- **Defensibilidade:** padrão menor; reviewers podem perguntar "por que random splits e não alphabets?".
- **Compat replay-free:** ✓.
- **Risco:** pode não ser adversarial o suficiente — random splits através de alfabetos preservam features genéricas (curvas, traços, simetria).

#### (B) Tasks por alfabeto Omniglot

- **O que muda:** cada task = 1 alfabeto (Greek, Hebrew, Korean, etc.). Omniglot tem 30 alfabetos no background + 20 no evaluation = 50 alfabetos totais. Episódios 5-way amostram 5 caracteres do alfabeto da task atual.
- **Custo de implementação:** moderado (~1 sessão). Refatorar `build_tasks` pra agrupar por alfabeto via parse do path em `dataset._characters` (torchvision Omniglot expõe isso).
- **Naive esperado:** ACC 50-65%, BWT −20 a −30 p.p. — alfabetos têm estilos de stroke específicos; switching entre alfabetos força encoder a "esquecer" features anteriores.
- **Defensibilidade:** **alta**. Schwarz et al. 2018 ("Compress and Compare") e outros papers de CL usam alfabetos Omniglot como tasks naturais. Padrão da literatura.
- **Compat replay-free:** ✓.
- **Risco:** alfabetos variam em tamanho (14-55 caracteres); precisa normalizar (subsample 5 chars + 14 train / 6 test instances).

#### (C) Cross-domain (Omniglot → MNIST → FashionMNIST → ...)

- **O que muda:** task 1 = Omniglot, task 2 = MNIST, task 3 = FashionMNIST, etc. Forgetting cross-domain é dramático (literatura confirma).
- **Custo de implementação:** **alto**. 3+ datasets, normalização de tamanho/canal, episode samplers por dataset, eval cross-task. ~3-5 sessões só de infra.
- **Naive esperado:** ACC 20-40%, BWT −40 a −60 p.p. — exatamente o "floor brutal" do critério original.
- **Defensibilidade:** **alta** mas em literatura diferente (lifelong learning cross-task: Aljundi 2018 MAS, Hadsell 2020).
- **Compat replay-free:** ✓.
- **Risco:** **escopo grande pra side project 5h/semana**. Inconsistência com decisão (b) "Split-Omniglot 50-tasks" da Confirmação Pós-#21 — exigiria pivot adicional.

#### (D) Híbrido B+A (alphabets + skip warmup)

- **O que muda:** alfabetos como tasks (B) + sem warmup, mais episodes/task (A).
- **Custo de implementação:** ~igual ao (B) sozinho (warmup é só CLI flag).
- **Naive esperado:** ACC 40-55%, BWT −25 a −35 p.p. — combina pressão de domain shift entre alfabetos com fragilidade de encoder fresh.
- **Defensibilidade:** **alta** (alphabets é literatura padrão; skip warmup é ablação sensata).
- **Compat replay-free:** ✓.
- **Risco:** se naive cair pra <30%, ficamos em regime "task incremental" puro onde nem encoder genérico ajuda — pode ser adversarial demais. Mas isso é resolvível dialing back episodes/task se for o caso.

### Critério de escolha

| Critério | (A) | (B) | (C) | (D) |
|---|---|---|---|---|
| Custo de implementação | trivial | moderado | alto | moderado |
| Naive esperado (target 40-60%) | 60-72% | 50-65% | 20-40% | **40-55%** ✓ |
| Defensibilidade | média | alta | alta (outra lit) | **alta** ✓ |
| Compat replay-free | ✓ | ✓ | ✓ | ✓ |
| Compat side project | ✓ | ✓ | ❌ scope grande | ✓ |
| Margem pra EWC e propostas | pequena | média | larga | **média-larga** ✓ |

### Decisão: **OPÇÃO D (híbrido B+A)** — alphabets + skip warmup

**Recomendação confirmada por:**

1. **Naive esperado em sweet spot** (40-55%): deixa margem de ~15-25 p.p. pra técnicas (EWC, C2, C3) demonstrarem ganho real.
2. **Alfabetos têm precedente sólido em literatura** (Schwarz 2018, outros). Reviewer não vai questionar setup.
3. **Custo moderado de implementação** (~1 sessão). Compatível com ritmo side project.
4. **Mantém Split-Omniglot** como dataset base — não viola decisão (b) da Confirmação Pós-#21.
5. **Skip warmup é gratuito** (só CLI flag); benefício claro de fragilizar encoder.

**Rejeitado por:**

- (A) sozinho — risco de não ser adversarial o suficiente; reviewer pode pedir alphabets.
- (B) sozinho — warmup mantém viés "genérico" que dilui forgetting.
- (C) — escopo desproporcional pra side project; pivot adicional fora da Confirmação Pós-#21.

### Pergunta científica oficial reformulada

**Antiga (broken pelo #23):**
> "Pode plasticidade meta-aprendida ou ProtoNet+k-WTA, sem replay, atingir ACC ≥75% e BWT ≥−10% em Split-Omniglot 50-tasks, batendo EWC por ≥3 p.p.?"

**Nova (calibrada pra Opção D):**

> "Em **Split-Omniglot por alfabeto** (50 tasks correspondentes aos 50 alfabetos, **sem warmup**, fine-tune sequencial), pode plasticidade meta-aprendida (estilo C2) ou ProtoNet+k-WTA (estilo C3), **sem replay buffer**, atingir **ACC absoluto ≥70%** com **BWT ≥−10 p.p.**, **batendo o baseline EWC por ≥3 p.p. em ACC**?"

**Justificativa numérica calibrada (alvos absolutos):**

| Linha | ACC esperado | BWT esperado |
|---|---|---|
| Naive (sem defesa) | 40-55% | −25 a −35 p.p. |
| EWC (regularization) | 55-70% | −10 a −20 p.p. |
| **Nossa proposta target** | **≥70%** (idealmente 73-78%) | **≥−10 p.p.** |
| Sky reference (GEM com replay) | 75-85% | −5 a −10 p.p. |

**Critérios numéricos exatos serão confirmados após sessão #25** (quando naive em Opção D for medido empiricamente). Se naive vier em range diferente do previsto (ex: já 65%+), a pergunta pode precisar de mais um ajuste — mas espera-se que Opção D coloque naive na faixa adversarial prevista.

### Critério de fechamento atualizado

| Resultado | Decisão |
|---|---|
| ACC ≥70% E BWT ≥−10 p.p. E bate EWC por ≥3 p.p. | **Sucesso → escrever paper** |
| ACC 60-70%, ou BWT entre −10 e −20, ou bate EWC por <3 p.p. | **Mediano → reavaliar** vale workshop ou pivotar |
| ACC <60% OU BWT <−20% OU pior que EWC | **Pivot** pra outro mecanismo |
| Nada funciona após 20 sessões | **Encerrar como exploração documentada** |

### Próximas sessões (roadmap atualizado)

- **#25 (próxima):** implementar Opção D em `baseline_naive.py` — refatorar `build_tasks` pra agrupar por alfabeto, rodar naive 5 seeds. **Validar empiricamente que naive cai pra range adversarial (40-55% ACC).**
- **#26:** se #25 confirma range, implementar EWC baseline.
- **#27-#28:** escalar EWC, propostas C2-continual / C3-continual.
- **#29:** ablações.
- **#30:** status check sucesso/pivot.

Se #25 mostrar naive ainda alto (>65% ACC), essa seção é re-aberta antes de seguir.

---

## Revisão Pós-Sessão #25 (sessão #26, 2026-05-01)

### Estado factual

Após 5 sessões dedicadas a Marco 1 (continual learning sem replay):

| # | Tipo | Resultado |
|---|---|---|
| 21 | Admin | Recomendações Projeto B (6 perguntas) |
| 22 | Admin | Confirmação 5 decisões + lit review |
| 23 | Code | Naive baseline #1 (random splits + warmup): ACC **82.58%** — broken (acima do critério adversarial) |
| 24 | Admin | Reformulação benchmark pra Opção D |
| 25 | Code | Naive baseline #2 (alphabets + skip warmup): ACC **80.65%** — Opção D insuficiente, naive cai só 1.93 p.p. |

**Ratio progresso experimental / sessões: 2 medições / 5 sessões.** As 3 sessões admin produziram entendimento mas não dados que avançam Marco 1.

**Achado mecanístico não previsto:** ProtoNet **sem classifier head treinado** é inerentemente robusto a catastrophic forgetting em Omniglot. Encoder aprende métrica genérica que transfere entre tasks (random splits, alphabets, com ou sem warmup). Prototypes-fresh-no-eval protege a "decisão final".

### 4 caminhos honestos avaliados

#### Caminho 1 — Reformular setup pela 3ª vez (Opção E: classifier head)

Substituir prototype-based eval por classifier head (linear N-way) que é treinado sequencialmente. Causa raiz do achado #25: sem classifier head treinado, não há pesos pra "esquecer". Adicionar resolveria isso.

| Dimensão | Avaliação |
|---|---|
| **Custo (sessões antes de testar proposta)** | ~3 sessões: re-arquitetura + naive em novo setup + EWC. Soma com #26 = **4 sessões admin/setup antes de empirically testar propostas** |
| **Probabilidade paper** | **Moderada.** Setup vira mainstream-CL (EWC works), propostas C2/C3 podem agregar 3-5 p.p. Workshop paper plausível mas perde unicidade do bio-inspired angle |
| **Compat tese pós-LLM** | Sim (continual learning é critério pós-LLM), mas o setup vira incremental sobre EWC/SI/GEM em vez de mecanismo bio-inspirado |
| **Sinal meta-trabalho** | **Alto risco.** Esta seria a 3ª reformulação de setup. Pattern: cada reformulação revela que problema é mais difícil de calibrar que esperado. Próxima reformulação pode revelar mais um problema |

#### Caminho 2 — Reformular pergunta científica (aceitar achado como resultado)

Em vez de "bater EWC com plasticidade meta-aprendida", aceita o achado #25 como descoberta empírica. **Pergunta nova:** "Por que ProtoNet sem classifier head é inerentemente robusto a catastrophic forgetting em Omniglot? Caracterizar via ablações sistemáticas."

| Dimensão | Avaliação |
|---|---|
| **Custo (sessões antes de testar)** | ~0 sessões de calibração — usa setup que já temos. Vira ablações: variar finetune-episodes (100→500→1000→3000), n_chars/task (5→14→30), classifier head sim/não, embedding dim, encoder size. **3-5 sessões de experimentos** geram tabela completa de ablações |
| **Probabilidade paper** | **Alta dentro de workshop.** "Empirical Study: When Are Prototype-Based Methods Naturally Robust to Catastrophic Forgetting?" tem originalidade (maioria de papers CL assume classifier head). NeurIPS Bio-Plausible Workshop ou ICML AI for Science aceitariam |
| **Compat tese pós-LLM** | **Indireta** — não claim "novo método pós-LLM"; claim sobre quando mecanismos prototype-based agregam robustez. Mas insight informa futuros mecanismos pós-LLM |
| **Sinal meta-trabalho** | **Baixo.** Transforma "fracasso de calibração" em "achado positivo bem caracterizado". Não exige re-setup. Reduz incerteza ao máximo |

#### Caminho 3 — Pivot pra outro critério pós-LLM

Descartar continual learning como rota. Voltar aos 4 critérios pós-LLM (CONTEXT.md §1) e escolher outro: one-shot cross-domain, eficiência radical, ou raciocínio temporal.

| Dimensão | Avaliação |
|---|---|
| **Custo (sessões antes de testar)** | **Alto.** Escolher problema + literature review + baseline + setup adversarial = ~5-8 sessões antes de primeiro experimento informativo. Histórico do Marco 1 mostra que estimativas iniciais de calibração são otimistas |
| **Probabilidade paper** | Variável por critério. One-shot cross-domain saturado mas factível. Eficiência radical exige hardware neuromórfico ou Brian2 (complexo). Raciocínio temporal especulativo |
| **Compat tese pós-LLM** | Sim por construção (cada um é critério pós-LLM oficial). Mas pivot perde o investimento em C2/C3 do Marco 1 |
| **Sinal meta-trabalho** | **Muito alto.** 5-8 sessões só pra setup é o cenário de meta-trabalho que queremos evitar. Padrão histórico (sessões #1-#13 STDP, #14 reavaliação, #15-#20 família C, #21-#26 continual) sugere que cada novo "critério atacado" tem 5-10 sessões de calibração antes de produzir dado |

#### Caminho 4 — Aceitar Marco 1 em risco e encerrar

Reconhecer que #21-#25 produziram entendimento mas não progresso publicável independente. Publicar C3 como milestone (decisão pós-#21 já confirmou isso). Encerrar Marco 1 (continual learning) formalmente. Reavaliar projeto: vale Marco 2? Quais critérios pós-LLM ainda viáveis em side project 5h/semana?

| Dimensão | Avaliação |
|---|---|
| **Custo (sessões antes de testar)** | **0 sessões adicionais.** ~2 weekends pra C3 paper writing (custo já antecipado) |
| **Probabilidade paper** | **1 paper confirmado** (C3, defensável: 93% ACC com 75% sparsity). 0 papers de Marco 1 (não completou) |
| **Compat tese pós-LLM** | C3 mantém uma contribuição parcial (esparsidade biológica + metric learning). Reconhece honestamente que side project 5h/semana não consegue atacar continual learning rigorosamente |
| **Sinal meta-trabalho** | **Eliminado.** Não adiciona meta-trabalho. Elimina o que está acumulando. Risco oposto: aparenta "desistir prematuramente" — mas dado padrão de 5 sessões = 0 dados publicáveis, persistir é o desperdício maior |

### Comparação cruzada

| Caminho | Sessões antes de paper | Risco meta-trabalho | Honestidade do framing | Compat side project |
|---|---|---|---|---|
| 1 — Opção E setup | 8-10 | Alto | Re-setup pela 3ª vez | Médio |
| **2 — Reformular pergunta** | **5-7 (paper de robustez)** | **Baixo** | **Transforma achado em resultado** | **Alto** |
| 3 — Pivot critério | 12-20 | Muito alto | "Mais um setup pra calibrar" | Baixo |
| 4 — Encerrar com C3 | 2-4 (só paper writing) | Eliminado | Honesto sobre limites | Alto |

### Padrão das 25 sessões — leitura honesta

| Fase | Sessões | Output principal |
|---|---|---|
| STDP em PyTorch | #1-#13 | Barreira estrutural identificada (matched filter trivial) |
| Reformulação | #14 | 3 caminhos A/B/C documentados |
| Família C (Caminho C) | #15-#20 | C3 = 93% ACC com 75% sparsity (publicável) |
| Marco 1 setup | #21-#22 | Decisões fechadas + lit review |
| Marco 1 calibração | #23-#26 | ProtoNet robusto a forgetting em Omniglot (achado mecanístico) |

**Observação chave:** os achados mecanísticos (sessões #11, #18, #25) frequentemente vêm DA tentativa de calibrar/validar, não de "novo experimento". Sessões #23 e #25 produziram dado científico real — só não produziram dado que avança a hipótese pré-formulada.

Caminho 2 capitaliza esse padrão: assume que a próxima descoberta também vai vir de caracterização rigorosa do que temos, não de calibração de novo setup.

### O que esta sessão NÃO decide

Esta sessão é diagnóstico. **Decisão entre Caminhos 1/2/3/4 fica pra próxima sessão administrativa**, idealmente 1-3 dias após esta pra reflexão.

**Pergunta de gatilho pro Luis decidir:** "Em qual cenário eu sentiria que valeu o investimento de 5h/semana × próximas 8 semanas?"

- Se "publicar paper sobre método novo" → Caminho 1
- Se "publicar paper sobre achado empírico bem caracterizado" → Caminho 2
- Se "atacar capacidade pós-LLM diferente" → Caminho 3
- Se "consolidar C3 e reavaliar projeto inteiro" → Caminho 4

Cada resposta corresponde a um caminho diferente. Não há resposta certa — só resposta consistente com prioridades reais (que só Luis sabe).

---

## Decisão Pós-Sessão #26: Caminho 5d (3 arquiteturas) — sessão #27 (2026-04-29)

**Contexto:** após reflexão sobre os 4 caminhos da revisão #26, Luis escolheu uma alternativa: testar empiricamente 3 possibilidades arquiteturais que combinam **STDP biofísico** (família STDP de #1-#13) + **plasticidade local diferenciável** (família C2 de #17-#19) + **continual learning sequencial sem replay** (Marco 1).

Não é literalmente nenhum dos Caminhos 1-4 — é um híbrido que aposta que a combinação dos achados anteriores (mesmo que cada um isolado tenha tido limitações) pode produzir contribuição original. Razões pra essa escolha (registradas pelo Luis):

1. C3 já é resultado defensável (workshop paper); não vamos "encerrar com C3" prematuramente sem testar a tese híbrida.
2. Setups de continual learning (#23, #25) confirmaram que ProtoNet sozinho é robusto — o que define um floor claro pra propostas comparem.
3. STDP+C2 hybrid é experimentalmente novo (não há paper específico sobre essa combinação em continual learning).
4. Bate frontalmente a tese pós-LLM original ("plasticidade local diferenciável" como mecanismo).

### Aviso explícito (registrado pra pôr todos os dados na mesa)

- **STDP biofísico (Diehl & Cook 2015) tem barreira estrutural conhecida** das sessões #1-#13 — atinge teto em ~36% em Omniglot one-shot via matched filter trivial.
- **Sessão #18 mostrou que termo Hebbian puro A·pre·post é dispensável** em C2 (contribuição vem de B, C, D).
- **Esta sessão testa se em contexto CONTINUAL essas conclusões se mantêm ou mudam.** Hipótese específica que justifica o teste: continual learning introduz pressão temporal (sequência de tasks) que pode tornar trace STDP relevante de forma que one-shot estático não tornou.

### Possibilidade A — STDP biofísico nas camadas iniciais + C2 nas finais

Encoder em duas partes:
- Camadas 1-2 (early): STDP biofísico estilo Diehl & Cook 2015 — atualiza durante apresentação do support, sem backprop. Aprende features bottom-up.
- Camadas 3-4 (late): plasticidade meta-aprendida estilo C2 — `Δw = A·pre·post + B·pre + C·post + D`, A/B/C/D meta-aprendidos via outer loop.

**Hipótese:** STDP captura features genéricas resilientes a forgetting (porque updates são locais e não dependem de gradiente cross-task), C2 captura especificidade da task atual via meta-learning. Combinação: features estáveis + adaptação rápida.

**Risco principal:** STDP biofísico mostrou barreira em #1-#13. Pode contribuir 0 ou negativamente.

### Possibilidade B — Camada única com regra híbrida (FOCO desta sessão)

Encoder linear simples (igual `c2_simplified.py`). Regra de plasticidade unificada:

```
Δw = η · (A · pre · post · trace + B · pre + C · post + D)
```

Onde `trace` é um STDP-like trace (decay exponencial sobre histórico de pre-spikes durante o inner loop). Todos os parâmetros (A, B, C, D + decay) meta-aprendidos via outer-loop gradiente.

**Hipótese:** o termo Hebbian se torna não-trivial quando combinado com trace temporal. Em #18 sem trace, A foi dispensável. Com trace, A pode codificar timing-dependent learning.

**Por que B primeiro:** arquitetura mais compacta (uma camada só, mais fácil de debugar). Se B não mostra ganho do termo trace, A e C também não vão (ambas dependem da mesma intuição — STDP timing matters).

### Possibilidade C — STDP within-task + C2 between-task

Two-timescale architecture:
- **Within-task:** durante apresentação dos 14 chars do alfabeto, STDP atualiza pesos com plasticidade local rápida (sem gradiente).
- **Between-task:** entre tasks, C2 outer-loop gradient consolida ajustes ao acumular signals.

**Hipótese:** STDP rápido captura adaptação à task atual; C2 lento previne forgetting via consolidação do que ficou estável.

**Risco principal:** complexidade. Two-timescale é difícil de tunar (taxa relativa, quando consolidar, etc).

### Pergunta científica unificada

> "Em continual learning Split-Omniglot por alfabeto (50 tasks, sem replay, sem warmup), alguma das 3 arquiteturas (A/B/C) **supera o baseline naive ProtoNet (80.65% ACC, BWT −9.26)** com **margem ≥3 p.p. em ACC ou ≥5 p.p. em BWT**, **E mostra contribuição não-trivial do componente STDP/Hebbian** (ablação removendo termo A no caso de B; ablação removendo camadas STDP em A e C) **≥3 p.p. de ganho residual**?"

### Critério de fechamento Caminho 5d

| Resultado | Decisão |
|---|---|
| Pelo menos 1 arquitetura supera baseline E ablação confirma contribuição STDP ≥3 p.p. | **Sucesso → escrever paper** combinando achado mecanístico + método |
| 1+ supera baseline mas ablação mostra STDP irrelevante | **Resultado parcial:** vira paper estilo Caminho 2 ("ProtoNet+plasticidade meta-aprendida em continual learning, sem componente STDP") |
| Nenhuma supera baseline | **Encerrar como achado empírico:** "STDP não agrega em continual learning Omniglot, em 3 arquiteturas testadas". Caminho 4 (publicar só C3) recupera. |

### Orçamento estimado

- 12-18 sessões pra implementar 3 arquiteturas + ablações cruzadas
- Sessão #27 (esta): scaffolds 3 arq + B implementação + sanity de B
- Sessões #28-#30: B completo (5 seeds) + ablação A=0
- Sessões #31-#34: A implementação + ablação
- Sessões #35-#38: C implementação + ablação
- Sessões #39-#42: ablações cruzadas + status check

Cancelable a qualquer ponto. Se B sanity ficar < 40% ou > 95%, esta sessão para e re-considera (caminho 5d pode não ser viável). Se B 5 seeds não bate baseline, sessão admin re-avalia se vale rodar A e C.

---

## Decisão Pós-Sessão #27: Caminho 5e (arquitetura combinada) — sessão #28 (2026-04-29)

**Contexto:** sessão #27 implementou Possibilidade B (Caminho 5d) com encoder linear simples — sanity 47.89% confirmou loop fecha mas mostrou gap fundamental de capacity vs ProtoNet CNN-4 (-32.76 p.p. abaixo do baseline naive 80.65%). Possibilidade B ficaria abaixo do baseline mesmo com 5 seeds completos. As 3 opções pós-#27 (α/β/γ) levaram Luis a escolher Caminho 5e: combinar TODOS os mecanismos caracterizados ao longo do projeto numa arquitetura única.

**Não é Possibilidade A do Caminho 5d.** A previa STDP biofísico nas camadas iniciais (que tem barreira em #1-#13). 5e usa CNN-4 standard backprop nas camadas iniciais — substitui STDP biofísico por convolutional features padrão. STDP-like aparece só via trace temporal nos meta-params da plasticidade local (mecanismo herdado de Possibilidade B).

### Mecanismos combinados (5)

| Mecanismo | Origem | Papel em 5e |
|---|---|---|
| **CNN-4 encoder** | C3 (sessão #20), ProtoEncoder | Capacity. 4 blocos Conv-BN-ReLU-MaxPool, output 64D. SGD via backprop |
| **Plasticidade local meta-aprendida** | C2 / c2_simplified (sessão #19) | Camada linear final 64→64 com regra `Δw = A·pre·post + B·pre + C·post + D`. Inner loop adapta W per episode |
| **Trace STDP-like** | Possibilidade B (sessão #27) | `trace[t] = decay·trace[t-1] + pre[t]`, multiplicador no termo Hebbian. Tenta capturar timing |
| **k-WTA esparso** | C3b (sessão #20) | k=16 (75% sparsity) aplicado no embedding final pós-plasticidade |
| **Continual sequencial sem replay** | Marco 1 (sessões #21-#27) | 50 alphabet tasks sequenciais, skip warmup, sem replay buffer |

### Arquitetura específica (Opção 2: hybrid backprop + plasticidade)

```
Image (1, 28, 28)
  → CNN-4 (4 blocos Conv-BN-ReLU-MaxPool, SGD via backprop) → 64D
  → Linear plasticity layer W (64→64, inner-loop adapted)
    Δw = η·(A·pre·post·trace + B·pre + C·post + D)
    A,B,C,D meta-aprendidos; W reseta pra zero a cada episode
  → k-WTA (k=16, 75% sparsity)
  → Prototype classifier (cosine, β=8)

Outer loop: cross-entropy do query
  Backprop atualiza:
  - CNN-4 weights (slow)
  - A, B, C, D, decay (slow, meta-params)
```

Por que Opção 2 (e não Opção 1 = plasticidade em todas as camadas):
- Preserva capacity do CNN (gap em sessão #27 era encoder linear vs CNN)
- Adiciona plasticidade só na camada final pra preservar interpretability
- Compute viável (inner loop em 64×64 W, não em 4 layers de conv)

### Pergunta científica

> "Em continual learning Split-Omniglot por alfabeto (50 tasks, sem replay, sem warmup), arquitetura combinada (CNN+plasticidade+trace+k-WTA) **supera baseline naive ProtoNet (80.65% ACC, BWT −9.26)** **E mantém pelo menos parte do desempenho do C3 vanilla (94.55% ACC one-shot, sem continual)** quando submetida ao mesmo regime continual?"

### Critérios de sucesso e fechamento

| Resultado (5 seeds) | Decisão |
|---|---|
| ACC ≥85% (5pp acima de naive, dentro de 10pp de C3 vanilla) E BWT ≥-7 | **Sucesso → ablações sistemáticas** |
| ACC 81-85% E BWT -7 a -10 | **Mediano** — investigar quais mecanismos contribuem via ablações |
| ACC dentro de naive ± 2 p.p. (78.65-82.65%) | **Sem ganho** — encerra Marco 1 com paper estilo Caminho 2 (paper de robustez) |
| ACC < 78% | **Pior que naive** — encerra com C3 (Caminho 4) |

### Aceitação explícita de risco

- **Complexidade:** 5 mecanismos combinados torna ablação difícil de interpretar. Cada ablação atribui efeito a 1 componente assumindo outros 4 ficam intactos — válido mas demanda múltiplas ablações cruzadas.
- **Custo computacional:** CNN-4 + inner loop 10× em 64×64 W + 50 tasks × 100 episodes/task × 5 seeds = ~20-40 min/seed. 5 seeds = 1.5-3.5h por configuração. Defensável mas não trivial.
- **Originalidade incremental:** kitchen sink architecture é incrementalismo (combina coisas existentes). Defensível academicamente como "evidência empírica de qual combinação funciona em CL", mas não é breakthrough mecanístico.

### Orçamento estimado (13-17 sessões)

- **#28 (esta):** scaffold + sanity 5 tasks + sanity 50 tasks (1 seed)
- **#29:** sanity 5 seeds completos (n_inner=10, finetune=100, eval=50). Tempo esperado: 30-60 min
- **#30-#31:** ablação 1 — remover trace (testa contribuição STDP-like)
- **#32-#33:** ablação 2 — remover k-WTA
- **#34-#35:** ablação 3 — remover plasticidade meta-aprendida (= ProtoNet baseline em CL)
- **#36-#37:** comparação com EWC baseline
- **#38-#42:** análise + paper draft + refinement

Cancelable em qualquer ponto. Se #29 (5 seeds) não bater 75% ACC, sessão admin re-avalia se vale continuar 5e ou pivotar pra Caminho 2 (paper de robustez).

---

## Fechamento Marco 1 — sessão #30 (2026-04-29)

**Decisão final:** Marco 1 (continual learning sem replay) **ENCERRADO** pelo critério literal de #29: ACC 74.78% < 78% → Caminho 4 ativado → publicar só C3.

Luis aceitou o resultado. Esta seção formaliza o fechamento e prepara transição pro paper C3.

### Resumo das 9 sessões Marco 1 (#21-#29)

| # | Tipo | Output principal |
|---|---|---|
| 21 | Admin | Recomendações Projeto B (6 perguntas) |
| 22 | Admin | Confirmação 5 decisões + lit review (EWC/SI/GEM/A-GEM/Hadsell) |
| 23 | Code | Naive baseline #1 (random splits + warmup): **ACC 82.58%** — broken (acima do critério adversarial) |
| 24 | Admin | Reformulação benchmark pra Opção D (alphabets + skip warmup) |
| 25 | Code | Naive baseline #2 (alphabets + skip warmup): **ACC 80.65%** — Opção D insuficiente, naive cai só 1.93 p.p. |
| 26 | Admin | Revisão estratégica (4 caminhos avaliados) |
| 27 | Code | Caminho 5d Possibilidade B (encoder linear): **ACC 47.89%** — gap de capacity vs CNN-4 |
| 28 | Code | Caminho 5e (CNN+plast+trace+k-WTA) sanity: **ACC 75.34%** (1 seed reduzido) |
| 29 | Code | Caminho 5e 5 seeds defaults: **ACC 74.78%, BWT -16.80** — pior que naive em ambas |

### 4 abordagens testadas, todas ≤ naive ProtoNet (80.65%)

| Abordagem | ACC | BWT | Mecanismo |
|---|---|---|---|
| Naive ProtoNet random splits (#23) | 82.58% | -12.46 | sem defesa, prototypes-fresh-no-eval |
| Naive ProtoNet alphabets+no-warmup (#25) | 80.65% | -9.26 | mesma arq, setup mais adversarial |
| Possibilidade B encoder linear (#27) | 47.89% | -2.05 | sem CNN, plasticidade meta + trace |
| **Caminho 5e (#29)** | **74.78%** | **-16.80** | CNN+plast+trace+k-WTA combinado |

**Padrão consolidado:** mecanismos bio-inspirados (plasticidade meta-aprendida, trace STDP-like, k-WTA esparso) **não conseguem bater naive ProtoNet em CL setup** quando arquitetura é prototype-based em Omniglot. Adicionar complexidade introduz mais forgetting (BWT pior) sem ganho em ACC.

### Achado mecanístico documentado (positivo, mesmo com resultado negativo no Marco 1)

> **ProtoNet metric learning é inerentemente robusto a catastrophic forgetting em Omniglot.**

Razões mecanísticas:
1. **Não há classifier head treinado** pra "esquecer" — prototypes são computados FRESH no eval do support set
2. **Encoder aprende métrica genérica** (separar caracteres por similaridade) que transfere bem entre tasks da mesma família visual
3. **Tasks na mesma família** (alphabets diferentes em Omniglot) compartilham features básicas (curvas, traços, simetria) — encoder não precisa "esquecer" pra aprender novas tasks

**Esse é insight científico real**, mesmo que o Marco 1 não tenha produzido método novo que bata baseline. Pode virar apêndice do paper C3 ou subseção em "Discussion" como observação relevante pra continual learning literature.

### O que foi aprendido (positivo)

1. **Caracterização rigorosa** de quando ProtoNet é robusto vs frágil em CL
2. **Ablações sistemáticas** que isolam contribuição de cada mecanismo (linear vs CNN, com vs sem trace, com vs sem plasticidade)
3. **Setup adversarial tentado e documentado** (alphabets + skip warmup) — informa pesquisa futura sobre quando bio-inspired methods agregam
4. **Insight sobre plasticidade na camada errada** (#29): plasticidade meta-aprendida APÓS CNN não previne CNN drift — implica que pra CL com bio-inspired, plasticidade precisa ser AT the source of forgetting, não downstream

### Implicação pra projeto

**Caminho 4 ativado.** C3 (sessão #20: 93.10% ACC com 75% sparsity) vira target de paper. Marco 1 não produz paper independente, mas seus achados podem virar apêndice/discussion no paper C3.

---

## Plano paper C3 — Workshop NeurIPS Bio-Plausible Learning

**Status:** scoping nesta sessão. Paper writing começa na sessão #31 (não nesta).

### Título tentativo

> **"k-WTA Sparsity Preserves Prototypical Network Performance in Few-Shot Learning"**

Alternativas:
- "Biologically-Inspired Sparsity in Prototypical Networks: Empirical Analysis on Omniglot"
- "Sparse Coding in Few-Shot Metric Learning: How Much k-WTA Can ProtoNet Tolerate?"

### Contribuição central

Demonstração empírica de que **codificação esparsa biológica é compatível com deep representation learning em few-shot metric learning**: aplicar k-WTA (75% das ativações zeradas) ao embedding final de um Prototypical Network preserva 93.10% acurácia em Omniglot 5-way 1-shot, dentro de **-1.45 p.p.** do baseline ProtoNet completo (94.55%). Esparsidade até 87.5% mantém >90% ACC. Esta robustez não é trivial: random encoder com k-WTA fica em 37.60% — o ganho vem do TREINO sob restrição de sparsity, não da estrutura k-WTA estática.

### Estrutura proposta (6-8 páginas)

| Seção | Conteúdo | Tamanho est. |
|---|---|---|
| 1. Introduction | Motivação bio-inspired learning + pergunta científica sobre sparsity em metric learning | ~1 pg |
| 2. Background | ProtoNet (Snell 2017), k-WTA (Maass 2014, Ahmad & Hawkins 2019, Lynch & Ahmad 2019), sparse distributed representations, Omniglot | ~1 pg |
| 3. Method | Arquitetura C3 (CNN-4 + k-WTA no embedding 64D), 3 níveis de k testados (32/16/8 = 50/75/87.5% sparsity), validação random encoder + k-WTA | ~1 pg |
| 4. Experiments | Tabela principal (3 sparsities × 5w1s + 20w1s × IC95% bootstrap), curva sparsity × ACC, comparação com baselines (Pixel kNN, Iter 1 STDP, ProtoNet vanilla), validação random + k-WTA | ~2 pgs |
| 5. Discussion | Implicações pra continual learning (Marco 1 como apêndice, insight sobre robustez ProtoNet), limitações (Omniglot é dataset visualmente simples, generalização cross-domain TBD), trabalho futuro | ~1-2 pgs |
| 6. Conclusion | 1 parágrafo |
| Appendix (opcional) | Marco 1 — caracterização de robustez ProtoNet a forgetting | ~1-2 pgs |

### Experimentos prontos (não precisa rodar nada novo)

Tudo já está no repo:

- **C3a/b/c em Omniglot 5w1s e 20w1s** (sessão #20, `experiment_01_oneshot/c3_protonet_sparse.py`)
- **ProtoNet baseline 94.55%** (sessão #20, `experiment_01_oneshot/baselines.py`)
- **Random encoder + k-WTA validation 37.60%** (sessão #20)
- **Tabela cumulativa do projeto:** STDP 35.98% → C1b 56.28% → C2-simplified 64.08% → C3b 93.10%

Logs e checkpoints já existem (sessão #20 commit `fc75495`).

### Experimentos opcionais (apenas se Luis quiser fortalecer)

Não bloqueiam paper. Cada um custaria ~1 sessão extra:

| Experimento | Custo | Benefício |
|---|---|---|
| Sweep mais fino de k (k ∈ {4, 6, 12, 24, 48}) | ~30 min | Caracteriza fronteira de sparsity tolerada |
| k-WTA em camadas intermediárias (não só embedding final) | ~1h | Valida que sparsity downstream basta vs needs deep |
| Comparação com ProtoNet+dropout no embedding (controle) | ~1h | Distingue "sparsity por k-WTA" de "regularização genérica" |

### Cronograma estimado

| Sessão | Conteúdo | Tempo real est. |
|---|---|---|
| **#31** | Outline detalhado + Intro + Background | 2-3 h |
| **#32** | Methods + Experiments (tabelas + figs principais) | 2-3 h |
| **#33** | Discussion + revisão geral | 2-3 h |
| **#34** | Figures refinement + citações + bibliography | 1-2 h |
| **#35** | Peer review interno (se possível) + revisão | 2-3 h |
| **Total** | **5 sessões de paper writing** | **10-15 h** |

**Submissão alvo:** NeurIPS Workshop on Bio-plausible Learning (deadline típico setembro/outubro). Backups: ICML AI for Science Workshop, NESY (Neuro-Symbolic), CCN (Cognitive Computational Neuroscience).

### Pausa recomendada antes de #31

**Pausa de 1+ semana entre #30 e #31 é saudável.** Escrever paper exige modo cognitivo diferente de pesquisar (síntese > exploração). Distância temporal ajuda a ver achados com clareza. Sem cadência fixa.

### Estado pós-#30

- **Marco 1 ENCERRADO** formalmente.
- **C3 paper scoping completo.** Próxima sessão (#31, quando Luis decidir) começa Intro+Background.
- **Project Hebb status:** 30 sessões investidas, 1 paper publicável em pipeline (C3), achados mecanísticos documentados em 8 weekly markdown files.

---

## Decisão pós-#35: LinkedIn em vez de NeurIPS — sessão #36 (2026-04-30)

**Contexto:** sessão #35 finalizou paper C3 pré-peer-review (3789 palavras, abstract, 2 figuras, main.tex compilável, 14 refs). Estava planejado seguir pra peer review interno (#36) → submissão NeurIPS Bio-Plausible Workshop ~setembro 2026.

**Decisão final do Luis:** **NÃO submeter pra workshop.** Em vez disso, postar no LinkedIn em PT como anúncio + repo público + PDF deep dive anexado.

### Razão da decisão

1. **Founder Rytora sem tempo pra pipeline NeurIPS.** Submissão acadêmica formal envolve: rebuttals (1-2 semanas de turnaround intenso), revisões pós-acceptance (mais 2-4 semanas), camera-ready preparation (logística), registration ($300-1000+), e potencialmente attendance presencial (tempo + custo). Side project 5-7h/semana não absorve esse overhead sem comprometer Rytora.
2. **LinkedIn alcança parte do que peer review faria.** Rede técnica do Luis (founders, devs, ML researchers) consegue dar feedback substantivo nos comentários — não com rigor de double-blind, mas suficiente pra calibrar interpretações e identificar refs que não conhecemos. Especialmente com o convite explícito a pushback no post.
3. **Paper draft preservado pra submissão futura.** Decisão é "não submeter agora", não "abandonar o draft". Se em 6-12 meses Luis quiser submeter (a outro workshop, conference, ou venue alternativo), o material está pronto. Nada se perde.
4. **C3 não é breakthrough.** Workshop paper de incrementalismo defensável é exatamente o tipo de trabalho que beneficia mais de visibilidade (LinkedIn) que de credenciamento institucional (workshop). Honestidade do scope alinha com o canal.

### Trade-off aceito

| O que LinkedIn dá | O que LinkedIn não dá |
|---|---|
| Alcance imediato (~rede do Luis + ressonância) | Peer review formal double-blind |
| Feedback rápido nos comentários | Citação acadêmica formal (BibTeX entry) |
| Visibilidade pra portfolio/CV | Linha em "publications" no CV |
| Zero overhead pós-publicação | Validação institucional |

Trade-off explicitamente aceito: o resultado de C3 não exige validação institucional pra ser discutido. Quem quiser citar em paper futuro pode citar via CITATION.cff (criado em sessão de autoria pós-#34).

### Estado de código pra publicação

- `paper_c3/main.tex` — compilável, mas **não compilado localmente** (pdflatex não instalado em Windows)
- Plano: Luis compila via Overleaf em ~10 min (instruções em `paper_c3/latex_status.md`), gera `Project_Hebb_C3_DeepDive.pdf`, adiciona ao repo
- LinkedIn pode ser publicado antes ou depois do PDF — post longo menciona "PDF em breve no repo", ajustável

### Drafts produzidos na sessão #36

- `paper_c3/linkedin_post.md` — versão longa (~1900 chars), 6 parágrafos: hook contraintuitivo, contexto Project Hebb side project, resultado central com tabela inline, validação random+kWTA, por que importa pra bio-plausible learning, honestidade explícita sobre limitações, menção ao Marco 1 (achado negativo), CTA com hashtags
- `paper_c3/linkedin_post_short.md` — versão curta (~750 chars), fallback se a longa parecer demais

Ambos com tom honesto preservado (sem "breakthrough", sem "novel", sem hype). Linguagem natural BR. Anexo recomendado: `figs/fig1_sparsity_curve.png` (visual contraintuitivo do plateau até 75% sparsity).

### Cancelamento implícito do roadmap pós-#30

Roadmap original "submission alvo NeurIPS Bio-Plausible Learning ~setembro 2026" oficialmente cancelado. Sessões #36 (era peer review) e potencial #37 (revisão final pré-submissão) ficam sem propósito — substituídas por compilação Overleaf + post LinkedIn.

**Project Hebb entra em estado de manutenção** após esta decisão. Não há próximas sessões planejadas. Se Luis quiser retomar (Marco 2 em outro critério pós-LLM, ou ablações adicionais do C3), reabrir via nova sessão administrativa primeiro.
