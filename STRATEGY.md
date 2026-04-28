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
