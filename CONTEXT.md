# CONTEXT.md — Project Hebb

> Briefing do projeto. Leia antes de qualquer ação no repositório.

---

## 1. Missão

Construir uma arquitetura de IA fundamentalmente nova, inspirada em neurociência, que ofereça capacidades que LLMs não têm:

- **Aprendizado contínuo** sem catastrophic forgetting
- **Aprendizado one-shot real** (formar representação durável a partir de 1 exemplo)
- **Eficiência radical** — rodar em CPU comum, não em clusters de GPU
- **Raciocínio temporal** via timing de spikes (event-driven, esparso)

Não é incrementalismo de LLM. É outra premissa: em vez de predição estatística de tokens em redes densas treinadas com backpropagation, usamos **plasticidade local diferenciável** — regras de update que operam só com sinais locais (pré, pós, modulação) e podem rodar online, sem backprop end-to-end.

> "Não tente construir a mente. Construa um neurônio que funcione diferente."
> — Luis Roberto Pinho da Silva Junior, Project Hebb (2026)

### Author

**Luis Roberto Pinho da Silva Junior** — Independent research
[LinkedIn: luisroberto0](https://www.linkedin.com/in/luisroberto0/)
Project Hebb (2026)

### 1.1 Refino #21 (2026-04-28, pós-sessão #20)

A missão original (acima) foi escrita antes das 20 primeiras sessões. Após elas, dois refinos honestos:

**(1) "STDP biofísico fiel" → "plasticidade local diferenciável".**
Sessões #1-#13 mostraram que STDP aditivo + k-WTA + clamp na parametrização Diehl & Cook (2015) tem barreira estrutural; melhor produzido foi 35.98% 5w1s com matched filter trivial. Sessões #17-#19 mostraram que plasticidade meta-aprendida (estilo Najarro & Risi 2020) é mais próxima de "differentiable plasticity" (Miconi 2018) que de Hebb biofísico — ablações refutaram que o termo Hebbian puro `A·pre·post` carrega o sinal. **A missão pós-LLM não exige fidelidade biofísica; exige mecanismos com propriedades target — locais, online, sem backprop global.** Isso amplia o espaço de exploração sem perder a tese central.

**(2) Sucesso numérico ≠ sucesso mecanístico.**
Sessão #20 (C3: ProtoNet + k-WTA esparso) atinge as duas metas numéricas do §4 (≥90% 5w1s, ≥70% 20w1s) MAS usa backprop end-to-end. **Atingiu metade da meta.** Documentar como milestone publicável (workshop) é honesto; tratar como objetivo cumprido seria desonesto. Próximo marco do projeto é **continual learning sem replay buffer** — capacidade que LLMs não têm, e que plasticidade local meta-aprendida pode atacar diretamente. Ver `STRATEGY.md` "Decisão Pós-Sessão #20".

### 1.2 Refino #30 (2026-04-29, pós-sessão #29) — Marco 1 encerrado

Após 9 sessões dedicadas a Marco 1 (continual learning sem replay buffer), 4 abordagens testadas — todas ≤ baseline naive ProtoNet (80.65%):

| Abordagem | ACC 5w1s | Sessão |
|---|---|---|
| Naive ProtoNet random splits | 82.58% | #23 |
| Naive ProtoNet alphabets+no-warmup | 80.65% | #25 |
| Possibilidade B encoder linear | 47.89% | #27 |
| Caminho 5e (CNN+plast+trace+k-WTA) | 74.78% | #29 |

Pelo critério literal definido em `STRATEGY.md` "Decisão Pós-Sessão #27" (#29 ACC 74.78% < 78% → Caminho 4), Marco 1 encerrado.

**Achado mecanístico positivo:** ProtoNet metric learning é inerentemente robusto a catastrophic forgetting em Omniglot — prototypes-fresh-no-eval + encoder métrica genérica protegem naturalmente. Mecanismos bio-inspirados (plasticidade meta-aprendida, trace STDP-like, k-WTA) não conseguem bater quando combinados com prototype-based metric learning em Omniglot. Insight científico documentado, vira apêndice possível do paper C3.

**Status atualizado das capacidades pós-LLM (§1):**

- ✅ **Aprendizado one-shot real:** atingido numericamente via C3 (93.10% 5w1s, 80.72% 20w1s). Mecanismo: ProtoNet + k-WTA esparso. Restrição não cumprida: usa backprop end-to-end (não plasticidade local). Aceito como milestone publicável.
- ❌ **Aprendizado contínuo sem catastrophic forgetting:** Marco 1 encerrado em #30 sem produzir método novo que bata baseline. ProtoNet naive já é robusto; bio-inspired methods não agregam. Achado caracterizado, mas hipótese pós-LLM não foi avançada além do baseline.
- 🔵 **Eficiência radical em hardware consumer:** não atacado nesta fase.
- 🔵 **Raciocínio temporal via timing:** não atacado nesta fase.

**Próximo passo:** paper C3 (workshop NeurIPS Bio-Plausible Learning ~setembro 2026). Ver `STRATEGY.md` "Plano paper C3 — Workshop NeurIPS Bio-Plausible Learning". Após paper, reavaliação do projeto pra decidir se vale atacar outros critérios pós-LLM ou encerrar Project Hebb como exploração documentada com 1 milestone publicado.

### 1.3 Refino #36 (2026-04-30) — Publicação via LinkedIn em vez de NeurIPS

Após sessão #35 finalizar paper C3 pré-peer-review, decisão estratégica do Luis: **NÃO submeter pra NeurIPS Bio-Plausible Workshop.** Em vez disso, postar no LinkedIn em PT como anúncio + repo público + PDF deep dive anexado.

Razão: founder Rytora não tem espaço pra rebuttals/revisões/registration/attendance que submissão acadêmica formal exige. LinkedIn alcança parte do que peer review faria (feedback técnico nos comentários) sem overhead institucional. Paper draft preservado pra submissão futura se fizer sentido depois.

**Implicação pra status §1.2:**
- ✅ one-shot via C3 — atingido numericamente, **publicado via LinkedIn** (em vez de workshop)
- ❌ continual learning — Marco 1 encerrado
- 🔵 eficiência radical e raciocínio temporal — não atacados

**Project Hebb entra em estado de manutenção pós-#36.** Não há próximas sessões planejadas. Se houver retomada (Marco 2 em outro critério pós-LLM, ablações C3 adicionais), reabrir via sessão administrativa primeiro. Ver `STRATEGY.md` "Decisão pós-#35: LinkedIn em vez de NeurIPS".

---

## 2. Princípio Operacional

Esse projeto **não é** o Rytora (BuildLabs/VoiceLabs). É pesquisa de longo prazo, separada do trabalho comercial. Métrica não é receita — é **insight verificável**.

90% do tempo é leitura, hipótese, refinamento conceitual. 10% é experimento. **Não otimize código antes de validar ideia.**

---

## 3. Stack Técnico

| Camada | Escolha | Justificativa |
|---|---|---|
| Linguagem principal | **Julia 1.10+** | Performance científica + sintaxe limpa pra prototipagem matemática. Rust fica pra port de produção *depois* que o modelo estiver validado. |
| Simulação SNN (Julia) | SpikingNeuralNetworks.jl, DifferentialEquations.jl | Nativo |
| Simulação SNN (referência) | Brian2 (Python), NEST | Pra reproduzir papers e validar implementações Julia |
| Deep learning híbrido | Flux.jl (Julia), snnTorch / Norse (Python) | Quando precisar comparar com baselines neurais |
| Notebook | Pluto.jl | Melhor que Jupyter pra Julia |
| Versionamento | Git + GitHub | Desde o dia 1 |
| Gestão de papers | Zotero | |
| Documentação de pesquisa | Markdown no próprio repo | Tudo aqui |

**Hardware disponível:** Notebook i9 + RTX 4070. Suficiente pra simular centenas de milhares a milhões de neurônios spiking. Liga de hardware da pesquisa publicada em NeurIPS. Hardware **não é gargalo** — iteração mental é.

---

## 4. Foco da Fase 1 (meses 1–3)

Três direções foram avaliadas:

1. **Aprendizado one-shot com plasticidade local** ← **ESCOLHIDO**
2. Memória episódica artificial (replay-based, hippocampus-inspired)
3. Raciocínio causal explícito (Pearl-style)

**Por que one-shot primeiro:** teoria mais madura (STDP, Hebbian learning), mais barato computacionalmente, benchmarks prontos (**Omniglot**, Lake et al. 2015), iteração rápida em CPU, progresso mensurável.

### Critério de sucesso da Fase 1

Ao final do mês 3, ter:

- [ ] Compreensão clara de por que STDP > backprop pra aprendizado contínuo
- [ ] Simulação rodando 10.000+ spiking neurons em tempo real em CPU
- [ ] Demonstração de aprendizado one-shot em Omniglot N-way K-shot, sem catastrophic forgetting
- [ ] Documento técnico de 5-10 páginas descrevendo hipótese e arquitetura
- [ ] Identificação clara do que torna a abordagem diferente do estado da arte

### Baselines a bater

| Baseline | Tipo | Acurácia esperada (5-way 1-shot Omniglot) |
|---|---|---|
| Pixel kNN | Trivial baseline | ~50–70% |
| Prototypical Networks (Snell 2017) | Deep metric learning | ~98% |
| MAML (Finn 2017) | Meta-learning | ~98% |
| **Nosso modelo SNN+STDP+Hopfield** | **Alvo** | **≥90% (5-way 1-shot), ≥70% (20-way 1-shot), sem backprop end-to-end** |

> Números alinhados com Lake et al. 2015, Snell et al. 2017 e Finn et al. 2017. O ponto não é bater o estado da arte em acurácia bruta — é fazer competitivamente *sem backprop global*, *sem labels no pretreino*, *com plasticidade local*.

---

## 5. Arquitetura Conceitual (v0.1)

```
┌─────────────────────────────────────────────────┐
│             SISTEMA COGNITIVO v0.1              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌───────────┐    ┌───────────┐                 │
│  │ SENSORY   │───▶│ CORTEX    │ Rede de SNNs    │
│  │ INPUT     │    │ (lento)   │ com STDP        │
│  └───────────┘    └─────┬─────┘                 │
│                         │                       │
│                   ┌─────▼─────┐                 │
│                   │HIPOCAMPO  │ Aprendizado     │
│                   │ (rápido)  │ rápido + replay │
│                   │  Hopfield │ (Ramsauer 2020) │
│                   └─────┬─────┘                 │
│                         │                       │
│                   ┌─────▼─────┐                 │
│                   │ DECISÃO   │ Integração +    │
│                   │           │ Classificação   │
│                   └───────────┘                 │
│                                                 │
└─────────────────────────────────────────────────┘
```

Dois sistemas de memória (córtex lento / hipocampo rápido) é a tese central — espelha a divisão biológica e ataca catastrophic forgetting na raiz.

---

## 6. Papers Essenciais

**Fundamentos SNN/STDP:**
- Maass (1997) — *Networks of Spiking Neurons: The Third Generation of Neural Network Models*
- Diehl & Cook (2015) — *Unsupervised learning of digit recognition using spike-timing-dependent plasticity* ← **paper-base do nosso pipeline**
- Zenke & Ganguli (2018) — *SuperSpike*
- Bellec et al. (2020) — *e-prop*

**Memória/atenção bio-plausível:**
- Ramsauer et al. (2020) — *Hopfield Networks Is All You Need* ← **base do módulo de memória rápida**

**One-shot / meta-learning (baselines):**
- Lake et al. (2015) — *Human-level concept learning through probabilistic program induction* (Omniglot)
- Snell et al. (2017) — *Prototypical Networks for Few-shot Learning*
- Finn et al. (2017) — *MAML*

**Livros de referência:**
- *Neuronal Dynamics* — Gerstner et al. (gratuito online)
- *The Computational Brain* — Churchland & Sejnowski
- *How to Build a Brain* — Eliasmith (arquitetura SPAUN)
- *Principles of Neural Design* — Sterling & Laughlin

---

## 7. Estrutura do Repositório (atual)

```
project-hebb/
├── CONTEXT.md          ← este arquivo (briefing do projeto)
├── PLAN.md             ← plano operacional, fases, próximos passos
├── README.md           ← guia de uso (comandos, baselines, avaliação)
├── config.py           ← hiperparâmetros (STDP, LIF, treino, avaliação)
├── data.py             ← Omniglot + EpisodeSampler N-way K-shot
├── model.py            ← ConvSTDPLayer + HopfieldMemory + STDPHopfieldModel
├── train.py            ← loop de pretreino STDP
├── evaluate.py         ← N-way K-shot com 1000 episódios + IC bootstrap
├── baselines.py        ← Pixel kNN + Prototypical Networks
├── validate_environment.py
├── environment.yml
└── requirements.txt
```

> **Nota:** o scaffold inicial está em Python por pragmatismo de ecossistema (Brian2, snnTorch, datasets prontos). A migração da camada central pra Julia entra na Fase 2, *depois* de validar que o modelo funciona. Não inverter a ordem.

---

## 8. Como Operar Neste Repositório (instruções pro agente)

1. **Sempre leia `PLAN.md` antes de agir.** Ele contém a fase atual e o próximo entregável concreto. Se `PLAN.md` divergir desse `CONTEXT.md`, o `CONTEXT.md` ganha — atualize o `PLAN.md`.

2. **Antes de implementar STDP do zero, rode o pipeline end-to-end com baselines.** Sequência obrigatória:
   ```
   python evaluate.py --ways 5 --shots 1 --episodes 100   # confirma que pipeline fecha (~20%, chance)
   python baselines.py --baseline pixel_knn               # número real a bater (~70% benchmark)
   ```
   Só depois implementar a regra STDP e iterar.

3. **Commite após cada fase concluída.** Mensagem de commit descreve o que foi validado, não só o que mudou.

4. **Se um experimento falhar, registre em `PLAN.md` na seção "Notas de iteração".** Falhas informam mais que sucessos.

5. **Não otimize prematuramente.** CPU primeiro, sempre. GPU só quando o modelo estiver provado e o gargalo for medido.

6. **Não inflar escopo.** Se aparecer ideia nova legítima, anote em `IDEAS.md` (cria se não existir) e siga o `PLAN.md`. Disciplina > entusiasmo.

7. **Mantenha o repositório autossuficiente.** Qualquer pessoa (incluindo outra sessão sua) deve conseguir reabrir o projeto e entender o estado em < 10 minutos lendo `CONTEXT.md` + `PLAN.md` + `README.md`.

---

## 9. Histórico — Como Chegamos Aqui

- Pesquisa originada em conversas no claude.ai sobre alternativas a LLMs
- Decisão Julia vs Rust: Julia primeiro (iteração), Rust depois (produção, se necessário)
- Decisão de escopo: três frentes possíveis (one-shot / memória episódica / causalidade) → começar por **one-shot com plasticidade local** por ser a mais madura teoricamente
- Decisão de pragmatismo: scaffold Python pra reaproveitar ecossistema científico, port pra Julia depois de validar modelo
- Sessão Cowork inicial gerou o scaffold atual (`PLAN.md` + 9 arquivos de código + ambiente)

---

## 10. Visão de Longo Prazo

Fase 1 (meses 1–3) — Fundação. Estamos aqui.
Fase 2 (meses 3–6) — Formular modelo teórico próprio. Documento de arquitetura.
Fase 3 (meses 6–12) — Prova de conceito demonstrando capacidade que LLMs não têm.
Fase 4 — Paper, demo pública, comunidade.

Documente tudo desde o dia 1. Se isso funcionar, a história desse projeto vai importar.
