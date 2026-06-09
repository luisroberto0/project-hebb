# Experimento 04 — Eficiência radical: inferência event-driven (Marco 2-B)

> **Status:** **ENCERRADO em #70 (2026-06-08).** Eixo A respondido com **Falha decisiva, confirmada formalmente** (sweep 5 seeds, IC95%, Fashion-MNIST). Eficiência radical via SNN NÃO se realiza em CPU von Neumann.
> **Resultado (#70):** denso 87.16% / melhor SNN 84.82% (−2.34 p.p.); SynOps 7× a 0.21× (melhor caso 4.79× menos custa −14.6 p.p. acc); latência CPU 80–327× pior sempre; inferência event-driven (sparse) ainda mais lenta que runtime denso. Nenhuma config bate acc −2pp E SynOps ≥5× E latência ≤ denso.
> **Achado:** eficiência neuromórfica é co-design HW-algoritmo (silício dedicado), não do algoritmo em CPU. Detalhes: `WEEKLY-1.md` + `STRATEGY.md` "Fechamento Marco 2-B — sessão #70".
> _(Plano original abaixo, preservado para histórico. Eixos B/C/2-B.2 ficam como possíveis extensões futuras.)_
>
> **Limite hard original:** ~10 sessões (#67-#76) — fechado antecipadamente em #70 (evidência inequívoca).

---

## Posicionamento na missão pós-LLM

CONTEXT.md §1 lista 4 capacidades pós-LLM. Marco 2-B ataca a terceira: **eficiência radical** — §1 linha 13, literal: *"rodar em CPU comum, não em clusters de GPU"*. É, das 4 capacidades, a **mais nativamente neuromórfica ainda não testada**: a vantagem estrutural de SNNs (computação event-driven, esparsa) é exatamente sobre *fazer menos trabalho*, não sobre acurácia.

Conexão com o histórico do projeto: o paper C3 estabeleceu expertise em **k-WTA** (esparsidade espacial sobre embeddings). Marco 2-B leva k-WTA para o domínio **temporal** (≤ k spikes por timestep), atacando esparsidade onde ela vira economia computacional. Não é "rodar snntorch"; é testar se a esparsificação que o projeto entende compra eficiência real.

---

## Pergunta científica

> **"Uma SNN-LIF classifica Fashion-MNIST com acurácia comparável (dentro de −2 p.p.) a um MLP denso de mesma arquitetura fazendo substancialmente menos trabalho na inferência — e essa economia se materializa em CPU comum, ou fica restrita à contagem teórica de SynOps? k-WTA temporal aumenta a esparsidade o suficiente para inverter o trade-off?"**

### Outcomes informativos

1. **Sucesso (eficiência radical real):** SNN+k-WTA bate o duplo critério → evidência de que esparsidade temporal entrega eficiência mensurável até em von Neumann. Justifica paper/post.
2. **Mediano (vantagem teórica não realizada):** SynOps menores mas latência CPU pior → achado: eficiência neuromórfica exige co-design de hardware/runtime event-driven; framework denso em CPU não a captura. Defensável e relevante pra tese pós-LLM.
3. **Falha:** nem acurácia comparável nem SynOps menores.

---

## Predição provisional (registrada ANTES do experimento)

**Aritmética da motivação.** SynOps por amostra ≈ Σ_camadas (spikes pré-sinápticos × fan-out), somado sobre T timesteps. MACs do denso = um forward (784×256 + 256×10 ≈ 203k). Para uma SNN rate-coded com T=25 e densidade de spikes ~10%, SynOps ≈ 0.10 × 25 × MACs ≈ **2.5× os MACs do denso — pior, não melhor**. Para ganhar 5×, a densidade de spikes precisaria cair para ≈ 1/(5×25) = **0.8%** — esparsidade extrema que só k-WTA temporal com k pequeno força.

| Métrica | Predição |
|---|---|
| Acurácia SNN (surrogate-gradient) | comparável, ~88–90% Fashion-MNIST (denso ~89%) |
| SynOps SNN-LIF vanilla (rate, T~25) | **NÃO** ≥5× menor (provável ~2–3× **pior**; overhead de T domina) |
| SynOps SNN + k-WTA temporal (k pequeno, T pequeno) | possível ≥5× menor SE acurácia aguentar a esparsidade extrema |
| Latência CPU single-thread | **provavelmente PIOR** que o denso (T matmuls num runtime sem kernel event-driven) |

**Predição central:** critério de **Sucesso falha**; resultado mais provável = **Mediano** — a economia aparece em SynOps (com k-WTA agressivo) mas **não** em latência CPU, porque PyTorch denso não pula computação de neurônios silenciosos. Achado honesto: *eficiência radical via SNN em hardware von Neumann comum exige runtime event-driven; a vantagem é estrutural mas não se materializa sem co-design.*

---

## Dataset

**Fashion-MNIST** (Xiao et al. 2017): 60k train / 10k test, 28×28 grayscale, 10 classes. Mais discriminativo que MNIST — o trade-off acurácia↔eficiência fica visível (MNIST satura ~98–99% e esconde o custo). **MNIST como sanity** rápido do pipeline.

---

## Setup experimental

| Componente | Definição |
|---|---|
| **Baseline denso** | MLP 784→256→10 (ReLU), treino padrão Adam. Referência de acurácia e de MACs. |
| **SNN-LIF vanilla** | LIF 2 camadas (esqueleto de `validate_snn_minimal.py`), rate coding, surrogate-gradient, sweep de T. |
| **SNN + k-WTA temporal** | mesma SNN com k-WTA aplicado por timestep (≤ k spikes ativos por camada por passo), sweep de k. Variável central do marco. |

Tudo em PyTorch + snntorch (já instalado). Treino na 4070; **medição de latência em CPU single-thread** (`torch.set_num_threads(1)`) — a régua é CPU comum, não GPU.

---

## Métrica primária (dupla camada — o coração honesto do marco)

1. **Neuromórfica (teórica):** SynOps/amostra (Σ spikes × fan-out, sobre T) ÷ MACs do denso → fator de eficiência. Vantagem que existiria em hardware neuromórfico.
2. **Von Neumann (real):** latência wall-clock de inferência/amostra em **CPU single-thread**, SNN vs denso. Régua literal do CONTEXT.

**Secundárias:** % densidade de spikes por camada; sweep de T (overhead) e de k (esparsidade); acurácia vs SynOps (curva de Pareto); estimativa de energia (proxy: SynOps × E_AC vs MACs × E_MAC, com fatores de literatura).

---

## Critério literal de fechamento

Test split, **5 seeds, IC95% bootstrap** (consistente com marcos anteriores). Os três casos são exaustivos:

| Resultado | Decisão |
|---|---|
| acc dentro de −2 p.p. do denso **E** SynOps ≥ 5× menores **E** latência CPU ≤ denso | **Sucesso** → eficiência radical real |
| acc dentro de −2 p.p. **E** SynOps ≥ 5× menores **mas** latência CPU pior | **Mediano** → vantagem teórica não realizada em von Neumann (admin #76 decide paper-de-caracterização vs encerrar) |
| caso contrário (acc cai > 2 p.p. **ou** SynOps não atinge 5× menores — inclui o resultado vanilla previsto) | **Falha** → eficiência radical não atingida no eixo testado; Marco 2-B encerrado |

Critério não-negociável. Mudanças de scope (dados event-based, outro backbone, hardware neuromórfico) viram **Marco 2-B.2** (extensão), não substituição. Régua sujeita a refino numérico após lit review (#68), mas a estrutura dupla (SynOps + latência CPU) é fixa.

---

## Plano de sessões #67-#76 (orientativo, sem cadência fixa)

| # | Tipo | Output esperado |
|---|---|---|
| **67 (esta)** | Admin + scoping | Design aprovado + `PLAN.md` + atualização STRATEGY/CONTEXT |
| 68 | Lit review | `PAPERS.md` — SynOps/energia (Merolla 2014, Davies 2018/Loihi), surrogate-gradient (Neftci 2019), Fashion-MNIST SNN baselines. Refinar régua 5×. |
| 69 | Code | Baseline denso MLP Fashion-MNIST + harness de medição (SynOps counter + latência CPU) + smoke test |
| 70 | Code | SNN-LIF vanilla + sweep de T; primeira curva acc vs SynOps vs latência |
| 71 | Code | k-WTA temporal + sweep de k; curva de Pareto acc↔esparsidade |
| 72-73 | Analysis | Comparação head-to-head multi-seed, IC95% bootstrap. Critério literal? |
| 74 | Analysis | Energia estimada (proxy) + análise do gap SynOps↔latência |
| 75 | Writing | Draft de caracterização (positivo se Sucesso, negativo defensável se Mediano/Falha) |
| **76** | **Admin obrigatória** | Critério atingido? Decide próximo passo (publicar / Marco 2-B.2 event-based / Marco 2-C temporal / encerrar) |

---

## Restrições

- **Não modifica `paper_marco2a/`, `paper_c3/`, nem os scripts congelados** dos marcos anteriores.
- **Régua é CPU comum** — medir latência em CPU single-thread é obrigatório, não opcional.
- **Honestidade da dualidade:** reportar SynOps E latência sempre juntos; nunca vender vantagem de SynOps como se fosse vantagem de CPU real.
- **Sem cadência fixa.** Cancelável se #70-#71 já mostrarem evidência inequívoca.

---

## Riscos e mitigação

| Risco | Mitigação |
|---|---|
| SynOps counter implementado errado (infla/deflaciona vantagem) | Validar contra cálculo analítico num caso trivial (1 amostra, esparsidade conhecida); checar contra MACs do denso. |
| Latência CPU dominada por overhead de framework, não por trabalho real | Medir também SynOps puro; reportar os dois; usar `torch.set_num_threads(1)` e warmup. |
| k-WTA temporal destrói acurácia antes de dar esparsidade útil | É o achado, não o bug — caracterizar a curva de Pareto acc↔k. |
| Surrogate-gradient (BPTT) é backprop — "não é plasticidade local" | Reconhecido: este marco mede **eficiência de inferência**, não treino sem backprop (esse é o eixo B, marco futuro). Documentar explicitamente. |

---

## Referências (lit review a fazer em #68)

- **Merolla et al. 2014** (TrueNorth) — SynOps como métrica de energia neuromórfica
- **Davies et al. 2018** (Loihi) — eficiência event-driven em hardware
- **Neftci, Mostafa & Zenke 2019** — surrogate gradient learning em SNNs
- **Xiao et al. 2017** — Fashion-MNIST
- **Maass 2000** + paper C3 (`pinho2026kwta`) — k-WTA (base da variante temporal)
