# Experiment 02 — Continual Learning sem replay buffer

> Plano operacional do Marco 1 do Projeto B reformulado.
> Criado na sessão #22 após confirmação das 5 decisões pós-#21.
> Para framing estratégico, ver `STRATEGY.md` "Confirmação Pós-Sessão #21".

---

## Pergunta de pesquisa

**Reformulada após sessão #24** (ver `STRATEGY.md` "Reformulação Pós-Sessão #23"):

> "Em **Split-Omniglot por alfabeto** (50 tasks correspondentes aos 50 alfabetos, **sem warmup**, fine-tune sequencial), pode plasticidade meta-aprendida (estilo C2) ou ProtoNet+k-WTA (estilo C3), **sem replay buffer**, atingir **ACC absoluto ≥70%** com **BWT ≥−10 p.p.**, **batendo o baseline EWC por ≥3 p.p. em ACC**?"

**Pergunta antiga (broken pelo resultado da sessão #23, ACC 82.58% naive):** "ACC ≥75% E BWT ≥−10% E batendo EWC por ≥3 p.p. em Split-Omniglot 50-tasks com random class splits". Trivialmente atingida pelo baseline naive — anulava margem científica.

**Justificativa numérica calibrada (alvos absolutos pra Opção D = alphabets + skip warmup):**

| Linha | ACC esperado | BWT esperado |
|---|---|---|
| Naive (sem defesa) | 40-55% | −25 a −35 p.p. |
| EWC (regularization) | 55-70% | −10 a −20 p.p. |
| **Nossa proposta target** | **≥70%** (idealmente 73-78%) | **≥−10 p.p.** |
| Sky reference (GEM com replay) | 75-85% | −5 a −10 p.p. |

Critérios numéricos exatos serão confirmados após sessão #25 (medição empírica de naive na Opção D). Se naive vier em range diferente do previsto, ajustar.

---

## Benchmark

**Split-Omniglot 50-tasks.**

- Total ~1623 classes (Omniglot completo). 50 tasks × ~5 classes/task = 250 classes usadas.
- Cada task: classification supervisionada N-way (N=5), shots e queries TBD na fase de implementação.
- Tasks treinadas sequencialmente: task 1 → task 2 → ... → task 50.
- Avaliação ao final: testar em todas as 50 tasks usando classes vistas no respectivo treino.

**Por que Split-Omniglot:**

1. Já temos infra Omniglot rodando (data.py, EpisodeSampler).
2. 50 tasks é número típico em literatura continual learning (Schwarz 2018, Lopez-Paz 2017).
3. Cabe em GPU laptop (RTX 4070), ciclos rápidos.
4. Permite reuso direto do encoder ProtoNet treinado em C3.

**Estrutura específica pós-#24 (Opção D adversarial):**

- **Tasks = alfabetos.** Omniglot tem 30 alfabetos no background + 20 no evaluation = 50 alfabetos totais. Cada task é 1 alfabeto, com seus N caracteres (N varia 14-55 entre alfabetos).
- **Sem warmup.** Encoder fresh aprende cada alfabeto sequencialmente. Sem fase de pretreino genérico que dilua forgetting.
- **Episode sampling:** 5-way 1-shot 5-query, com 5 caracteres amostrados aleatoriamente dos N do alfabeto.
- **Train/test split por caractere:** mantido como pós-#23 (14 train + 6 test instances, deterministic shuffle por seed).

**Alternativas consideradas e rejeitadas (pós-#24):**

- *Permuted MNIST*: clássico mas pobre (cada "task" é só uma permutação de pixels). Menos representativo.
- *Split CIFAR-10/100*: maior compute, dataset mais difícil — escapa do scope side project.
- *CLEAR*: muito grande, requer infra cloud.
- *Random class splits (setup #23)*: insuficientemente adversarial — naive atinge 82.58%, anulando margem científica.
- *Cross-domain (Omniglot→MNIST→FashionMNIST)*: escopo grande pra side project, viola decisão (b) Pós-#21.
- *Skip warmup sozinho com random splits (Opção A)*: insuficiente — alfabetos compartilham features genéricas que warmup-skip não destrói.

---

## Métricas

Métricas padrão da área (Lopez-Paz & Ranzato 2017, Hadsell 2020):

| Métrica | Definição | Critério-alvo |
|---|---|---|
| **Average Accuracy (ACC)** | Média de accuracy nas 50 tasks após treinar todas | **≥75%** |
| **Backward Transfer (BWT)** | Média de (acc_final − acc_logo_após_treino) por task. Negativo = forgetting. | **≥−10%** |
| Forward Transfer (FWT) | Quanto cada task se beneficia das anteriores. | Tracked, não restrição |

IC95% via bootstrap (1000 resamples), n_seeds = 3-5.

---

## Critérios de fechamento

**Reformulado após sessão #24** (calibrado pra Opção D adversarial):

| Resultado | Decisão |
|---|---|
| ACC ≥70% E BWT ≥−10 p.p. E bate EWC por ≥3 p.p. | **Sucesso → escrever paper** (workshop em 8 sessões, conference em 12-16). |
| ACC 60-70%, ou BWT entre −10 e −20, ou bate EWC por <3 p.p. | **Resultado mediano → reavaliar** se vale workshop paper ou pivotar. |
| ACC <60% OU BWT <−20% OU pior que EWC | **Pivot:** plasticidade meta-aprendida não é o motor. Considerar replay-light ou pivotar pra outro critério pós-LLM. |
| Nada funciona após 20 sessões + ablações exaustivas | **Encerrar como exploração documentada.** Aprendizado pessoal mantido. |

Critérios anteriores (pós-#20, baseado em random class splits): ACC ≥75% e BWT ≥−10. Ajustados pra refletir naive baseline esperado mais baixo na Opção D adversarial.

---

## Roadmap previsto (sessões #22-#32, atualizado pós-#24)

| # | Tipo | Objetivo |
|---|---|---|
| 22 | Admin | Confirmação decisões + literatura review (PAPERS.md) — feito |
| 23 | Code | Baseline naive sequential fine-tuning, random splits — feito (ACC 82.58%, broken) |
| **24** | **Admin** | **Reformulação benchmark pra Opção D (alphabets + skip warmup) — feito** |
| **25 (próxima)** | **Code** | **Re-implementar naive em Opção D, validar empiricamente naive cai pra 40-55%** |
| 26 | Code | Baseline EWC em Opção D |
| 27-29 | Code | Propostas C2-continual e/ou C3-continual implementação |
| 30 | Code | Ablações |
| 31 | Admin | Status check sucesso/pivot/encerramento |
| 32 | TBD | Refinement ou paper writing conforme #31 |

Cancelable em qualquer ponto. Sem cadência fixa — Luis decide sessão-a-sessão.

---

## Restrições mecanísticas (conforme CONTEXT.md §1.1)

- **Sem backprop end-to-end** durante adaptação a novas tasks (pode usar pretreinamento standard como ponto de partida).
- **Sem replay buffer explícito.** Métodos com replay (GEM, A-GEM) servem como upper bound de comparação, não como solução.
- **Plasticidade local diferenciável** aceita (não exigimos Hebb biofísico estrito; ver §1.1).

---

## Não-objetivos desta fase

Pra evitar scope creep:

- **Não** otimizar pra hardware neuromórfico (Loihi etc.) nesta fase.
- **Não** atacar simultaneamente os outros 3 critérios pós-LLM (one-shot inédito, eficiência radical, raciocínio temporal).
- **Não** implementar SNN com STDP biofísico (já mostrou barreira estrutural em sessões #1-#13).
- **Não** estender pra dataset diferente de Omniglot até Marco 1 fechar.
