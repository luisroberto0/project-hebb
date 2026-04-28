# Experiment 02 — Continual Learning sem replay buffer

> Plano operacional do Marco 1 do Projeto B reformulado.
> Criado na sessão #22 após confirmação das 5 decisões pós-#21.
> Para framing estratégico, ver `STRATEGY.md` "Confirmação Pós-Sessão #21".

---

## Pergunta de pesquisa

> "Pode uma rede com **plasticidade local meta-aprendida** (estilo C2 do experimento 01) ou **encoder esparso ProtoNet com k-WTA** (estilo C3 do experimento 01), **sem replay buffer**, atingir **average accuracy ≥75% em Split-Omniglot 50-tasks sequenciais com forgetting (BWT) ≤−10%**, batendo baseline **EWC** por ≥3 p.p. em average accuracy?"

Carry-over literal de `STRATEGY.md` "Decisão Pós-Sessão #20" item (c).

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

**Alternativas consideradas e rejeitadas:**

- *Permuted MNIST*: clássico mas demais (cada "task" é só uma permutação de pixels, não classes diferentes). Menos representativo.
- *Split CIFAR-10/100*: maior compute, dataset mais difícil — escapa do scope side project.
- *CLEAR*: muito grande, requer infra cloud.

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

Carry-over literal de `STRATEGY.md` "Decisão Pós-Sessão #20" item (d):

| Resultado | Decisão |
|---|---|
| ACC ≥75% E BWT ≥−10% E bate EWC por ≥3 p.p. | **Sucesso → escrever paper** (workshop em 8 sessões, conference em 12-16). |
| ACC 65-75%, ou BWT entre −10% e −20%, ou bate EWC por <3 p.p. | **Resultado mediano → reavaliar** se vale workshop paper ou pivotar. |
| ACC <65% OU BWT <−20% OU pior que EWC | **Pivot:** plasticidade meta-aprendida não é o motor. Considerar replay-light ou pivotar pra outro critério pós-LLM. |
| Nada funciona após 20 sessões + ablações exaustivas | **Encerrar como exploração documentada.** Aprendizado pessoal mantido. |

---

## Roadmap previsto (sessões #22-#31)

| # | Tipo | Objetivo |
|---|---|---|
| **22 (esta)** | Admin | Confirmação decisões + literatura review (PAPERS.md) |
| 23 | Code | Baseline naive sequential fine-tuning |
| 24-25 | Code | Baseline EWC + escalar pra 50 tasks |
| 26-28 | Code | C2-continual e/ou C3-continual implementação |
| 29 | Code | Ablações |
| 30 | Admin | Status check sucesso/pivot/encerramento |
| 31 | TBD | Refinement ou paper writing conforme #30 |

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
