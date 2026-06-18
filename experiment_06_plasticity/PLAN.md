# Marco 3 — Plasticidade local sem backprop (eixo SoftHebb)

> **Reaberto pós-#78** (Luis decidiu pivotar para plasticidade sem backprop; eixo SoftHebb escolhido após scoping de 7 famílias — ver `writeups/plasticidade-landscape.md`). Este é o primeiro teste LIMPO da premissa-mãe do projeto (CONTEXT §1): *plasticidade local, online, sem backprop end-to-end*. Critério literal fixado ANTES do código.

## Por que este eixo (e não os becos já mapeados)

O projeto já descartou: **STDP biofísico puro** (Exp 01 — saturou, ~80% do sinal era arquitetural) e **differentiable plasticity** (C2 — outer-loop é backprop, termo Hebbian dispensável). SoftHebb (Journé et al., ICLR 2023) evita os dois becos:
- **Regra local fechada de verdade:** `requires_grad=False` na pilha, ZERO autograd — não há backprop nem global nem local para disfarçar.
- **Mecanismo não-dispensável por construção:** desligar a competição WTA literalmente reduz a regra a random features — então o controle de ablação E o controle random são a *mesma* pergunta, binária e honesta.
- **Corrige os 2 bugs do Exp 01:** a regra normaliza (Oja-like, evita saturação/rich-get-richer) e a competição está na *dinâmica* (soft-WTA), não como penalidade pós-hoc.

## Pergunta de pesquisa

Uma representação aprendida **só** por plasticidade Hebbiana competitiva local (pré × pós-competido, sem backprop na pilha) carrega sinal REAL — isto é, supera a MESMA arquitetura com pesos random congelados (+ mesma competição + mesmo probe) por margem grande — ou o ganho é arquitetural, como foi no STDP?

## Critério literal (fixado ANTES — não mover depois)

Métrica: acurácia de **linear-probe** (classificador linear supervisionado treinado sobre features CONGELADAS da pilha Hebbiana) em CIFAR-10. 3 seeds, IC95% bootstrap, HP fixados antes.

| Resultado | Condição |
|---|---|
| **SUCESSO** | probe ≥ **75%** E supera random+WTA por ≥ **15 p.p.** E fica a ≤ **15 p.p.** do backprop e2e (mesma arquitetura) |
| **MEDIANO** | probe 65–75% E supera random+WTA por 8–15 p.p. |
| **FALHA** | probe < 65% **OU** supera random+WTA por < 8 p.p. (a competição Hebbiana não carregou o sinal — o STDP de novo) |

## Predição registrada (antes do experimento)

Mediano provável, inclinado a Sucesso. SoftHebb reporta ~80% CIFAR-10 (acima do critério bruto), mas a incógnita real é o **random-control**: WTA + conv + patch-norm + probe linear já dão muita acurácia sozinhos. Aposto que o gap Hebbian-vs-random fica em **10–20 p.p.** — positivo, mas menor que o número bruto sugere. Estimativa: **~55% Sucesso limpo / ~35% Mediano / ~10% o random come quase tudo** (replay do STDP, achado negativo forte e definitivo).

## Baseline + controles (a lição do STDP: rodar o controle ANTES de concluir)

1. **Backprop end-to-end** na MESMA arquitetura (teto, ~90–95% esperado) + ProtoNet/ref para contexto.
2. **CONTROLE (a) — random+WTA+probe:** pesos random congelados + mesma competição WTA + mesmo linear-probe. **Gatilho de fechamento:** só declarar "carregou sinal" se Hebbian+WTA bate (a) por ≥ 15 p.p. Rodar ANTES do experimento principal.
3. **CONTROLE (b) — WTA desligado:** Hebbian sem competição (verifica que a competição é o motor).

## Tarefa e viabilidade

CIFAR-10 (nem toy nem inviável; é o regime onde SoftHebb tem número defensável). Pilha conv Hebbiana single-pass → features congeladas → linear-probe. Roda em minutos-horas na RTX 4070, single-pass, **sem outer-loop de meta-aprendizado**. Opcional barato (frente honesta): 1 época single-pass para testar a premissa ONLINE de verdade.

## Plano de implementação (4–6 sessões)

| Sessão | Conteúdo |
|---|---|
| 1 | Setup: CIFAR-10 (torchvision) + **baseline backprop** na arquitetura-alvo (estabelece teto) |
| 2–3 | Camada SoftHebb (Hebbian local + soft-WTA + normalização Oja-like, `requires_grad=False`) + pilha conv + linear-probe |
| 4 | **Controles (a) random+WTA e (b) WTA-off** — antes da conclusão principal |
| 5 | Multi-seed (3) + IC95% + análise contra o critério |
| 6 | Buffer / frente online (1 época single-pass) |

## Referências

- **Journé, Rodriguez, Guo, Moraitis (2023)** — "Hebbian Deep Learning Without Feedback", ICLR 2023. CIFAR-10 80.3%, STL-10 76.2%, ImageNet 27.3% (probe). Repo oficial PyTorch.
- **Oja (1982)** — regra Hebbian normalizada (evita saturação).
- **Krotov & Hopfield (2019)** — "Unsupervised learning by competing hidden units", PNAS.
- Conexão interna: k-WTA do C3 (`paper_c3/`), o controle random é a lição do Exp 01 (#10).

## Honestidade (registrada antes)

- SoftHebb usa backprop **apenas** no linear-probe final (classificador), não na pilha de features — isso é padrão e legítimo (mede a qualidade da representação *não-supervisionada/local*). O claim é sobre a PILHA, não o probe.
- Um resultado NEGATIVO honesto aqui (Hebbian competitivo também não bate random) é valioso: fecharia a porta da "plasticidade local pura" com evidência. Não inflar.
