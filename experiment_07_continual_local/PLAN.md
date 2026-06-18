# Marco 4 — Continual learning com plasticidade local Hebbiana

> **Confirmado pelo Luis (2026-06-18):** "unir [o SoftHebb] a uma capacidade pós-LLM real que o backprop não tem — continual learning sem esquecimento". Reaproveita o harness do Marco 3 (`experiment_06_plasticity/softhebb_cifar.py`). Critério literal fixado ANTES do código (padrão do projeto). **Roadmap: Marco 4 (este) → Marco 5 escala ImageNet → Marco 6 hardware neuromórfico.**

## Contexto (de onde viemos)

- **Marco 3 (SoftHebb):** plasticidade local Hebbiana, sem backprop, **online (single-pass)**, aprende features genéricas úteis — 80% CIFAR-10, +12 p.p. sobre random, a 6.8 p.p. do backprop. 1º positivo limpo da premissa-mãe.
- **Marco 1 (continual, #21-30):** FALHOU — ProtoNet/backprop era robusto, mecanismos bio não agregavam. Mas usava backprop. **Marco 4 revisita com o mecanismo que funciona (SoftHebb local).**

## Pergunta de pesquisa

A plasticidade local Hebbiana (SoftHebb), treinada em tarefas **sequenciais** (sem revisitar dados antigos), sofre **menos catastrophic forgetting** que o backprop — porque aprende features **genéricas não-supervisionadas** (edges/blobs estáveis para qualquer tarefa), em vez de otimizar para a tarefa atual?

## Hipótese / predição (registrar ANTES)

- **SoftHebb sequencial:** forgetting (BWT) ≈ 0 — as features são estatísticas visuais genéricas que convergem independentemente da ordem das tarefas; ver tarefa 2 não destrói o que serve à tarefa 1.
- **Backprop e2e sequencial:** BWT muito negativo — otimiza para a tarefa atual, sobrescreve features antigas.
- **Predição honesta:** SoftHebb deve esquecer dramaticamente menos. Risco: pode ser que isso valha para *qualquer* método não-supervisionado (não ser específico da localidade Hebbiana) — daí o controle crítico abaixo.

## Protocolo

**Dataset:** Split-CIFAR-100 — 100 classes divididas em **T tarefas** (começar com T=5 × 20 classes; testar T=10 × 10). Reaproveita o loader CIFAR-100 do Marco 3 (`cifar100.npz`).

**Treino sequencial:** a pilha de features é treinada tarefa-a-tarefa (tarefa 1, depois 2, ... sem revisitar dados antigos), online/single-pass.

**Métricas (padrão continual learning):**
- **ACC** = acurácia média sobre todas as T tarefas após treinar todas (linear-probe nas features finais congeladas, treinado com labels de todas as classes vistas).
- **BWT (backward transfer / forgetting)** = média sobre i de [acc na tarefa i ao FINAL − acc na tarefa i logo após treiná-la]. Negativo = esquece. **Esta é a métrica central.**

## Condições a comparar (5)

1. **SoftHebb sequencial** (plasticidade local não-sup) — o método-alvo.
2. **Backprop e2e sequencial** (CNN supervisionada treinada por tarefa) — o que sofre forgetting clássico.
3. **CONTROLE-CHAVE (a lição do GRU/#78): backprop NÃO-supervisionado sequencial** (autoencoder ou SimCLR-leve treinado sequencialmente) — isola se a resistência a forgetting vem da **localidade Hebbiana** ou só do **não-supervisionado**. Se o backprop-não-sup TAMBÉM não esquece → não é especial do Hebbiano. Se SÓ o SoftHebb não esquece → a localidade é o que importa.
4. **SoftHebb JOINT** (treina em todos os dados juntos) — upper bound (sem forgetting possível).
5. **Random features** — lower bound.

## Critério literal (fixar ANTES; refinar com 1 smoke se necessário)

- **SUCESSO:** SoftHebb seq **BWT ≥ −5 p.p.** (quase não esquece) **E** backprop e2e seq **BWT ≤ −15 p.p.** (esquece muito) **E** SoftHebb seq acc dentro de 5 p.p. do SoftHebb joint **E** o gap de forgetting (SoftHebb vs backprop-não-sup do controle 3) ≥ +8 p.p. (a localidade importa, não só o não-sup).
- **MEDIANO:** SoftHebb esquece menos que backprop e2e por 5–15 p.p., mas o controle 3 (backprop-não-sup) também resiste (→ é o não-sup, não a localidade).
- **FALHA:** SoftHebb esquece tanto quanto backprop e2e (gap < 5 p.p.) — plasticidade local não dá vantagem em continual.

3 seeds, IC95% bootstrap, HP fixos antes.

## Plano de código (reaproveita Marco 3)

| Passo | Conteúdo |
|---|---|
| 1 | `split_cifar.py` — divide CIFAR-100 em T tarefas (do `cifar100.npz`); loaders por tarefa |
| 2 | `continual.py` — treino sequencial da pilha SoftHebb (reusa `SoftHebbConv2d`/`DeepSoftHebb` do Marco 3) + medição ACC/BWT (probe por tarefa) |
| 3 | Condições 2 e 4-5 (backprop e2e seq, joint, random) — reusa `DeepBackpropCNN` |
| 4 | **Condição 3 (controle-chave): backprop não-sup sequencial** (autoencoder simples na mesma arquitetura) |
| 5 | Sweep 3 seeds + análise contra critério + WEEKLY |

## Honestidade (registrada antes)

- O SoftHebb é não-supervisionado; "não esquecer" pode ser trivial se ele não aprende nada específico. Por isso o critério exige **acc útil** (≥ dentro de 5pp do joint) E o **controle backprop-não-sup** (isola a localidade). Sem esses, "não esquece" não significaria nada.
- Reportar BWT de TODAS as condições; não inflar. Um resultado onde o backprop-não-sup também resiste seria honesto e informativo (resistência vem do não-sup, não do Hebbiano).
