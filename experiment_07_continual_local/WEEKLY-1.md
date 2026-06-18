# Marco 4 — WEEKLY-1: continual learning com plasticidade local

## Setup

Split-CIFAR-100, 5 tarefas × 20 classes, treino SEQUENCIAL da pilha (sem revisitar dados antigos).
`acc_matrix[t][i]` = acc na tarefa i com a pilha após treinar até a tarefa t. `BWT` = forgetting (≈0 = não esquece). Reaproveita o SoftHebb do Marco 3.

## Resultado parcial — SoftHebb sequencial (smoke, probe-8ep, seed 0)

```
ACC = 65.52%   BWT = +0.34
acc_matrix:
  67.1   .     .     .     .     (após tarefa 0)
  62.0  56.5   .     .     .
  65.4  58.8  68.9   .     .
  68.5  52.7  67.3  65.2   .
  69.6  58.9  67.7  62.9  68.7   (após tarefa 4 — linha final)
```

**A hipótese central CONFIRMA (preliminar):** o SoftHebb **não sofre catastrophic forgetting** — BWT = **+0,34** (≈ zero, levemente positivo). A tarefa 0 vai de 67,1% (recém-treinada) a **69,6%** após ver mais 4 tarefas: as features Hebbianas genéricas **melhoram**, não se degradam. É o oposto do backprop, que esqueceria.

**Mecanismo:** o SoftHebb aprende estatísticas visuais genéricas (edges/blobs) que convergem independentemente da ordem das tarefas; ver dados novos enriquece as features em vez de sobrescrevê-las.

## CONTRASTE — backprop e2e sequencial (seed 0)

```
ACC = 55.46%   BWT = -16.78
acc_matrix:
  68.3   .     .     .     .
  55.5  62.8   .     .     .
  52.0  54.7  71.1   .     .
  53.8  46.7  69.9  69.6   .
  35.2  45.6  61.8  62.0  72.7   (linha final)
```

**Catastrophic forgetting clássico:** a tarefa 0 DESPENCA de 68,3% → **35,2%** (−33 p.p.) após ver +4 tarefas. BWT = **−16,78**.

## Comparação central (preliminar, seed 0)

| Método | ACC | BWT |
|---|---|---|
| **SoftHebb** (plasticidade local) | **65,52%** | **+0,34** (não esquece) |
| backprop e2e | 55,46% | **−16,78** (esquece) |

**Gap de forgetting = +17,1 p.p. a favor do SoftHebb.** E o SoftHebb tem ACC *maior*. **Isto é potencialmente o achado mais forte do projeto:** pela 1ª vez a bio-inspiração **SUPERA o backprop** numa capacidade — justamente uma que o backprop *não tem* (continual sem esquecimento). No Marco 3, SoftHebb ficava ABAIXO do backprop (80 vs 87); aqui, na capacidade pós-LLM real, fica ACIMA.

**Critério literal (PLAN.md) — 2 de N condições já batem:** SoftHebb BWT ≥ −5 ✓ (+0,34); backprop BWT ≤ −15 ✓ (−16,78). FALTA p/ fechar SUCESSO: (a) controle backprop-NÃO-sup (isolar localidade vs não-sup), (b) SoftHebb joint (acc dentro de 5pp?), (c) 3 seeds + IC95%.

## Próximos passos (próxima sessão / iteração)

1. **CONTRASTE backprop e2e sequencial** (multi-head) — esperado BWT muito negativo (esquece). É o que torna o achado do SoftHebb significativo.
2. **CONTROLE-CHAVE backprop NÃO-supervisionado sequencial** (autoencoder) — isola se a resistência vem da *localidade Hebbiana* ou só do *não-supervisionado* (a lição do GRU/#78).
3. SoftHebb joint (upper bound) + random (lower bound).
4. 3 seeds + IC95% + avaliar contra o critério literal (PLAN.md): SUCESSO = SoftHebb BWT ≥ −5 E backprop BWT ≤ −15 E gap vs backprop-não-sup ≥ +8.

## Honestidade

Resultado é de 1 seed, probe curto (8 ep), e SEM o contraste backprop ainda — então é **preliminar e promissor, não conclusivo**. O sinal (BWT ≈ 0) é forte, mas o achado só fecha com o backprop esquecendo E o controle não-sup isolando a localidade. Não inflar antes disso.
