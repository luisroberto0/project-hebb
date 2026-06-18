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

## Próximos passos (próxima sessão / iteração)

1. **CONTRASTE backprop e2e sequencial** (multi-head) — esperado BWT muito negativo (esquece). É o que torna o achado do SoftHebb significativo.
2. **CONTROLE-CHAVE backprop NÃO-supervisionado sequencial** (autoencoder) — isola se a resistência vem da *localidade Hebbiana* ou só do *não-supervisionado* (a lição do GRU/#78).
3. SoftHebb joint (upper bound) + random (lower bound).
4. 3 seeds + IC95% + avaliar contra o critério literal (PLAN.md): SUCESSO = SoftHebb BWT ≥ −5 E backprop BWT ≤ −15 E gap vs backprop-não-sup ≥ +8.

## Honestidade

Resultado é de 1 seed, probe curto (8 ep), e SEM o contraste backprop ainda — então é **preliminar e promissor, não conclusivo**. O sinal (BWT ≈ 0) é forte, mas o achado só fecha com o backprop esquecendo E o controle não-sup isolando a localidade. Não inflar antes disso.
