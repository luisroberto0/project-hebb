# Figures — paper C3

Geradas via `paper_c3/generate_figures.py` (sessão #35).

| Figura | Conteúdo | Inserir em |
|---|---|---|
| `fig1_sparsity_curve.{png,pdf}` | Sparsity-Accuracy Trade-off Curve. X: sparsity %. Y: accuracy %. Duas curvas (5w1s e 20w1s) com error bars IC95%. Linha de referência ProtoNet baseline. Anotação destacando C3b (75% sparse, −1.45 p.p.). | Section 4 (Experiments), subseção 4.3 |
| `fig2_validation.{png,pdf}` | Validation comparison bar chart. ProtoNet baseline + C3a/b/c trained vs Random encoder + k-WTA. Highlights gap +55.50 p.p. entre treinado e random. Linhas chance (5w e 20w). | Section 4 (Experiments), subseção 4.4 |

## Regenerar

```bash
cd paper_c3
python generate_figures.py
```

Dados são hardcoded no script (sessão #20 commit `fc75495`). Não rerodada `c3_protonet_sparse.py` necessária — IC95% e accuracies já documentados em `experiment_01_oneshot/WEEKLY-2.md` sessão #20.

## Resolução

PNG e PDF gerados em **300 DPI** (workshop submission standard). PDFs vetoriais — escalam sem perda. PNGs úteis pra preview/markdown.

## Style notes

- Cores: matplotlib defaults (`#1f77b4` blue para 5w1s, `#d62728` red para 20w1s)
- Fonte: matplotlib default (DejaVu Sans). Conferir contra workshop template oficial quando disponível
- Error bars: IC95% bootstrap (1000 reamostragens)
- Annotations sutis pra não poluir
