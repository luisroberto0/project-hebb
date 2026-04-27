# RESULTS — Experimento 01 (One-shot Omniglot com STDP + Hopfield)

> Este arquivo é **gerado automaticamente** por `analysis.py` após
> execução completa do pipeline. Os placeholders abaixo são apenas
> a estrutura — não preencher manualmente. Rode:
>
>     python analysis.py --logs-dir logs/run_<timestamp> --out RESULTS.md

## Resultado principal

**Status: [a preencher por analysis.py — sucesso forte / parcial / falha]**

## Acurácia por configuração

| Config | Episódios | Acurácia | IC 95% | Tempo (s) |
|---|---:|---:|:---:|---:|
| 5-way 1-shot | _ | _ | _ | _ |
| 5-way 5-shot | _ | _ | _ | _ |
| 20-way 1-shot | _ | _ | _ | _ |
| 20-way 5-shot | _ | _ | _ | _ |

## Comparação com baselines

| Modelo | 5-way 1-shot | Backprop end-to-end | Parâmetros |
|---|---:|:---:|---:|
| Random (chance) | 20.00% | n/a | 0 |
| Pixel kNN | _% | não | 0 |
| Prototypical Networks | _% | sim | ~110k |
| **STDP + Hopfield (este trabalho)** | **_%** | **não** | **_** |

## Critérios de sucesso (PLAN.md §11)

- [ ] 5-way 1-shot ≥ 90%
- [ ] 20-way 1-shot ≥ 70%
- [ ] Parâmetros < 100k
- [ ] Sem backprop end-to-end

## Próximo passo

[gerado por analysis.py com base no outcome]
