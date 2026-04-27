# WEEKLY-5 — Avaliação completa nas 4 configs

> 1000 episódios em cada config (5w1s, 5w5s, 20w1s, 20w5s) com
> intervalos de confiança bootstrap. Análise estatística + diagnósticos.

## Resultados finais

(gerados automaticamente por `analysis.py` → ver `RESULTS.md` na raiz do experimento)

| Config | Acurácia | IC 95% | Tempo (s) |
|---|---:|:---:|---:|
| 5-way 1-shot | __% | [_, _] | _ |
| 5-way 5-shot | __% | [_, _] | _ |
| 20-way 1-shot | __% | [_, _] | _ |
| 20-way 5-shot | __% | [_, _] | _ |

## Diagnósticos

- t-SNE dos embeddings em `logs/run_*/figs/tsne.png`
- Histograma de pesos em `logs/run_*/figs/weight_hist.png`
- Curvas TensorBoard em `logs/run_*/tb/`

## Critério de sucesso (PLAN.md §11)

- [ ] 5-way 1-shot ≥ 90%
- [ ] 20-way 1-shot ≥ 70%
- [ ] Filtros visualmente coerentes
- [ ] Esparsidade ≥ 90% (% neurônios silenciosos médio)

## Decisão

Conforme outcome detectado em `RESULTS.md`:
- **Sucesso forte:** começar Semana 6 (escrita de paper rascunho)
- **Sucesso parcial:** preencher `NEXT.md`
- **Falha:** preencher `POSTMORTEM.md`
