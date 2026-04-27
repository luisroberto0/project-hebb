# WEEKLY-4 — Tuning de hiperparâmetros + baselines

> Comparação contra Pixel kNN e Prototypical Networks.
> Grid search pequeno em hiperparâmetros chave do STDP.

## Baselines confirmados

| Baseline | 5-way 1-shot | Configuração |
|---|---:|---|
| Random | 20.00% | n/a |
| Pixel kNN | __% | euclidean, k=1 |
| Prototypical Networks | __% | CNN-4, 5000 train episodes |

## Grid search

Variações testadas (ver `logs/run_*/grid.json`):

| n_filters L1 | n_filters L2 | tau_pre_ms | A_pre | embedding_dim | Acc 5w1s |
|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 20 | 0.01 | 64 | __% |
| 16 | 32 | 20 | 0.01 | 64 | __% |
| 8 | 16 | 30 | 0.01 | 64 | __% |
| 8 | 16 | 20 | 0.005 | 64 | __% |
| 8 | 16 | 20 | 0.01 | 128 | __% |

Melhor config: __

## Decisão

Travar config vencedora em `config.py` e seguir pra Semana 5 (avaliação final).
