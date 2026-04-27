# WEEKLY-3 — Pipeline completo (Conv-STDP × 2 + Hopfield)

> Layer 2 funcionando em cima da Layer 1. Memória Hopfield operacional.
> Primeira medição de 5-way 1-shot — mesmo se baixa, é o sinal de que
> o sistema fecha.

## Comando

```powershell
python train.py --n-images 24000 --epochs 1
python evaluate.py --checkpoint checkpoints/stdp_model.pt --ways 5 --shots 1 --episodes 100
```

## Resultado preliminar

| Config | Acurácia (100 episódios) |
|---|---:|
| 5-way 1-shot | __% |

## Análise

- [ ] Pipeline fecha sem erro de runtime
- [ ] Acurácia significativamente acima de chance (>30%) em 5w1s
- [ ] Tempo por episódio < 1s na 4070

## Decisão

Se >30% em 5w1s: sistema funciona, partir pra Semana 4 (tuning).
Se ≤30%: investigar embeddings (t-SNE), saturação da memória (β), normalização.
