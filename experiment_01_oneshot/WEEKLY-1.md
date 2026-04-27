# WEEKLY-1 — Sanity Check (Diehl & Cook 2015 em MNIST)

> Reprodução simplificada do paper-base do nosso pipeline: STDP
> não-supervisionado + label assignment pós-hoc + voto majoritário.
> Critério de sucesso (PLAN.md §Semana 1): ≥ 85% em test set MNIST.

## Comando

```powershell
python sanity_mnist.py --n-images 10000 --epochs 1
# ou treino completo:
python sanity_mnist.py --n-images 60000 --epochs 3 --n-filters 400
```

## Resultados (a preencher após execução)

```
[cole aqui o output completo do sanity_mnist.py]
```

| Config | n_filters | n_images | epochs | Acurácia teste |
|---|---:|---:|---:|---:|
| Default | 100 | 5000 | 1 | __% |
| Estendido | 400 | 60000 | 3 | __% |

## Análise

- [ ] Acurácia ≥ 85% (critério de sucesso)
- [ ] Distribuição de labels nos filtros é razoavelmente balanceada (não há classe sem filtro)
- [ ] Pesos saturam bimodalmente (assinatura de STDP convergente)
- [ ] Tempo de treinamento aceitável (< 10 min na 4070 com config default)

## Decisão

Se ✅ ≥85%: STDP funciona end-to-end. Liberar Semanas 2-3 (adaptação Omniglot).

Se ⚠️ 70-85%: STDP aprende mas abaixo do paper. Investigar hiperparâmetros antes de seguir:
- aumentar `--n-filters` (Diehl & Cook usaram até 6400)
- aumentar `--timesteps` (default 100, paper usa 350)
- ajustar `cfg.stdp.tau_pre_ms` / `tau_post_ms`
- ajustar `cfg.stdp.A_pre` / `A_post` (proporção LTP/LTD)

Se ❌ <70%: bug na regra STDP, na codificação, ou no label assignment. Documentar em `POSTMORTEM.md`. Bloquear Semanas 2-6 até resolver.
