# WEEKLY-2 — Adaptação para Omniglot

> Pipeline de dados + Conv-STDP layer 1 funcionando em Omniglot.
> Inspeção visual dos filtros aprendidos.

## O que foi feito

- [ ] Pipeline de dados Omniglot (split background/evaluation, codificação spike) testado
- [ ] Conv-STDP layer 1 pretreinada em ~500 imagens (debug rápido)
- [ ] Filtros visualizados em `logs/run_*/figs/filters_layer1.png`

## Inspeção dos filtros

```
[cole imagem ou descreva o que aparece — esperado: Gabor-like ou strokes]
```

Critério qualitativo: filtros NÃO devem parecer ruído. Se aparecem como
arestas, curvas ou strokes (parte de caracteres), STDP está extraindo
padrões locais. Se forem manchas aleatórias, há problema.

## Estatísticas dos pesos

| Métrica | Layer 1 |
|---|---:|
| média | __ |
| std | __ |
| % saturados em w_max | __% |
| % saturados em w_min | __% |

Bimodalidade (alto % perto de w_max E w_min) é assinatura de STDP convergente.

## Decisão

Se filtros são coerentes: liberar Semana 3 (pipeline completo + memória Hopfield).

Se ruído: investigar `cfg.stdp.lateral_inhibition`, número de imagens, `cfg.spike.timesteps`.
