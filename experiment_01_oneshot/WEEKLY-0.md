# WEEKLY-0 — Validação Pré-Semana-1 (Infraestrutura)

> Subfase pré-roadmap mandada pelo `CONTEXT.md` §8.2: rodar pipeline
> Omniglot end-to-end com baselines triviais **antes** de implementar STDP.
> Objetivo: confirmar que a infra (EpisodeSampler, HopfieldMemory,
> evaluate.py, baselines.py) fecha sem bugs com pesos random; descobrir
> problemas de plumbing antes de gastar tempo de pretreino.

## O que foi feito (lado Claude)

- Bugfix em `data.py:EpisodeSampler.__init__`: substituí o loop ineficiente
  que iterava todo o dataset (carregando imagens só pra coletar labels) por
  indexação direta via `_flat_character_images` que torchvision expõe.
  Ganho: startup do EpisodeSampler vai de minutos pra segundos com 24k+
  amostras.
- Validações de pré-condição no EpisodeSampler: erro explícito se
  `n_way > num_classes` ou se alguma classe não tem `k_shot + n_query`
  amostras suficientes.
- Refinamentos em `model.py:ConvSTDPLayer`: marquei `conv.weight` como
  `requires_grad=False` (modelo é sem backprop por design — STDP atualiza
  pesos manualmente em `stdp_update`); removi `register_buffer` mal-usado
  pra traços apre/apost (eles têm shape dependente de input, vão ser
  alocados em `reset_traces`).
- Refinamentos em `model.py:STDPHopfieldModel`: cache de shapes
  intermediárias (evita dummy forward em cada `extract_features`),
  projeção lazy explicitamente sem grad.

## O que precisa ser feito (lado usuário, na 4070)

Sequência mandada pelo `CONTEXT.md` §8.2 — rodar nesta ordem dentro do
ambiente conda `neuro` com a 4070 ativa:

```powershell
cd C:\Users\pinho\Projects\project-hebb\experiment_01_oneshot

# 0. Validar ambiente (pré-requisito)
python ..\validate_environment.py

# 1. Pipeline end-to-end com pesos random — esperado ~chance (20%)
python evaluate.py --ways 5 --shots 1 --episodes 100

# 2. Baseline trivial (pixels crus, kNN) — esperado ~50-70% em 5w1s
python baselines.py --baseline pixel_knn --ways 5 --shots 1 --episodes 1000

# 3. Baseline forte (ProtoNet) — esperado ~98% em 5w1s; demora algumas
#    centenas de iterações de treino
python baselines.py --baseline proto_net --ways 5 --shots 1 \
    --episodes 1000 --train-episodes 5000
```

## Critério de sucesso desta subfase

Não é acurácia — é **plumbing**. A subfase fecha quando:

- [ ] `validate_environment.py` retorna tudo verde (CUDA disponível, GPU
      detectada como RTX 4070, todos os frameworks importáveis)
- [ ] `evaluate.py` com pesos random roda 100 episódios sem erro e
      retorna acurácia próxima de 1/N (~20% em 5-way, indicando que
      memory + sampler funcionam mas o sinal vem só de ruído)
- [ ] `baselines.py --baseline pixel_knn` retorna ~50-70% em 5w1s,
      consistente com a literatura (Lake 2015)
- [ ] `baselines.py --baseline proto_net` retorna >90% em 5w1s,
      consistente com Snell 2017 (~98%)

Se algum desses falhar com erro de código, é bug que **eu** preciso
consertar antes de seguir. Se o número estiver fora do esperado, **nós**
precisamos investigar — pode ser bug, pode ser config inadequada.

## Resultados (a preencher após execução)

> Cole aqui o output dos 4 comandos. Eu interpreto, atualizo este arquivo
> com a análise, e a Semana 1 é liberada.

### `validate_environment.py`

```
[aguardando execução]
```

### `evaluate.py` (pesos random, 5w1s, 100 episódios)

```
[aguardando execução]
```

Acurácia esperada: **~20%** (chance pra 5-way). Se vier ≥30%, há sinal
acidental no embedding random — investigar.

### `baselines.py --baseline pixel_knn` (5w1s, 1000 episódios)

```
[aguardando execução]
```

Acurácia esperada: **~50-70%**. Esse é o **número a bater claramente**
com nosso modelo final.

### `baselines.py --baseline proto_net` (5w1s, 1000 episódios)

```
[aguardando execução]
```

Acurácia esperada: **~95-98%**. Esse é o **teto realista** do que
abordagens deep com backprop alcançam — nosso alvo é chegar a ≥90% sem
backprop end-to-end.

## Decisão de transição

Se as 4 caixas acima ficarem checked, escrevo `WEEKLY-1.md` (Semana 1
oficial: sanity check Diehl & Cook 2015 em MNIST), implemento a regra
STDP em `model.py:ConvSTDPLayer.stdp_update` e o script
`experiment_01_oneshot/sanity_mnist.py`. Comito como
`feat: week 1 — STDP rule + Diehl & Cook MNIST sanity check`.

Se algo falhar, registro em `PLAN.md` (raiz) §"Notas de iteração" e
analisamos juntos antes de seguir.
