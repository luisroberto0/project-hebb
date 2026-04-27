# Experiment 01 — One-shot Omniglot com STDP + Memória Episódica

Primeiro experimento da pesquisa neuromorfa. Veja [PLAN.md](./PLAN.md) pra
motivação completa, hipótese, arquitetura e roadmap.

## Estrutura

```
experiment_01_oneshot/
├── PLAN.md           Plano de pesquisa detalhado (leia primeiro)
├── README.md         Este arquivo
├── config.py         Hiperparâmetros centralizados
├── data.py           Omniglot loader + codificação spike
├── model.py          Conv-STDP + memória Hopfield Moderna
├── train.py          Pretreino STDP não-supervisionado
├── evaluate.py       Avaliação N-way K-shot
└── baselines.py      Pixel kNN e Prototypical Networks
```

## Setup rápido

Já dentro do ambiente conda `neuro` configurado em `setup_neuromorfa.md`:

```bash
cd experiment_01_oneshot
mkdir -p data checkpoints logs
```

## Fluxo de uso

### 1. Sanity check sem pretreino (random weights baseline)

```bash
python evaluate.py --ways 5 --shots 1 --episodes 100
```

Acurácia esperada: pouco acima de 20% (chance). Confirma que o pipeline
de avaliação roda end-to-end.

### 2. Baselines de comparação

```bash
# Pixel kNN — baseline-zero, ~70% em 5w1s
python baselines.py --baseline pixel_knn --ways 5 --shots 1 --episodes 1000

# Prototypical Networks — baseline forte, ~98% em 5w1s
python baselines.py --baseline proto_net --ways 5 --shots 1 --episodes 1000
```

### 3. Pretreino STDP

```bash
# Debug rápido: 500 imagens, 1 epoch
python train.py --n-images 500

# Treino completo
python train.py
```

### 4. Avaliação após pretreino

```bash
python evaluate.py --checkpoint checkpoints/stdp_model.pt --ways 5 --shots 1 --episodes 1000
python evaluate.py --checkpoint checkpoints/stdp_model.pt --ways 20 --shots 1 --episodes 1000
```

## Estado do código

Esse repositório é **scaffolding inicial** — pipeline completo end-to-end,
mas com a regra STDP em `model.py:ConvSTDPLayer.stdp_update` ainda como
placeholder. A primeira tarefa real (Semana 1 do roadmap em PLAN.md) é
implementar essa regra de forma vetorizada e validar reproduzindo
Diehl & Cook 2015 em MNIST.

Por que entregar com placeholder em vez de implementação completa: STDP
convolucional vetorizado tem várias formulações na literatura
(Kheradpisheh 2018, Falez 2019, Mozafari 2018) com trade-offs distintos.
Você vai querer escolher uma e iterar — não receber uma decisão pronta
que talvez não queira.

## Próximos checkpoints

- [ ] Implementar `ConvSTDPLayer.stdp_update` vetorizado
- [ ] Validar STDP em MNIST (~85% com leitor linear)
- [ ] Visualizar filtros aprendidos (devem parecer Gabor/strokes)
- [ ] Rodar `train.py` no Omniglot completo
- [ ] Avaliar 5w1s, 20w1s, 5w5s, 20w5s
- [ ] Análise de esparsidade e robustez
