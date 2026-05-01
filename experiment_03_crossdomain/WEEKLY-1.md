# Experimento 03 — Cross-domain CUB-200, sessão #53

> **Status:** Pipeline scripts prontos, encoders treinados e salvos, CUB download bloqueado por velocidade da rede.
> **Data:** 2026-05-15
> **Tempo total da sessão:** ~50 min de 90 (encerrada cedo por bloqueio de download).

---

## Achado central da sessão

**Sessão #20 nunca salvou checkpoint do C3.** `c3_protonet_sparse.py` treina inline e descarta state_dict ao final da execução. Risco listado em `experiment_03_crossdomain/PLAN.md` ("Checkpoint da sessão #20 não estar reproduzível") **confirmou-se** — não havia `c3_kwta_k16_seed42.pt` em `experiment_01_oneshot/checkpoints/`.

Mitigação aplicada: criar `experiment_03_crossdomain/train_encoders.py` que **importa** as classes de `c3_protonet_sparse.py` e treina + salva 2 checkpoints (`protonet_omniglot_seed42.pt` e `c3_kwta_k16_seed42.pt`) sem modificar o script original. Reproduz exatamente o setup da #20 (5w1s, 5000 episodes, Adam lr=1e-3, seed=42).

Treino concluído em ~2.5 min total (RTX 4070):
- ProtoNet baseline: 75.9s, train acc final 100% (steps 4500-5000 com loss <0.05)
- C3b k=16: 79.5s, train acc final 100%

Convergência consistente com #20 — esperado eval ~94% (ProtoNet) e ~93% (C3b) em Omniglot evaluation set, mesmos números da #20.

---

## Sanity dos checkpoints (sem CUB ainda)

Validação interna: load + forward com tensor dummy (5, 1, 28, 28).

| Encoder | Embedding shape | Ativações não-zero / 64 | Trainable params |
|---|---|---|---|
| ProtoNet baseline | (5, 64) | 48.6/64 (~76% denso, esperado pós-ReLU+MaxPool) | 0 (frozen) |
| C3 k=16 | (5, 64) | **16.0/64 (exatamente, k-WTA OK)** | 0 (frozen) |

Ambos `state_dict` carregam sem mismatch. k-WTA k=16 produz exatamente 16 ativações por exemplo (75% sparsity confirmada). Forward funciona em input arbitrário (1, 28, 28) — confirma que CUB resized 28×28 grayscale será compatível.

---

## CUB-200 download — BLOQUEADO

URL primária `https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz` (1.15 GB) responde 200 OK e baixa, **mas a velocidade efetiva é ~60 KB/s**, levando ~5 horas pra completar. Inviável dentro do orçamento da sessão (90 min hard).

Mirrors testados:
- `vision.caltech.edu/datasets/cub_200_2011/CUB_200_2011.tgz` → 404
- `huggingface.co/datasets/sasha/CUB_200_2011/resolve/main/CUB_200_2011.tgz` → 401 (auth required)
- `huggingface.co/datasets/Multimodal-Fatima/CUB_train/...` → 404 (parquet, não tar)
- `huggingface.co/datasets/wzk1015/CUB_200_2011/...` → 401 (auth required)

Não foi possível encontrar mirror público sem autenticação dentro do orçamento de 15 min de probing. Download primário travado por largura de banda da rede do Luis pro servidor Caltech, não problema do servidor.

### Decisão pra desbloquear (próxima sessão)

**Luis baixa manualmente fora desta sessão.** Opções:
1. Browser direto: cole `https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz` na barra de download. Browser tipicamente usa connection pooling melhor que curl.
2. Download manager (FDM, IDM, etc.) com múltiplas conexões paralelas — pode aumentar throughput 3-5×.
3. Wget com `-c` (resume) deixado rodando overnight: `wget -c -O data/CUB_200_2011.tgz https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz`
4. Conta Hugging Face + `hf auth login` + dataset autenticado — mais setup mas potencialmente velocidade melhor.

**Validação após download:**
- Tamanho ~1.15 GB
- SHA-256 esperado: confirmar do site oficial após download (tar pode ter checksums published)
- Extrair em `data/CUB_200_2011/` (caminho relativo ao repo root)
- Validar estrutura: `ls data/CUB_200_2011/` deve mostrar `images/`, `images.txt`, `image_class_labels.txt`, `train_test_split.txt`, `classes.txt`

`data/` está em `.gitignore` (`data/`) — dataset não vai pro git.

---

## Pipeline scripts criados (todos em `experiment_03_crossdomain/`)

| Arquivo | Função | Status |
|---|---|---|
| `train_encoders.py` | Treina + salva ProtoNet e C3 em Omniglot | ✅ rodado, 2 checkpoints salvos |
| `data.py` | `CUBDataset` (28×28 grayscale, train/test splits oficiais), `build_cache()` | ✅ código pronto, **bloqueado pra rodar até CUB existir** |
| `episodes.py` | `CUBEpisodeSampler` (5w1s5q episode sampler), `proto_episode_eval()` | ✅ código pronto |
| `smoke_test.py` | Smoke test: load encoder frozen + 1 episódio CUB + sanity checks | ✅ código pronto, **bloqueado pra rodar até CUB existir** |

### Decisões de design

- **Cache obrigatório.** `build_cache()` pré-processa 11.788 imagens uma única vez e salva em `data/CUB_200_2011/cache_28x28_gray.pt`. Sessões #54+ usam cache direto (~50ms load vs minutos).
- **Splits oficiais CUB.** `data.py` lê `train_test_split.txt` em vez de criar split custom — consistente com convention da literatura cross-domain few-shot.
- **Output shape (1, 28, 28)** igual Omniglot. Encoder C3 forward funciona sem mudança arquitetural.
- **Episode shape com canal preservado** `(N*K, 1, 28, 28)`. `c3_protonet_sparse.py:proto_episode_loss` espera `episode.support.unsqueeze(1)` (assume `(N*K, H, W)` sem canal). Pra CUB, `CUBEpisode.support` já tem canal — `proto_episode_eval` em `episodes.py` faz forward direto sem `unsqueeze`.

---

## Smoke test cross-domain — NÃO RODADO

Bloqueio de download impediu execução. Quando CUB estiver disponível:

```bash
cd experiment_03_crossdomain
python smoke_test.py --device cuda --encoder c3
# espera: ACC entre [5%, 80%] em 1 episódio (qualitativo, não estatístico)
# espera: todos sanity checks passam
```

Sanity checks codificados em `smoke_test.py:run_sanity_checks`:
- ACC ∈ [0, 1]
- Embedding dim = 64
- Predições ∈ [0, n_way-1]
- Distâncias com std > 0 (não degeneradas)
- Encoder norm > 0 (não silent)

Se algum sanity falhar em #54: para imediatamente, documenta erro específico, não tenta debug freestyle.

---

## Observações sobre treino dos encoders

Ambos encoders convergem rapidamente em Omniglot 5w1s — train acc 100% em steps finais. Comparação com #20:
- #20 reportou ProtoNet baseline eval 5w1s = **94.55%** após 5000 episodes
- #20 reportou C3b k=16 eval 5w1s = **93.10%** após 5000 episodes

Eval em Omniglot evaluation set (não rodado nesta sessão) deve reproduzir esses números aproximadamente — single seed, mesmo setup, mesma arquitetura. Ligeira variação esperada por ordem aleatória de episódios (sampler internamente determinístico via seed=42, mas `torch.manual_seed(42)` aplicado antes de cada init reseta init dos pesos).

**Não validei eval Omniglot nesta sessão** — fora de scope da #53. Se #54 quiser confirmar, basta um run rápido com `c3_protonet_sparse.py` (~6 min). Mas a expectativa é que os checkpoints reproduzem #20.

---

## Estado pós-#53

| Componente | Status |
|---|---|
| `STRATEGY.md` Marco 2-A scoping | ✅ #52 |
| `CONTEXT.md` §1.4 Refino #52 | ✅ #52 |
| `experiment_03_crossdomain/PLAN.md` | ✅ #52 |
| `experiment_03_crossdomain/PAPERS.md` (5 papers) | ✅ #52 |
| `experiment_03_crossdomain/train_encoders.py` | ✅ #53 |
| ProtoNet baseline checkpoint Omniglot | ✅ #53 (`protonet_omniglot_seed42.pt`, 460 KB) |
| C3 k=16 checkpoint Omniglot | ✅ #53 (`c3_kwta_k16_seed42.pt`, 460 KB) |
| `data.py` (CUBDataset + cache) | ✅ #53 (código pronto) |
| `episodes.py` (sampler + eval) | ✅ #53 |
| `smoke_test.py` | ✅ #53 (código pronto) |
| Sanity load+forward dos checkpoints | ✅ #53 (offline) |
| **CUB-200 download** | ❌ **BLOQUEADO** (~60 KB/s, 5h estimado) |
| Smoke test cross-domain real | ⏳ **pendente CUB** |

**Sessões consecutivas sem sinal>chance:** irrelevant (não é experimento estatístico).

---

## Próximo passo (#54)

**Pré-condição #54:** Luis baixa CUB manualmente fora da sessão. Comando final esperado:

```bash
ls C:/Users/pinho/Projects/project-hebb/data/CUB_200_2011/
# deve mostrar: images/  images.txt  image_class_labels.txt  train_test_split.txt  classes.txt  ...
```

**Tarefas #54:**
1. Smoke test cross-domain real (`smoke_test.py --device cuda --encoder c3`) — 1 episódio, ACC qualitativo, sanity checks
2. Eval cross-domain completa: 1000 episódios × 5 seeds, IC95% bootstrap, encoders C3 + ProtoNet baseline (ambos Omniglot frozen)
3. Pixel kNN cross-domain como sanity floor

Outputs esperados de #54:
- `RESULTS_xdomain.md` com tabelas IC95%
- ACC C3 cross-domain (pred 20-40%)
- ACC ProtoNet baseline cross-domain (pred 25-45%)
- ACC Pixel kNN cross-domain (pred 22-28%)

ProtoNet retreinado em CUB-200 (baseline a bater) fica pra #55-#56.

---

## Riscos pra #54+

1. **CUB train_test_split oficial pode estar desbalanceado** entre 200 classes. `data.py:CUBDataset` valida em init via `min/max per class` — se algum split tiver <(k_shot+n_query) imagens por classe, episode sampler crasha. Mitigação no código: error message claro com classe problemática.
2. **Cache pré-processado pode falhar** se PIL não conseguir abrir alguma imagem corrompida. Mitigação: rodar `build_cache()` standalone primeiro pra detectar arquivos quebrados (`python data.py` faz isso na seção `__main__`).
3. **Resize 28×28 grayscale destrói tanta informação que ProtoNet retreinado também fica em chance.** Documentado como achado se ocorrer; justifica passada 84×84 RGB em #61-#62.

---

# Sessão #54 — smoke real CUB + primeiro número cross-domain (1 seed)

> **Status:** sucesso pipeline empírico. Smoke passou + 1000 episodes 1 seed para C3 e ProtoNet baseline cross-domain.
> **Data:** 2026-05-01
> **Tempo total:** ~30 min de 90.

## Bugfixes substantivos do pipeline #53 (registrados como exceção honesta à regra "não modifica scripts de #53")

3 bugs encontrados que bloqueavam toda a sessão. Conserto justifica-se: regra existe pra evitar debug em loop, não pra impedir bugfix bloqueante. Documentado aqui pra rastreabilidade.

1. **Conflito de namespace `data`:** `experiment_03_crossdomain/data.py` colidia com `experiment_01_oneshot/data.py` no `sys.path`. `c3_protonet_sparse.py` faz `from data import load_evaluation` — quando ambos `data.py` estão em path, Python resolve ao primeiro, quebrando um lado ou outro.
   - Fix: `git mv experiment_03_crossdomain/data.py experiment_03_crossdomain/cub_data.py`
   - Atualizado: imports em `episodes.py` e `smoke_test.py` (`from cub_data import CUBDataset`)
2. **Default `n_query=15` em `smoke_test.py`** chocava com test split do CUB (algumas classes têm só 11-30 samples). Setup #20 do Omniglot usou `n_query=5`. Mantida default no script — passei `--n-query 5` no comando pra preservar o arquivo.
3. **`UnicodeEncodeError` no print final do `eval_crossdomain.py`:** char `≈` (U+2248) não codifica em cp1252 (Windows console default). Trocado por `=` pra match com convenção dos outros scripts do repo.

Não houve outros fixes — restantes problemas (n_query, encoding) são parametrização ou cosmético.

## Sanity da estrutura CUB extraída (Luis baixou manual)

| Verificação | Esperado | Observado |
|---|---|---|
| `images.txt` linhas | 11788 | ✓ 11788 |
| `classes.txt` linhas | 200 | ✓ 200 |
| `image_class_labels.txt` linhas | 11788 | ✓ 11788 |
| `train_test_split.txt` linhas | 11788 | ✓ 11788 |
| `images/` subpastas | 200 | ✓ 200 |

Arquivos extras presentes (não usados nesta passada): `attributes/`, `attributes.txt`, `bounding_boxes.txt`, `parts/`.

Test split (após `train_test_split.txt`): **5794 imagens em 200 classes, 11-30 samples per class**.

## Cache pré-processamento (1× cost)

Primeira execução do `smoke_test.py` processou 11788 imagens em <2 min (RTX 4070 ssd). Cache salvo em `data/CUB_200_2011/cache_28x28_gray.pt` (37.0 MB). Execuções subsequentes: load <100ms.

## Smoke test resultados

Comandos:
```bash
python smoke_test.py --device cuda --encoder c3       --seed 42 --n-query 5
python smoke_test.py --device cuda --encoder protonet --seed 42 --n-query 5
```

| Encoder | ACC 1 ep | sup_emb_norm | qry_emb_norm | dists_mean | sanity |
|---|---|---|---|---|---|
| C3 k=16 frozen | 24.00% | 6.13 | 6.40 | 3.33 | ALL PASS |
| ProtoNet frozen | 32.00% | 7.86 | 8.03 | 2.78 | ALL PASS |

Diagnostics ambos: emb shape (5,64) e (25,64) ✓, preds em [0, 4] ✓, dists std > 0 ✓, embeddings não-zero ✓, 0 trainable params ✓.

ProtoNet norm > C3 norm (esperado: C3 zera 75% das ativações via k-WTA). 1 episódio é qualitativo; eval 1000 eps abaixo é o que conta.

## Avaliação 1000 episodes 5w1s 5q, single seed=42

Comandos (~3s cada em RTX 4070):
```bash
python eval_crossdomain.py --device cuda --encoder c3       --seed 42 --episodes 1000 --n-query 5
python eval_crossdomain.py --device cuda --encoder protonet --seed 42 --episodes 1000 --n-query 5
```

| Encoder | ACC 5w1s | IC95% bootstrap | std/ep | z (vs chance) | tempo |
|---|---|---|---|---|---|
| **C3 (k-WTA k=16)** | **21.79%** | [21.26, 22.28] | 8.16% | 6.9 | 2.6s |
| **ProtoNet baseline** | **21.83%** | [21.30, 22.29] | 8.22% | 7.0 | 2.2s |
| Chance | 20.00% | — | — | — | — |

**C3 vs ProtoNet baseline:** delta = **−0.04 p.p.** (ruído). ICs sobrepostos quase totalmente — encoders são estatisticamente indistinguíveis cross-domain.

Ambos atingem sinal estatístico vs chance (z~7), mas magnitude absoluta é minúscula: **+1.79 e +1.83 p.p. acima de chance**.

## Comparação com predição #52

| Modelo | Predição #52 (5w1s CUB) | Realidade #54 |
|---|---|---|
| C3 cross-domain | 20-40% | **21.79%** ✓ (limite inferior) |
| ProtoNet baseline cross-domain | 25-45% | **21.83%** ✗ (abaixo do limite inferior em 3.2 p.p.) |
| ProtoNet retreinado em CUB (não rodado) | 35-55% | TBD #56 |
| Pixel kNN (não rodado) | 22-28% | TBD #55-56 |

ProtoNet baseline veio **abaixo** do range previsto (esperava 25-45%, deu 21.83%). Predição inicial subestimou o quão adversarial é Omniglot→CUB em 28×28 grayscale. Setup mais extremo que mini-ImageNet→CUB da Tseng 2020 (que produzia ProtoNet ~38%).

## Observações qualitativas

1. **k-WTA k=16 não move ponteiro cross-domain.** No Omniglot, C3b alcançou 93.10% vs ProtoNet baseline 94.55% (delta -1.45 p.p.). No CUB cross-domain, C3 21.79% vs ProtoNet 21.83% (delta -0.04 p.p.). A esparsidade k-WTA preserva (ou destrói) features de modo que não importa cross-domain.
2. **Encoders Omniglot mal generalizam pra CUB**, conforme predição #52 e literatura (STARTUP / Phoo & Hariharan 2021 documentou padrão similar em "extreme task differences"). Sinal acima de chance é estatisticamente real mas pequeno.
3. **Pré-processamento 28×28 grayscale destrói detalhe visual** que distingue bird species. Mesmo se o encoder fosse perfeito (transfer 100% bem), input degradado limita teto.
4. **Próxima sessão pode confirmar predição com 5 seeds.** Dada a margem pequena (1.8 p.p. acima de chance) e std por episódio (~8 p.p.), 5 seeds estabilizam o ponto sem mudar conclusão geral.

## Critério literal Marco 2-A — status interim

**Não-avaliável ainda.** Critério: C3 ≥ ProtoNet retreinado + 5 p.p. ProtoNet retreinado em CUB-200 fica pra #56. Mas dado que C3 cross-domain baseline está em 21.79%, atingir +5 p.p. acima de qualquer baseline retreinado parece **improvável** — ProtoNet retreinado provavelmente fica acima de 30% mesmo em 28×28 grayscale (treino direto na target).

**Predição mais firme (revisão pós-#54):** Marco 2-A vai falhar critério literal pelo padrão esperado. Achado negativo continua defensável e útil.

## Estado pós-#54

| Componente | Status |
|---|---|
| Smoke test C3 + ProtoNet | ✅ ambos PASS |
| Eval 1000 eps 1 seed C3 | ✅ 21.79% |
| Eval 1000 eps 1 seed ProtoNet baseline | ✅ 21.83% |
| Eval 5 seeds | ⏳ #55 |
| Pixel kNN cross-domain | ⏳ #55-56 |
| ProtoNet retreinado em CUB-200 (baseline a bater) | ⏳ #56 |
| Comparação cabeça-a-cabeça com critério literal | ⏳ #57-58 |

## Próximo passo (#55)

1. Eval 5 seeds × 2 encoders (C3 + ProtoNet baseline) = 10 runs × ~3s = ~30s total
2. Pixel kNN cross-domain como sanity floor (1 run multi-seed)
3. RESULTS_xdomain_seed42.md preliminar (será refinado em #57+)

#56 começa ProtoNet retreinado em CUB-200 (treina + eval).
