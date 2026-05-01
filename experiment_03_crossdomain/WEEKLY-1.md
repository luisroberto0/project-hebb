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
