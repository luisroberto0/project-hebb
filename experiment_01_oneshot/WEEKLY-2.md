# Semana 2 — Adaptação pra Omniglot (conv real)

**Iniciada:** 2026-04-27 (sessão #7)
**Pré-condições:** Semana 1 fechada como caso patológico (`WEEKLY-1.md`), infra Omniglot validada via inspeção (`WEEKLY-2-NEXT.md`).

---

## Sessão #7 (smoke test + calibração)

### Etapa 0 — Smoke test do pipeline Omniglot ✅

**Objetivo:** confirmar que o pipeline executa end-to-end na arquitetura conv real (kernel=5 + pool, 2 layers).

**Configuração quick:** 500 imgs / 1 epoch / pretreino + baselines + eval.

**Resultados:**

| Componente | Status | Saída |
|------------|--------|-------|
| `validate_environment.py` | ✅ | PyTorch 2.6.0+cu124, RTX 4070 (8.9), 2933 GFLOPS, todas libs core OK |
| `evaluate.py` (random, no ckpt) | ✅ | 20.92% (chance ≈20%, z≈0.2) — pipeline fecha |
| `baselines.py pixel_knn` | ✅ | 45.76% ± 10.47% — pixel kNN funciona |
| `baselines.py proto_net` (500 train eps) | ✅ | **85.88% ± 10.95%** — SOTA baseline forte estabelecido |
| `train.py` (500 imgs, 1 epoch) | ⚠️ | Executou (17.3s, 28.9 imgs/s) MAS w1 colapsou (μ=0.000) |
| `evaluate.py` (com ckpt colapsado) | ✅ pipeline / ❌ resultado | 20.00% exato (predição constante, IC=[20,20], z=inf) |

**Conclusão Etapa 0:** Pipeline integra perfeitamente. Bug crítico de pretreino exposto: layer 1 colapsa em segundos.

---

### Bug fix: lambda não-picklável em `data.py`

`build_transforms` usava `transforms.Lambda(lambda x: 1.0 - x)` que não é picklável no Windows multiprocessing (spawn). DataLoader com `num_workers=2` falhava com `AttributeError: Can't get local object 'build_transforms.<locals>.<lambda>'`.

**Fix:** substituído por função module-level `_invert_intensity`. Mudança mínima, semântica idêntica.

---

### Bug ativo (a resolver na Etapa 1): layer 1 colapsa no pretreino

Pretreino de 500 imgs, 1 epoch:
```
epoch 0 step    0  w1=μ0.000/σ0.000  w2=μ0.609/σ0.417  seen=8  elapsed=4.6s
```

Layer 1 weights vão pra zero já no primeiro batch (8 imagens). Layer 2 cresce para μ=0.609 — mas alimentada por entrada zero, não significa nada (pesos saindo da inicialização interagindo com noise).

Eval com este checkpoint: 20.00% exato com IC zero — confirma que o modelo virou função constante (zero spikes propagam zero ativação).

**Hipótese mecânica (a confirmar na Etapa 1):**
Em conv real, regime de spikes é radicalmente diferente do MNIST:
- **MNIST kernel=28:** 1 winner global por timestep × 100 ts = 100 pós-spikes/imagem
- **Omniglot kernel=5, padding=2:** 1 winner por POSIÇÃO espacial × 28×28 posições × 100 ts = 78400 pós-spikes/imagem total

Pré-spikes da entrada Omniglot (após inversão, fundo preto/traços brancos): ~5-10% dos 784 pixels com intensidade alta → ~50-100 pré-spikes/timestep × 100 ts = 5000-10000 pré-spikes/imagem.

**Razão estimada R = pré:pós ≈ 5000/78400 = 0.06** (vs. 10.1 em MNIST kernel=28 e ~1 no paper Diehl & Cook).

Com `A_pre=0.01, A_post=-0.0105` (paper original) e essa razão invertida, intuição diz que LTD deveria ficar fraco, mas o produto `apost · pre_patches` no STDP rule depende não-linearmente das densidades. Análise teórica não basta, precisa medir.

**Próximo passo (Etapa 1):** adaptar `tests/test_spike_balance.py` pra `STDPHopfieldModel.layer1`, medir empiricamente, decidir calibração.
