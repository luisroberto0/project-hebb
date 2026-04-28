# WEEKLY-2-NEXT — Preparação para Semana 2 (Omniglot conv real)

> Documento de transição: o que está pronto, o que precisa atenção,
> e ordem sugerida de execução. Criado em 2026-04-27 (sessão #6),
> imediatamente após o pivot da Semana 1.

---

## Status da infraestrutura

### ✅ Pronto e validado em sessões anteriores

- **`data.py`**: `load_background()` + `load_evaluation()` (Omniglot via torchvision), `poisson_encode()` e `temporal_encode()`, `EpisodeSampler` para N-way K-shot. Indexação eficiente via `_flat_character_images` (evita carregar imagens só pra coletar labels).
- **`model.py:STDPHopfieldModel`**: pipeline completo `image → conv1+pool → conv2+pool → flatten → projeção ortogonal → memória Hopfield`. Usa as duas instâncias de `ConvSTDPLayer` (com k=1 WTA + homeostasis já implementados na Semana 1).
- **`model.py:HopfieldMemory`**: `store(embeddings, labels, n_classes)` + `query(embeddings)` com cossine ou euclidean, β configurável.
- **`train.py`**: loop de pretreino com TensorBoard logging, salvamento periódico de checkpoint, suporte a `--n-images` e `--epochs` por CLI.
- **`evaluate.py`**: avaliação N-way K-shot com 1000 episódios + IC95% bootstrap.
- **`baselines.py`**: Pixel kNN e Prototypical Networks (158 linhas, validado em pré-Semana-1 do `run_all.ps1`).
- **`run_all.ps1`**: pipeline automatizado das 6 semanas (já executado parcialmente).

### ⚠️ Pontos de atenção identificados nesta auditoria

#### 1. Hiperparâmetros de homeostasis foram calibrados pra MNIST kernel=28

Estado atual em `config.py`:
```
theta_plus = 0.0005   # 100× menor que paper, calibrado pra ~100 ts × k=1 WTA com 1 winner global
tau_theta_ms = 1e7    # paper original
A_pre = 0.01          # paper original
A_post = -0.0105      # paper original
```

**Em Omniglot conv real, o regime de spikes muda fundamentalmente:**
- MNIST (kernel=28): 1 winner global por timestep → ~100 spikes/imagem total
- Omniglot (kernel=5, padding=2): 1 winner por POSIÇÃO espacial → ~28×28 = 784 winners por timestep → ~78400 spikes/imagem total (muito mais)

Razão pré:pós-spikes vai cair drasticamente (talvez R≈1, igual no paper), o que pode tornar:
- `theta_plus=0.0005` excessivamente fraco (homeostasis sub-ativa)
- `A_post=-0.0105` adequado de novo (paper original calibrado pra esse regime)

**Ação recomendada:** rodar `tests/test_spike_balance.py` adaptado para `STDPHopfieldModel.layer1` antes de treinar a sério. Re-medir R no novo regime, ajustar `theta_plus` proporcionalmente. Custo: ~10 min.

#### 2. Número de filtros (8 / 16) pode ser baixo

`config.py:ArchitectureConfig`:
```
conv1_filters = 8
conv2_filters = 16
```

Diehl & Cook 2015 usaram 400-6400 unidades excitatórias (em FC). Em arquitetura conv com pool, o número total de unidades é `n_filters × output_spatial_size`, então 8 filtros × 28×28 = 6272 unidades efetivas na layer 1 — coerente com a escala do paper. Mas pode valer experimentar 16/32 se filtros não diversificarem.

**Ação recomendada:** ficar com 8/16 no primeiro experimento (default já existe). Se filtros não diversificarem visualmente (ver `utils/visualize.py`), aumentar.

#### 3. `STDPHopfieldModel._proj` é criado on-demand e pode não entrar no state_dict

Em `model.py:223-229`, `_ensure_projection` cria um `nn.Linear` na primeira chamada de `extract_features`. Como é atribuído via `self._proj` (não via `register_module` ou no `__init__`), pode não aparecer no `state_dict()`.

**Risco:** ao salvar e recarregar checkpoint, projeção aleatória pode não ser restaurada → embeddings entre runs ficam inconsistentes.

**Ação recomendada (~15 min):** no primeiro teste, salvar checkpoint, recarregar, comparar embeddings de uma imagem fixa antes/depois. Se diferentes, mover criação de `_proj` pra `__init__` (precisa pré-computar `flat_dim` baseado no `image_size`) OU registrar manualmente no state_dict.

#### 4. Smoke test antes de treino completo

`run_all.ps1` já tem suporte a `$env:HEBB_QUICK="1"` que reduz `n_pretrain_images` pra 500. Recomendado rodar isso ANTES de qualquer experimento sério, pra confirmar:
- Pipeline executa sem crash
- TensorBoard escreve dados
- Checkpoint salva e recarrega
- `evaluate.py` produz números (qualquer que seja a acurácia)

**Comando sugerido (custo ~5-10 min na 4070):**
```powershell
$env:HEBB_QUICK="1"
pwsh -File experiment_01_oneshot/run_all.ps1
```

---

## Ordem sugerida pra Semana 2

### Etapa 0 — Smoke test (15 min)

1. Rodar `run_all.ps1` em modo `HEBB_QUICK=1`
2. Verificar que cada fase completa sem erro
3. Confirmar que `RESULTS.md` é gerado (mesmo com números ruins)

**Critério de sucesso:** todas as fases do pipeline executam end-to-end. Acurácia esperada com pretreino mínimo (500 imgs): ~chance (20% em 5w1s) — o que importa aqui é integridade, não performance.

### Etapa 1 — Calibrar regime de spikes (10 min)

1. Adaptar `tests/test_spike_balance.py` pra `STDPHopfieldModel.layer1` (usa Omniglot, não MNIST)
2. Medir nova razão R = pré:pós em 100 imagens
3. Decidir se `theta_plus` precisa recalibração (provavelmente subir de 0.0005 pra ~0.005-0.05)

### Etapa 2 — Pretreino STDP completo (1-2h GPU)

1. `python train.py --n-images 24000 --epochs 1` (ou começar com 5000 se quiser checkpoint intermediário)
2. Monitorar via TensorBoard: pesos w1/w2 não devem morrer nem saturar
3. Visualizar filtros aprendidos com `utils/visualize.py`
4. **Critério de sanidade:** filtros devem ter aparência visual interpretável (formas tipo Gabor, traços, partes de caracteres). Se forem ruído puro, voltar pra Etapa 1.

### Etapa 3 — Primeira medição N-way K-shot (5 min)

1. `python evaluate.py --checkpoint checkpoints/stdp_model.pt --ways 5 --shots 1 --episodes 100`
2. **Critério mínimo:** acurácia > 30% (chance é 20%). Esperado: 40-60% se features STDP têm sinal discriminativo.
3. Se ≥ 40%, rodar 1000 episódios pra IC válido, depois ir pra 20w1s.
4. Se < 30%, problema é features. Voltar pra Etapa 2 com hiperparâmetros diferentes.

### Etapa 4 — Documentação (30 min)

1. Criar `WEEKLY-2.md` com tabela comparativa: chance, Pixel kNN, ProtoNet, STDP+Hopfield
2. Atualizar `experiment_01_oneshot/PLAN.md` § Roadmap (Semana 2 → concluída)
3. Decisão sobre Semana 3: tuning vs medição expandida

---

## Hipóteses pré-Semana-2 (a confirmar/refutar empiricamente)

### H2_arch (otimista): Output multi-posição resolve degenerescência de k-WTA

**Esperado:** acurácia 5w1s ≥ 40% no primeiro experimento sem mais tuning.
**Se confirmada:** Semana 2 sucesso, segue cronograma.

### H2_homeostasis (médio): theta_plus precisa recalibração pra novo regime

**Esperado:** primeiro experimento ~chance porque homeostasis sub-ativa (theta nunca cresce). Recalibrar via medição empírica → segundo experimento dá > 40%.
**Se confirmada:** custo é só 1 ciclo extra de calibração.

### H2_paper (pessimista): Mesmo com arquitetura conv real, dinâmica STDP+Hopfield não bate baselines

**Esperado:** acurácia 5w1s < 30% mesmo após calibração de Etapa 1.
**Se confirmada:** próximo passo é H_paper_replicability (Brian2) — caminho honesto que estava reservado como fallback no `BLOCKED.md` da Semana 1.

---

## O que NÃO fazer na Semana 2

- **Não retomar MNIST sanity** — caso patológico documentado, ROI baixo.
- **Não inventar hiperparâmetros novos sem medir.** Etapa 1 (calibração via `test_spike_balance` adaptado) deve preceder qualquer ajuste.
- **Não rodar `train.py` com 24000 imgs antes de smoke test** — pipeline pode ter regressão silenciosa entre Semanas; sempre validar end-to-end primeiro.
- **Não esperar atingir 90% no primeiro experimento.** Meta original (90% 5w1s) requer pretreino completo + tuning + possivelmente readout linear (Fase C do `PLAN.md` § 5).

---

## Estado de prontidão

✅ Stack PyTorch+CUDA validada
✅ Mecanismos STDP, k-WTA e homeostasis implementados
✅ Pipeline de dados Omniglot pronto
✅ Loop de pretreino + avaliação + baselines pronto
⚠️ Hiperparâmetros calibrados pra regime de MNIST — provavelmente precisam ajuste
⚠️ Smoke test não executado nesta semana — recomendado antes de qualquer experimento sério
⚠️ Verificação de `_proj` no state_dict não feita — risco de embeddings inconsistentes entre runs

**Próxima sessão pode começar pela Etapa 0 (smoke test) sem dependências adicionais.**
