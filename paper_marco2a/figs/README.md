# Figures — paper Marco 2-A

Geradas via `paper_marco2a/generate_figures.py` (sessão #60).

| Figura | Conteúdo | Inserir em |
|---|---|---|
| `fig1_crossdomain_bars.{png,pdf}` | Bar chart horizontal das 8 condições da Table 1 + linha de chance (20%). Banda sombreada destaca o cluster k-WTA cross-domain (5 condições CNN-forward) colapsando num spread de 0.52 p.p., enquanto os baselines retreinados em CUB ficam muito acima. Cores por categoria: verde=retreinado (upper bound), laranja=Pixel kNN, azul=C3 source-trained, cinza hachurado=random encoder. | Section 4.2 (Cross-Domain Results) |
| `fig2_effect_collapse.{png,pdf}` | Dual-panel in-domain (Omniglot) vs cross-domain (CUB-200) ao longo da sparsity k∈{8,16,32,64}. Painel esquerdo mostra o spread de 3.78 p.p. in-domain; painel direito (mesmo span de 8 p.p.) mostra o colapso pra 0.52 p.p. de ruído ~22%, com linha de chance. Visualiza o k-WTA effect collapse. | Section 4.3 (In-Domain vs Cross-Domain Effect) |
| `fig3_bottleneck_waterfall.{png,pdf}` | Waterfall da decomposição de gargalos: random encoder (21.91%) → +Omniglot training C3 k=16 (+0.18 p.p., negligível) → +target training CUB 28×28 (+12.22 p.p.) → +resolução & cor CUB 84×84 RGB (+15.53 p.p.). Mostra os dois contribuidores grandes e ~aditivos vs. o passo de treino na fonte, irrelevante. | Section 4.4 (Bottleneck Decomposition) |

## Regenerar

```bash
cd paper_marco2a
python generate_figures.py
```

Dados hardcoded no script (sessões #52-#57, consolidados em `experiment_03_crossdomain/WEEKLY-1.md`). Não rerodam experimentos — accuracies e IC95% bootstrap já documentados. Todos os valores conferem 1-a-1 com Table 1 (`experiments.md`) e Table 2 (effect-collapse).

## Resolução

PNG e PDF em **300 DPI** (workshop submission standard). PDFs vetoriais escalam sem perda; PNGs úteis pra preview/markdown.

## Style notes

- Cores: estende a paleta do paper C3 (`#1f77b4` azul, `#d62728` vermelho) com `#2ca02c` verde (retreinado/upper bound), `#ff7f0e` laranja (pixel kNN), `#7f7f7f` cinza (random encoder).
- Fonte: matplotlib default (DejaVu Sans). Conferir contra workshop template oficial quando definido.
- Error bars (Fig 1): IC95% bootstrap sobre as 5 médias por seed.
- Random encoder hachurado (`//`) pra distinguir condição sem treino.
- Annotations sutis (caixas `wheat`) destacando os spreads-chave sem poluir.
