# Figures — paper Marco 2-C

Geradas via `paper_marco2c/generate_figures.py`. Números hardcoded de `experiment_05_temporal/WEEKLY-1.md` e `results_*.txt` (#72–#77b); não rerodam experimentos.

| Figura | Conteúdo | Seção |
|---|---|---|
| `fig1_shd_conditions` | Barras das 3 condições SHD (timing-blind / SNN-ff / SNN-rec) com IC95%, chance line, e o gap bruto de +19.7 p.p. anotado (com a ressalva de que ~metade é timing). | §4.1 |
| `fig2_resolution` | Acc vs nº de bins temporais. A curva crescente isola o **timing genuíno (+10.18 p.p.,** bins 1→100) do confound arquitetural (+9.5 p.p., bins=1 vs cego). | §4.2 |
| `fig3_kwta_temporal` | Acc vs sparsity k-WTA temporal. Tolerante até 75% (−1.50 p.p., paralelo ao C3 espacial); colapsa ao baseline cego em >96%. | §4.4 |

## Regenerar

```bash
cd paper_marco2c
python generate_figures.py
```

300 DPI, PNG + PDF. Cores: cinza=timing-blind, laranja=feedforward, azul=recurrent.
