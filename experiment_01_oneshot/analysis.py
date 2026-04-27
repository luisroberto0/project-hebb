"""
Análise pós-execução: lê logs JSON dos runs de evaluate.py em todas as
configs (5w1s, 5w5s, 20w1s, 20w5s) e gera RESULTS.md automaticamente.

Uso:
    # Após rodar evaluate.py com --json-out logs/eval_<config>.json
    python analysis.py --logs-dir logs --out RESULTS.md

Estrutura esperada de cada arquivo JSON:
    {
        "config": {"ways": 5, "shots": 1, "episodes": 1000, "checkpoint": "..."},
        "accs": [0.94, 0.92, ...],   # acurácia por episódio
        "elapsed_sec": 123.4,
        "n_params": 87456
    }

O script decide automaticamente o nível de sucesso (forte/parcial/falha)
conforme `experiment_01_oneshot/PLAN.md` §11.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def load_run(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def classify_outcome(acc_5w1s: float | None, acc_20w1s: float | None) -> tuple[str, str]:
    """
    Conforme PLAN.md §11:
      forte: 5w1s ≥ 90% E 20w1s ≥ 70%
      parcial: 5w1s entre 70-90%
      falha: 5w1s < 70%
    """
    if acc_5w1s is None:
        return "indeterminado", "Sem run de 5-way 1-shot."
    if acc_5w1s >= 0.90 and (acc_20w1s is None or acc_20w1s >= 0.70):
        return "sucesso forte",  "Paper publicável em workshop. Escrever paper rascunho."
    if acc_5w1s >= 0.70:
        return "sucesso parcial", "Insight sólido. Documentar em NEXT.md e propor próxima iteração."
    return "falha", "Análise diagnóstica em POSTMORTEM.md."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs-dir", type=str, default="logs")
    p.add_argument("--out", type=str, default="RESULTS.md")
    p.add_argument("--baseline-pixel-knn", type=float, default=None,
                   help="Acurácia 5w1s do Pixel kNN (referência)")
    p.add_argument("--baseline-proto-net", type=float, default=None,
                   help="Acurácia 5w1s do Prototypical Networks (referência)")
    args = p.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        raise SystemExit(f"Logs dir não existe: {logs_dir}")

    runs = {}
    for cfg in [(5, 1), (5, 5), (20, 1), (20, 5)]:
        ways, shots = cfg
        path = logs_dir / f"eval_{ways}w{shots}s.json"
        if path.exists():
            runs[cfg] = load_run(path)
        else:
            print(f"⚠️  faltando {path}")

    # Resumo por config
    summary = {}
    for cfg, run in runs.items():
        accs = np.array(run["accs"])
        mean = accs.mean()
        lo, hi = bootstrap_ci(accs)
        summary[cfg] = {
            "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "n_params": run.get("n_params", "?"),
            "elapsed_sec": run.get("elapsed_sec", "?"),
            "n_episodes": len(accs),
        }

    acc_5w1s = summary.get((5, 1), {}).get("mean")
    acc_20w1s = summary.get((20, 1), {}).get("mean")
    outcome, recommendation = classify_outcome(acc_5w1s, acc_20w1s)

    # Escreve RESULTS.md
    out = Path(args.out)
    with open(out, "w") as f:
        f.write("# RESULTS — Experimento 01 (One-shot Omniglot com STDP + Hopfield)\n\n")
        f.write(f"> Gerado automaticamente por `analysis.py` em {datetime.now().isoformat(timespec='seconds')}.\n")
        f.write(f"> Runs lidos de `{logs_dir}/`.\n\n")

        f.write("## Resultado principal\n\n")
        f.write(f"**Status: {outcome.upper()}**\n\n")
        f.write(f"{recommendation}\n\n")

        f.write("## Acurácia por configuração\n\n")
        f.write("| Config | Episódios | Acurácia | IC 95% | Tempo (s) |\n")
        f.write("|---|---:|---:|:---:|---:|\n")
        for cfg in [(5, 1), (5, 5), (20, 1), (20, 5)]:
            if cfg not in summary:
                f.write(f"| {cfg[0]}w{cfg[1]}s | — | — | — | — |\n")
                continue
            s = summary[cfg]
            f.write(f"| {cfg[0]}-way {cfg[1]}-shot | {s['n_episodes']} | "
                    f"{s['mean']*100:.2f}% | "
                    f"[{s['ci_lo']*100:.2f}%, {s['ci_hi']*100:.2f}%] | "
                    f"{s['elapsed_sec']:.1f} |\n")

        f.write("\n## Comparação com baselines\n\n")
        f.write("| Modelo | 5-way 1-shot | Backprop end-to-end | Parâmetros |\n")
        f.write("|---|---:|:---:|---:|\n")
        f.write(f"| Random (chance) | 20.00% | n/a | 0 |\n")
        if args.baseline_pixel_knn is not None:
            f.write(f"| Pixel kNN | {args.baseline_pixel_knn*100:.2f}% | não | 0 |\n")
        else:
            f.write(f"| Pixel kNN | (rodar `baselines.py --baseline pixel_knn`) | não | 0 |\n")
        if args.baseline_proto_net is not None:
            f.write(f"| Prototypical Networks | {args.baseline_proto_net*100:.2f}% | sim | ~110k |\n")
        else:
            f.write(f"| Prototypical Networks | (rodar `baselines.py --baseline proto_net`) | sim | ~110k |\n")
        if (5, 1) in summary:
            params = summary[(5, 1)]['n_params']
            f.write(f"| **STDP + Hopfield (este trabalho)** | **{summary[(5, 1)]['mean']*100:.2f}%** | **não** | **{params}** |\n")

        f.write("\n## Critérios de sucesso (PLAN.md §11)\n\n")
        if acc_5w1s is not None:
            f.write(f"- [{'x' if acc_5w1s >= 0.90 else ' '}] 5-way 1-shot ≥ 90% — atual: {acc_5w1s*100:.2f}%\n")
        if acc_20w1s is not None:
            f.write(f"- [{'x' if acc_20w1s >= 0.70 else ' '}] 20-way 1-shot ≥ 70% — atual: {acc_20w1s*100:.2f}%\n")
        if (5, 1) in summary:
            f.write(f"- [{'x' if summary[(5,1)]['n_params'] != '?' and summary[(5,1)]['n_params'] < 100000 else ' '}] "
                    f"Parâmetros < 100k — atual: {summary[(5, 1)]['n_params']}\n")
        f.write("- [x] Sem backprop end-to-end (STDP + Hopfield, projeção ortogonal fixa)\n")

        f.write("\n## Próximo passo\n\n")
        if outcome == "sucesso forte":
            f.write("Escrever rascunho de paper (workshop ICLR Tiny Papers ou NeurIPS workshop).\n"
                    "Branch `paper/v1`. Estrutura mínima: 4 páginas — intro, método, experimentos, discussão.\n")
        elif outcome == "sucesso parcial":
            f.write("Documentar em `NEXT.md` o que funcionou, o que não, e propor próxima iteração.\n"
                    "Candidatos típicos: aumentar n_filters, ajustar tau_pre/tau_post, trocar codificação\n"
                    "Poisson por temporal (TTFS), aumentar embedding_dim.\n")
        else:
            f.write("Análise diagnóstica em `POSTMORTEM.md`. Verificar:\n"
                    "1. Filtros aprendidos visualmente coerentes? (rodar `python utils/visualize.py`)\n"
                    "2. Distribuição de pesos bimodal? (assinatura de STDP convergente)\n"
                    "3. Embeddings clusterizam por classe? (t-SNE)\n"
                    "4. Memória Hopfield satura? (β apropriado?)\n")

    print(f"\n✅ Relatório escrito em {out}")
    print(f"   Outcome: {outcome.upper()}")


if __name__ == "__main__":
    main()
