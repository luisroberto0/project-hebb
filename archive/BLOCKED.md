# BLOCKED — Experimento 01 Semana 1

**Status:** **FECHADO em 2026-04-27 (sessão #6).** Semana 1 declarada caso patológico documentado.

---

## Resumo do fechamento

Após 5 sessões de iteração (k-WTA, A_pre/A_post, theta_plus, combos), espaço de hiperparâmetros foi exaurido sem destravar acima de 17.76%. Causa raiz isolada: **MNIST com kernel=28 é caso degenerado pra k-WTA** (output espacial 1×1, todos filtros disputam mesma posição). Isso não é falha de stack — stack foi validada por outras vias (testes sintéticos passam, STDP atualiza pesos, homeostasis funciona mecanicamente).

Decisão registrada como permanente em `PLAN.md` raiz (§ Decisões arquiteturais, 2026-04-27): pivot pra Semana 2 (Omniglot conv real, kernel=5 + pool, output multi-posição).

## Hipóteses

Todas atendidas ou rejeitadas. Espaço esgotado:

| Hipótese | Status final |
|----------|--------------|
| H_assignment (bug em assign/evaluate) | ✗ Descartada via `tests/test_assignment.py` (3/3 ✓) |
| H_balance (LTP/LTD sozinho resolve) | ✗ Descartada (sessão #3, R=3 e R=10 colapsam) |
| H_dose (escalar filtros/dados) | ✗ Descartada (sessão #3, Config A piorou) |
| H_homeostasis-sozinha | ✗ Descartada parcialmente (sessão #4, mecânica OK acc não destrava) |
| H_combo (homeostasis + LTP/LTD juntos) | ✗ Descartada (sessão #5, 13.76% pior que componentes) |
| H_arch (caso patológico, mover pra Omniglot) | ✓ **ACEITA** (sessão #6, esta decisão) |
| H_paper_replicability (Brian2) | ⏸ Não testada — fica como fallback se Semana 2 também falhar |

## Onde a discussão continua

- `experiment_01_oneshot/WEEKLY-1.md` — resumo executivo final da Semana 1
- `experiment_01_oneshot/WEEKLY-2-NEXT.md` — preparação Semana 2 (próximos passos concretos)
- `PLAN.md` raiz § Decisões arquiteturais — racional do pivot
- `experiment_01_oneshot/PLAN.md` § Roadmap — status atualizado de Semana 1 e 2

Este arquivo (`BLOCKED.md`) fica como artefato histórico do bloqueio. **Não há ação pendente.**
