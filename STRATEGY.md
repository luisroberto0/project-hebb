# Estratégia da pesquisa STDP+Hopfield

## Decisão tomada em 2026-04-27 (sessão #7, encerramento)

Combinar tuning + opção de pivot:

1. **Próximas 2-3 sessões:** testar H_theta_omn e H_norm (custo baixo, alto ROI)
2. **Se H_theta_omn e H_norm não destravarem:** considerar pivot pra abordagem adjacente (Hopfield puro, meta-learning bio, ou prototypical com features esparsas)
3. **Brian2 (validação contra paper original) fica como último recurso** — alto custo (~1 semana), só justificável se pivot também falhar

## Critério pra cada decisão futura

- Sessão = 60-90 min, max 2x/semana
- Após cada sessão: atualizar PLAN.md "Notas de iteração"
- Se 3 sessões consecutivas sem progresso (sinal acima de chance), revisar STRATEGY.md
