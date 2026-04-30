# Section 6: Conclusion

> Status: draft sessão #34. Revisar na sessão #35.
> Word count target: 150-300 palavras. Atual: ~220.
> Tom: declarativo, sem repetir abstract.

---

Este paper apresentou um estudo empírico controlado da tolerância a sparsity em Prototypical Networks aplicadas a few-shot learning em Omniglot. Aplicando k-WTA com três níveis de sparsity (50%, 75%, 87.5%) ao embedding final de uma CNN-4 padrão e treinando end-to-end com gradient flow pelos top-$k$ channels selecionados, observamos que **75% das ativações podem ser zeradas com perda de apenas 1.45 pontos percentuais** em acurácia 5-way 1-shot (93.10% vs 94.55% baseline). Validação com encoder não treinado confirma que o ganho vem da otimização sob restrição (37.60% sem treino vs 93.10% treinado), não da estrutura k-WTA isolada.

O resultado contribui empiricamente para a discussão sobre compatibilidade entre princípios de codificação esparsa biológica e deep representation learning. Embora não argumentemos pela fidelidade biológica do mecanismo testado — k-WTA em embedding final de uma CNN é uma simplificação substancial de qualquer processo cortical — o resultado oferece um ponto de comparação concreto: prototype-based metric learning não é destruído por restrições explícitas de sparsity, e o encoder consegue aprender a usar dimensões esparsas de forma discriminativa quando treinado para isso.

Trabalho futuro inclui sweeps mais finos em torno do threshold observado entre 75% e 87.5% sparsity, validação multi-seed para variabilidade de treino, aplicação em camadas intermediárias além do embedding final, e teste em datasets visualmente mais complexos. A combinação de k-WTA com regras de plasticidade local em arquiteturas continual permanece direção em aberto, motivada pela robustez natural de prototype-based methods a catastrophic forgetting observada em exploração paralela.

---

## Notas de revisão (pra sessão #35)

- [ ] Verificar word count (atual ~220, target 150-300) ✓
- [ ] Conferir que números batem com Tabela 1 (93.10%, 94.55%, 37.60%)
- [ ] Decidir tone do parágrafo final — atualmente menciona Marco 1 indiretamente, considerar se isto fortalece ou dilui a conclusão
