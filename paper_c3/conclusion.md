# Section 6: Conclusion

> Status: slim revisado sessão #35. Word count: ~210 palavras (target 150-300).
> Tom: declarativo, sem repetir abstract.

---

Este paper apresentou um estudo empírico controlado da tolerância a sparsity em Prototypical Networks aplicadas a few-shot learning em Omniglot. Aplicando k-WTA com três níveis de sparsity (50%, 75%, 87.5%) ao embedding final de uma CNN-4 padrão e treinando end-to-end, observamos que **75% das ativações podem ser zeradas com perda de apenas 1.45 pontos percentuais** em acurácia 5-way 1-shot (93.10% vs 94.55% baseline). Validação com encoder não treinado confirma que o ganho vem da otimização sob restrição (37.60% sem treino vs 93.10% treinado), não da estrutura k-WTA isolada.

O resultado contribui empiricamente para a discussão sobre compatibilidade entre princípios de codificação esparsa biológica e deep representation learning. Embora não argumentemos pela fidelidade biológica do mecanismo testado, o resultado oferece um ponto de comparação concreto: prototype-based metric learning não é destruído por restrições explícitas de sparsity, e o encoder consegue aprender a usar dimensões esparsas de forma discriminativa quando treinado para isso.

Trabalho futuro inclui sweeps mais finos em torno do threshold observado entre 75% e 87.5% sparsity, validação multi-seed, aplicação em camadas intermediárias, e teste em datasets visualmente mais complexos.
