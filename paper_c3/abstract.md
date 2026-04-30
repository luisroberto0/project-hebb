# Abstract

> Status: draft sessão #35. Word count: ~180 palavras (target 150-200).
> Tom: declarativo, workshop honesto, sem hype, sem "we propose" se não há método novo.

---

Sparsity é uma característica observada na computação cortical, mas seu impacto em métodos modernos de few-shot metric learning não é bem caracterizado. Investigamos quanto sparsity Prototypical Networks (Snell et al., 2017) toleram quando aplicada explicitamente ao embedding final via k-WTA. Aplicamos a operação k-WTA top-$k$ ao embedding 64-dimensional de uma CNN-4 padrão, treinando end-to-end com gradient flow pelos top-$k$ channels selecionados, em três níveis de sparsity (50%, 75%, 87.5%) e avaliando em Omniglot 5-way 1-shot e 20-way 1-shot. **Sparsity de 75% custa apenas 1.45 pontos percentuais em acurácia (93.10% vs 94.55% baseline ProtoNet completo).** Mesmo em 87.5% sparsity, a performance permanece em 90.77%. Como controle, validamos que random encoder com k-WTA aplicado atinge apenas 37.60% — confirmando que o ganho vem da otimização sob restrição, não da estrutura k-WTA isolada. O resultado oferece evidência empírica de que codificação esparsa neural-inspirada é compatível com prototype-based metric learning, complementando trabalhos prévios em redes de Hopfield modernas e Sparse Distributed Memory.
