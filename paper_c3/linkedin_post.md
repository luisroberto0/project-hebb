# LinkedIn post — versão longa (PT-BR)

> Status: draft sessão #36. Pra Luis revisar e publicar quando quiser.
> Caracteres alvo: 1500-2500. Atual: ~1900.
> Anexar imagem: `paper_c3/figs/fig1_sparsity_curve.png` (visual contraintuitivo, plateau até 75% sparsity)

---

🧠 **75% das ativações de uma rede neural podem ser zeradas com perda de só 1.5% de acurácia.**

Resultado de um side project pessoal que rodei nos últimos meses. Compartilho como anúncio + convite pra discussão.

**Contexto.** Project Hebb é pesquisa pessoal em arquiteturas neurais bio-inspiradas — plasticidade local, codificação esparsa, few-shot learning. 36 sessões de trabalho, ~50h totais. É exploração de fundo, guiada pela pergunta: o que LLMs *não* fazem que cérebros fazem, e como mecanismos biológicos clássicos se comportam quando aplicados a métodos modernos de deep learning?

**O que testei.** Apliquei *k-WTA esparso* (winner-take-all top-k, princípio biológico clássico do córtex) ao embedding final de uma Prototypical Network em few-shot Omniglot. Três níveis de sparsity: 50%, 75%, 87.5%. Treinamento end-to-end com gradient fluindo pelos top-k channels selecionados.

**Resultado.** Em Omniglot 5-way 1-shot:

▪️ ProtoNet baseline (denso): 94.55%
▪️ Com 50% sparsity: 93.35% (−1.20 p.p.)
▪️ Com 75% sparsity: **93.10% (−1.45 p.p.)** ← sweet spot
▪️ Com 87.5% sparsity: 90.77%

Curva é dramaticamente plana até 75% — só começa a degradar após.

**Validação importante.** Como controle: se eu uso encoder com pesos *aleatórios* (não treinado) + a mesma operação k-WTA, atinjo apenas 37.60%. Isso confirma que o ganho vem do treinamento *sob restrição* de sparsity — o encoder aprende a colocar informação útil nas dimensões que ficam ativas. Não é a estrutura k-WTA isolada que faz o trabalho.

**Por que importa.** Sparsity é característica fundamental da computação cortical (~2-4% de unidades ativas em córtex). Esse resultado é evidência empírica de que princípios bio-inspirados podem coexistir com deep learning mainstream sem custo significativo de performance. Ponto de comparação concreto pra discussão sobre como aproximar arquiteturas neurais de modelos biológicos.

**Honestidade explícita.** Isso não é breakthrough — é estudo empírico controlado de uma combinação não-testada explicitamente. Limitações reconhecidas no paper: single seed, único dataset (Omniglot, visualmente simples), k-WTA aplicado só no embedding final. Workshop-scope, não conference. Side project com tempo limitado.

Marco paralelo do projeto também caracterizou continual learning (sem replay buffer): achado mecanístico de que ProtoNet metric learning é *naturalmente* robusto a catastrophic forgetting em Omniglot. Métodos bio-inspirados que testei não conseguiram bater o baseline naive — documentado honestamente como exploração negativa.

**Por que não submeti pra workshop?** Tempo. Não tenho espaço pra rebuttals/revisões/registration que submissão acadêmica formal exige. Postar aqui alcança parte do que peer-review faria — feedback de gente da área — sem o overhead institucional. Paper draft preservado pra submissão futura se fizer sentido.

📂 Repo completo (código, dados, 8 weekly notes documentando o processo): github.com/luisroberto0/project-hebb
📄 PDF deep dive (paper draft, 8 páginas): [github.com/luisroberto0/project-hebb/blob/main/paper_c3/Project_Hebb_C3_DeepDive.pdf](https://github.com/luisroberto0/project-hebb/blob/main/paper_c3/Project_Hebb_C3_DeepDive.pdf)

Comentários e pushback são muito bem-vindos. Especialmente: alguém já testou k-WTA explícito em ProtoNet com sweep de sparsity? Refs que eu deveria conhecer?

#PesquisaIA #DeepLearning #BiologicallyInspiredAI #ProjectHebb #SideProject

---

## Notas de revisão

- [ ] Verificar contagem de caracteres (atual ~1900, target 1500-2500) ✓
- [ ] Confirmar repo URL correto (github.com/luisroberto0/project-hebb)
- [ ] Decidir se inclui o "Marco paralelo" parágrafo — adiciona ~200 chars mas mostra rigor (caracterização negativa documentada)
- [ ] Decidir entre #PesquisaIA / #IA / #InteligenciaArtificial — qual hashtag tem mais reach na rede do Luis
- [ ] Anexar `paper_c3/figs/fig1_sparsity_curve.png` no post (visual forte)
- [ ] Considerar se "PDF deep dive [github.com/luisroberto0/project-hebb/blob/main/paper_c3/Project_Hebb_C3_DeepDive.pdf](https://github.com/luisroberto0/project-hebb/blob/main/paper_c3/Project_Hebb_C3_DeepDive.pdf)" deveria ser "PDF disponível em [link]" se Overleaf compile já estiver pronto na hora de postar
