# Rascunho — post LinkedIn (PT): a jornada pós-LLM do Project Hebb

> Rascunho reversível, no estilo do post do C3. **Não publicado** — Luis revisa e decide (publicar / editar / descartar / versão EN). Distila o `SYNTHESIS.md` para um público geral, com a honestidade metodológica como protagonista.

---

## Versão A — narrativa "o que aprendi" (~480 palavras)

**Passei o último ano testando uma hipótese incômoda: e se o caminho para uma IA fundamentalmente diferente das LLMs não fosse mais escala, mas sim *neurônios que funcionam diferente*?**

Plasticidade local em vez de backpropagation. Codificação esparsa. Dinâmica temporal de spikes. A promessa: capacidades que LLMs não têm — aprender com 1 exemplo, aprender continuamente sem esquecer, rodar em CPU comum, raciocinar sobre tempo.

Ataquei as quatro capacidades, uma a uma, com um critério numérico definido **antes** de cada experimento e a regra de documentar as falhas com mais cuidado que os sucessos. O placar honesto:

🔴 **One-shot cross-domain** — um encoder esparso treinado num domínio e congelado vira ruído em outro. Pior: fica indistinguível de um encoder aleatório. A esparsidade que ajudava *dentro* do domínio colapsa sob mudança de domínio.

🔴 **Aprendizado contínuo** — os mecanismos bio-inspirados não bateram um baseline simples. O baseline já era robusto; não havia o que melhorar.

🔴 **Eficiência radical** — a rede spiking só é eficiente em *operações teóricas*. Em hardware real (CPU), foi 80–300× **mais lenta**. A vantagem energética exige silício neuromórfico dedicado — não se realiza no computador que você tem.

🟡 **Raciocínio temporal** — aqui parecia haver um positivo: a rede spiking explorava o *timing* dos spikes, ganhando ~20 pontos sobre um baseline cego ao tempo. O primeiro resultado positivo do projeto.

Aí veio a parte que mais me ensinou. Numa revisão adversarial do meu próprio trabalho, a pergunta certa apareceu: *e se eu trocar a rede spiking por uma RNN comum?* Rodei o controle. Uma **GRU convencional fez 79,6% — 10 pontos acima da minha rede spiking.** O timing era real, mas não tinha nada de "spiking": qualquer recorrência o captura, e a convencional captura melhor.

O único positivo se desinflou. As quatro capacidades convergiram para a mesma conclusão honesta: **no regime que consigo testar, a bio-inspiração não entrega vantagem competitiva sobre métodos convencionais.**

Isso é um fracasso? Não acho. É um mapa. Mapeei *onde* a bio-inspiração não funciona e *por quê* — e isso vale mais que um número inflado. Encontrei um fio coerente e original no caminho (uma forma de esparsidade que é tolerada dentro do domínio em qualquer eixo — espaço ou tempo — mas frágil fora dele). E aprendi que o controle adversarial mais valioso é o que você roda contra a sua *própria* hipótese favorita.

A aposta pós-LLM de verdade — plasticidade local, online, **sem** backpropagation — eu nunca cheguei a testar de fato. Todos os "sucessos" do caminho ainda usavam backprop por baixo. É lá, e em hardware neuromórfico real, que a tese ainda pode morar. Mas é uma aposta maior, para um próximo capítulo.

Resultados negativos rigorosos, documentados com honestidade, são ciência. Compartilho na esperança de que poupem o tempo de alguém — ou provoquem o experimento que eu não pensei.

*Project Hebb — pesquisa independente. Código e síntese completa abertos.*

---

## Notas para decisão (Luis)

- **Tom:** honestidade como herói, sem auto-flagelo nem hype. Calibrar se quer mais técnico ou mais acessível.
- **Idioma:** PT (rede brasileira, como o C3). Posso fazer versão EN para alcance de pesquisa.
- **Gancho do GRU:** é o clímax — funciona bem em post ("rodei o controle contra minha própria hipótese"). Confirmar se quer expor o "desinflar" tão diretamente.
- **Link:** apontar para o repo / SYNTHESIS.md / paper_marco2c como apêndice técnico.
- **Alternativa:** versão B mais curta (~200 palavras, só o gancho do GRU + a lição) se preferir punch.
