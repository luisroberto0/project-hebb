# Rascunho — post LinkedIn (PT): a jornada pós-LLM do Project Hebb (arco completo)

> Rascunho reversível, estilo do post do C3. **Não publicado** — Luis revisa e decide (publicar / editar / versão EN). Cobre o arco de 3 atos: 4 capacidades sem vantagem → a premissa-mãe testada (SoftHebb) funciona → mas é eficiência, não capacidade. Honestidade metodológica como protagonista.

---

## Versão A — narrativa "o que aprendi" (~620 palavras)

**Passei um ano testando uma hipótese incômoda: e se o caminho para uma IA fundamentalmente diferente das LLMs não fosse mais escala, mas sim *neurônios que funcionam diferente*?**

Plasticidade local em vez de backpropagation. Codificação esparsa. Aprendizado sem rótulos. A promessa: capacidades que LLMs não têm — aprender com 1 exemplo, aprender continuamente sem esquecer, rodar com energia mínima, raciocinar sobre tempo.

Ataquei as quatro capacidades, uma a uma, com um critério numérico definido **antes** de cada experimento e a regra de documentar as falhas com mais cuidado que os sucessos. O placar honesto da primeira metade:

🔴 **One-shot cross-domain** — um encoder esparso treinado num domínio vira ruído em outro, indistinguível de pesos aleatórios.
🔴 **Aprendizado contínuo** — os mecanismos bio-inspirados não bateram um baseline simples.
🔴 **Eficiência radical** — a rede spiking só é eficiente no papel; em hardware real foi 80–300× *mais lenta*.
🟡 **Raciocínio temporal** — parecia um positivo: a rede spiking explorava o *timing*. Aí rodei o controle certo — uma RNN comum (GRU) — e ela ganhou da minha rede por 10 pontos. O timing era real, mas não tinha nada de "spiking".

Quatro capacidades, nenhuma vantagem competitiva. Parecia o fim — "a bio-inspiração não entrega". Mas havia um detalhe que me incomodava: **eu nunca tinha testado a premissa-mãe do projeto**. Todos os "sucessos" do caminho ainda usavam backpropagation por baixo. A pergunta original — *uma regra de plasticidade puramente local, sem backprop, aprende algo útil?* — continuava em aberto.

Então testei. E aqui a história vira.

**Uma pilha convolucional treinada SÓ por uma regra Hebbiana local e competitiva — zero backpropagation nas features — atinge 80% no CIFAR-10.** Não por sorte: ela supera pesos aleatórios na mesma arquitetura por ~12 pontos (o sinal é real, não um artefato da arquitetura), a competição entre neurônios é essencial (desligá-la colapsa tudo), e fica a só 7 pontos do backprop. Generaliza: a margem sobre o aleatório persiste (≈9 pontos) em 200 classes do Tiny-ImageNet. **Pela primeira vez em toda a jornada, o mecanismo bio-inspirado, isolado, carregava sinal genuíno.**

Empolgado, fui testar continual learning: treinei essa pilha em tarefas sequenciais. **Ela não esquece** — enquanto o backprop sofre *catastrophic forgetting* (a acurácia de uma tarefa despenca de 68% para 35% ao aprender as outras), a versão Hebbiana até *melhora*. Parecia a vitória que faltava.

Mas eu já tinha aprendido a lição. Rodei o controle contra a minha própria hipótese favorita: um autoencoder — backprop, mas **não-supervisionado**. E ele **também não esquece**. A resistência ao esquecimento não vinha da plasticidade local. Vinha de ser *não-supervisionado*. O que destrói o conhecimento antigo é a *supervisão*, não o backprop em si.

Então qual é, afinal, a contribuição? Eu media: a regra local treina as mesmas features **21× mais rápido** que o backprop (33 segundos contra 11 minutos), sem rótulos e numa única passada pelos dados.

**A tese honesta com que termino:** a plasticidade local não te dá superpoderes que o backprop não tem. Ela te dá os *mesmos* poderes — features úteis, robustas, que escalam — por **uma fração do custo**. Não é superioridade. É **eficiência**. E para casos onde rótulos são caros, dados chegam em fluxo, ou energia é escassa (sensores, edge, talvez silício neuromórfico), isso pode importar muito.

A meta-lição, que vale mais que qualquer resultado: **o experimento mais valioso é o controle que você roda contra a sua própria hipótese favorita.** Foi ele que, vez após vez, separou o que eu *queria* que fosse verdade do que *era*. Cada "vitória" da bio-inspiração se dissolveu sob escrutínio numa propriedade mais simples — mas a bio-inspiração frequentemente entrega essa propriedade de forma mais eficiente.

Resultados negativos rigorosos, e um positivo honestamente delimitado, são ciência. Compartilho na esperança de poupar o tempo de alguém — ou provocar o experimento que não pensei.

*Project Hebb — pesquisa independente. Código e síntese completa abertos.*

---

## Notas para decisão (Luis)

- **Arco de 3 atos:** 4 negativos → premissa-mãe funciona (SoftHebb) → mas é eficiência (controle do autoencoder + 21×). Muito mais forte que a versão antiga (só negativos).
- **Números-âncora:** 80% CIFAR-10 sem backprop; +12pp sobre random; não-esquece vs backprop −16.78; 21× mais rápido. Conferir contra os WEEKLY antes de postar.
- **Idioma:** PT (rede BR, como o C3). Posso fazer versão EN para alcance de pesquisa.
- **Gancho do autoencoder:** é o clímax honesto ("rodei o controle contra minha própria hipótese") — funciona muito bem. Confirmar se quer expor o "desinflar" tão diretamente.
- **Crédito:** o SoftHebb (Journé et al., ICLR 2023) não é invenção minha — reproduzi com rigor + os controles. Vale mencionar no post ou nos comentários para não soar como autoria do método.
- **Alternativa:** versão B curta (~250 palavras) só com o arco "testei minha hipótese favorita e o controle a desinflou — mas revelou eficiência" se preferir punch.
- **Link/anexo:** repo + SYNTHESIS.md. Figura sugerida: a dos filtros Hebbian (`experiment_06_plasticity/figs/fig_filters.pdf`) ou a curva de escala.
