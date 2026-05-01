# LinkedIn post — versão curta (PT-BR)

> Status: draft sessão #36. Fallback se versão longa parecer demais.
> Caracteres alvo: ~800. Atual: ~750.
> Anexar imagem: `paper_c3/figs/fig1_sparsity_curve.png`

---

🧠 Project Hebb — side project pessoal de pesquisa em IA bio-inspirada.

**Pergunta:** quanto sparsity uma rede de few-shot learning tolera?

**Setup:** apliquei k-WTA esparso (mecanismo clássico do córtex) ao embedding final de uma Prototypical Network em Omniglot. Treinamento end-to-end com gradient pelos top-k channels.

**Resultado:** zerar 75% das ativações custa só 1.45 p.p. de acurácia (93.10% vs 94.55% baseline). Validação importante: random encoder + k-WTA = 37.60% — ganho vem do treino *sob restrição*, não da estrutura k-WTA isolada.

**Por que importa:** sparsity é característica do córtex (~2-4% unidades ativas). Esse resultado mostra que princípios bio-inspirados podem coexistir com deep learning mainstream sem custo significativo. Não é breakthrough — é estudo empírico controlado, workshop-scope, com limitações documentadas (single seed, único dataset, k-WTA só no embedding).

📂 Repo: github.com/luisroberto0/project-hebb
📄 PDF deep dive (8 pgs): em breve no repo

Pushback bem-vindo, especialmente refs que eu deveria conhecer.

#PesquisaIA #DeepLearning #ProjectHebb

---

## Notas de revisão

- [ ] Contagem de caracteres (atual ~750, target ~800) ✓
- [ ] Decidir entre versão curta vs longa baseado em audiência alvo
- [ ] Confirmar repo URL
- [ ] Anexar `paper_c3/figs/fig1_sparsity_curve.png`
