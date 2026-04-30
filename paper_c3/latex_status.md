# LaTeX compilation status

> Status pós-sessão #35.

## Compilação local: NÃO testada

`pdflatex` e `bibtex` não instalados na máquina de desenvolvimento local (verificado via `which pdflatex` em 2026-04-30). `main.tex` foi escrito mas **não compilado** localmente nesta sessão.

## Pra compilar quando LaTeX estiver disponível

```bash
cd paper_c3
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Esperado: `main.pdf` gerado, ~6-8 páginas, com:
- Title + author + abstract
- 6 sections (Introduction, Background, Method, Experiments, Discussion, Conclusion)
- Figure 1 (sparsity-accuracy curve, inserida em Section 4.3 via `\includegraphics{figs/fig1_sparsity_curve.pdf}`)
- Bibliography no fim (estilo `plainnat`, 14 refs de `refs.bib`)

## Opções para compilação

1. **Instalar TeX Live** (Windows): `winget install TeXLive.TeXLive` ou MikTeX. ~3-5 GB.
2. **Overleaf** (online): upload de `paper_c3/` inteiro, configurar main.tex como entrypoint.
3. **Docker**: `docker run --rm -v $(pwd):/work -w /work texlive/texlive pdflatex main.tex`
4. **Sessão #36 ou #37**: compilação fica como checkpoint pra peer review.

## Style file pra workshop específico

`main.tex` atualmente usa `\documentclass[11pt]{article}` + `geometry` margins 1in como template genérico funcional. Adaptação pra style file específico do workshop (ex: `neurips_2024.sty`) é troca trivial:

```latex
% Substituir:
% \documentclass[11pt]{article}
% \usepackage[margin=1in]{geometry}

% Por (NeurIPS workshop):
% \documentclass{article}
% \usepackage[final]{neurips_2024}
```

Style file oficial baixado do site do workshop quando submission window abrir.

## Verificações pré-compilação (manual review feito sessão #35)

- [x] `\documentclass` declarado
- [x] Todos os pacotes `\usepackage` listados (amsmath, graphicx, booktabs, hyperref, natbib)
- [x] `\title`, `\author`, `\affil`, `\date` corretos
- [x] `\begin{document}` ... `\end{document}`
- [x] `\maketitle` chamado
- [x] `\bibliography{refs}` aponta pra `refs.bib`
- [x] `\bibliographystyle{plainnat}` antes do `\begin{document}`
- [x] Citações no formato `\citep{key}` e `\citet{key}`
- [x] 14 keys do `refs.bib` aparecem em citações no texto (verificar diferenças via grep antes da submissão)
- [x] Figura 1 referenciada via `\ref{fig:sparsity_curve}` e `\includegraphics{figs/fig1_sparsity_curve.pdf}`
- [x] Tabela 1 referenciada via `\ref{tab:main}` e `\label{tab:main}` no booktabs

## Possíveis erros esperados na primeira compilação

- **Missing fonts**: usar `lualatex` ou instalar pacotes de fonte adicionais
- **natbib + neurips style conflict**: alguns workshop styles redefinem natbib; pode precisar ajustar `\bibliographystyle`
- **Encoding UTF-8**: `\usepackage[utf8]{inputenc}` já incluso
- **figs/fig1_sparsity_curve.pdf** path: confirmar que LaTeX encontra `figs/` (cwd deve ser `paper_c3/` na hora de `pdflatex`)

## Status pra peer review (#36)

Paper LaTeX está **escrito mas não compilado**. Peer reviewer pode:
1. Compilar localmente (se tiver LaTeX) e revisar PDF
2. Revisar markdown sources diretamente (ainda válidos: intro.md, background.md, etc.)
3. Pedir compilação pré-review (sessão #36 pode tentar via Overleaf/Docker)

Próxima sessão (#36): peer review interno + compilação + revisão final + decisão sobre submissão.
