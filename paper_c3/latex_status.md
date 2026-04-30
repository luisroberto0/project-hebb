# LaTeX compilation status

> Status pós-sessão #36 (verificação reconfirmada, decisão de publicação alterada).

## Compilação local: NÃO disponível em Windows dev machine

`pdflatex` e `bibtex` não instalados (verificado novamente em sessão #36, mesmo resultado da #35). `main.tex` foi escrito mas **não compilado** localmente. Sem TeX Live ou MikTeX no ambiente atual.

## Decisão de publicação atualizada (sessão #36)

**Não submeter pra NeurIPS Bio-Plausible Workshop.** Em vez disso, postar no LinkedIn em PT como anúncio + link pro repo + PDF como deep dive anexado. Razão registrada em `STRATEGY.md` "Decisão pós-#35: LinkedIn em vez de NeurIPS".

**Implicação pra LaTeX:** PDF ainda é necessário (anexo do post LinkedIn), mas urgência reduzida. Não bloqueia publicação — Luis pode compilar via Overleaf no momento que quiser.

## Plano de compilação recomendado: Overleaf (~10 min, sem instalação)

Caminho mais rápido pra Luis ter o PDF:

1. Acessar [overleaf.com](https://www.overleaf.com/) (criar conta gratuita se necessário)
2. **New Project → Upload Project**
3. Upload do conteúdo de `paper_c3/`:
   - `main.tex`
   - `refs.bib`
   - Pasta `figs/` com `fig1_sparsity_curve.pdf` e `fig2_validation.pdf`
4. Definir `main.tex` como Main Document (botão "Menu" → "Settings" → "Main document")
5. Compile button — Overleaf roda os 3 passes automaticamente (pdflatex → bibtex → pdflatex × 2)
6. Download PDF
7. Renomear pra `Project_Hebb_C3_DeepDive.pdf` (nome amigável pra LinkedIn)
8. Adicionar ao repo manualmente: `git add paper_c3/Project_Hebb_C3_DeepDive.pdf && git commit -m "paper(c3): add compiled PDF"`

## Outras opções (alternativas se Overleaf não funcionar)

### Opção 2: Docker (rápido, sem instalação permanente)

```bash
cd paper_c3
docker run --rm -v $(pwd):/work -w /work texlive/texlive:latest sh -c \
  "pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"
```

Requer Docker Desktop instalado no Windows. ~3 GB de download na primeira vez.

### Opção 3: Instalar MikTeX local (Windows)

```powershell
winget install MiKTeX.MiKTeX
# após instalação:
cd paper_c3
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

~500 MB. Persistente; útil se vai compilar várias vezes.

### Opção 4: TeX Live (mais completo, maior)

```powershell
winget install TeXLive.TeXLive
```

~3-5 GB. Recomendado se vai escrever muitos papers.

## Verificações pré-compilação (manual review feito sessão #35)

- [x] `\documentclass` declarado
- [x] Todos os pacotes `\usepackage` listados (amsmath, graphicx, booktabs, hyperref, natbib)
- [x] `\title`, `\author`, `\affil`, `\date` corretos (autoria sessão #32 do PLAN)
- [x] `\begin{document}` ... `\end{document}`
- [x] `\maketitle` chamado
- [x] `\bibliography{refs}` aponta pra `refs.bib`
- [x] `\bibliographystyle{plainnat}` antes do `\begin{document}`
- [x] Citações no formato `\citep{key}` e `\citet{key}`
- [x] 14 keys do `refs.bib` aparecem em citações no texto
- [x] Figura 1 referenciada via `\ref{fig:sparsity_curve}` e `\includegraphics{figs/fig1_sparsity_curve.pdf}`
- [x] Tabela 1 referenciada via `\ref{tab:main}` e `\label{tab:main}` no booktabs

## Status pra publicação

`main.tex` compilável. Aguarda Luis rodar Overleaf compile + adicionar PDF ao repo, ou alternativa equivalente. Sem isso bloquear, post LinkedIn pode mencionar "PDF deep dive em breve" se Luis quiser publicar antes.

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
