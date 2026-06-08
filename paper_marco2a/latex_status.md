# LaTeX status — paper Marco 2-A

> Status: `main.tex` consolidado em #63. Validação estrutural ✅. Compilação local pendente (sem toolchain).

## Estado

`main.tex` consolida as 6 seções + abstract + 3 figuras num único arquivo (padrão inline, igual ao `paper_c3/main.tex`). Bibliography via `\bibliography{refs}` + `refs.bib` (15 entradas, todas citadas).

| Componente | Estado |
|---|---|
| Preamble (article 11pt, geometry, natbib, authblk, graphicx, booktabs) | ✅ |
| Abstract | ✅ ~170 palavras |
| §1 Introduction | ✅ |
| §2 Background | ✅ |
| §3 Method | ✅ |
| §4 Experiments (Table 1 + Table 2 + Fig 1/2/3) | ✅ |
| §5 Discussion | ✅ |
| §6 Conclusion | ✅ |
| Bibliography (`refs.bib`, 15 entradas) | ✅ |

## Validação estrutural (sem compilar)

Rodado `c:\tmp\validate_tex.py` (reproduzível) sobre `main.tex` + `refs.bib`:

- **Citações:** 15 keys citadas, todas com entrada no `.bib` (0 faltando).
- **Cross-refs:** 32 `\label`, 6 `\ref` distintos, todos resolvem.
- **Ambientes:** 13 `\begin` / 13 `\end`, balanceados (document, abstract, table×2, tabular×2, figure×3, equation, itemize, enumerate).
- **Figuras:** 3 `\includegraphics`, todos os PDFs existem em `figs/`.
- **Chaves:** 224 `{` / 224 `}`, balanceadas.

Isso não substitui uma compilação real (não captura erros de pacote, overfull boxes, ou referências de página), mas elimina as classes de erro mais comuns antes do Overleaf.

## Compilação

`pdflatex`/`bibtex` **não disponíveis** no ambiente de dev (Windows), mesmo caso do paper C3.

**Opção recomendada — Overleaf (~10 min, sem instalação):**
1. Novo projeto em [overleaf.com](https://www.overleaf.com).
2. Upload de `main.tex`, `refs.bib`, e a pasta `figs/` (os 3 PDFs).
3. Compilar (Overleaf roda `pdflatex → bibtex → pdflatex → pdflatex` automaticamente).
4. Verificar: abstract, Tabelas 1–2 com booktabs, Figuras 1–3 renderizadas, citações `(Autor, ano)` resolvidas (sem `(?)`).

**Alternativas:** instalar MiKTeX/TeX Live local; ou Docker `texlive/texlive` com `latexmk -pdf main.tex`.

## Checklist pós-compilação (verificar no PDF)

- [ ] Sem `(?)` em citações (todas no `.bib`).
- [ ] Figuras 1–3 aparecem nos pontos certos (§4.2, §4.3, §4.4) e legíveis a 300 DPI.
- [ ] Tabelas 1–2 alinhadas (booktabs `\toprule/\midrule/\bottomrule`).
- [ ] Word count do corpo dentro do alvo (~3500–4500; soma das seções ≈ 3700).
- [ ] Trocar `\documentclass{article}` pelo style file oficial do workshop se/quando houver venue (decisão #66).
