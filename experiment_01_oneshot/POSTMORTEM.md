# POSTMORTEM — Experimento 01

> Preencher SOMENTE se outcome em `RESULTS.md` for **falha** (5w1s < 70%).
> Análise diagnóstica das causas e próximos passos. Não inflar — uma
> página é mais útil que cinco.

## Resumo

- **Acurácia 5-way 1-shot atingida:** __% (alvo: ≥ 90%, mínimo: ≥ 70%)
- **Acurácia 20-way 1-shot atingida:** __% (alvo: ≥ 70%)
- **Hipótese principal foi rejeitada?** [sim / parcialmente / não]

## Diagnósticos

### 1. Filtros aprendidos são visualmente coerentes?

[Examinar `logs/run_*/figs/filters_layer1.png` e `filters_layer2.png`]

- Layer 1: [coerentes / ruído / outro padrão]
- Layer 2: [coerentes / ruído / outro padrão]

Se filtros são ruído: problema é STDP (regra ou hiperparâmetros).
Se coerentes: problema é a jusante (memória, embedding, classificação).

### 2. Distribuição de pesos é bimodal?

[Examinar `logs/run_*/figs/weight_hist.png`]

- Layer 1: [bimodal / unimodal / saturada num extremo]
- Layer 2: [bimodal / unimodal / saturada num extremo]

Não-bimodal indica STDP não convergiu — explorar mais epochs ou ajustar
A_pre/A_post.

### 3. Embeddings clusterizam por classe?

[Examinar `logs/run_*/figs/tsne.png`]

- Separação visual: [forte / moderada / nenhuma]

Sem clustering = features pós-STDP não são discriminativas. O problema
não é a memória Hopfield, é upstream.

### 4. Memória Hopfield satura?

- β atual: __ (default 8.0)
- Recall correto em queries idênticas a support? [sim / não]

Se memória não recupera nem queries idênticas, é bug na implementação.
Se recupera idênticas mas falha em variações, β ou normalização precisam
ajuste.

## Causa raiz hipotética

[ESCREVER UMA OU DUAS FRASES SECAS. Exemplos:]

- "STDP convergiu mas embeddings têm dimensionalidade efetiva baixa
  demais — todos os clusters colapsam em uma região do espaço."
- "Codificação Poisson rate é muito ruidosa pra Omniglot que é binário —
  trocar pra temporal coding (TTFS)."
- "Lateral inhibition forte demais — só uma fração dos filtros aprende
  algo, resto fica ruído."

## Próxima iteração

3 ações concretas a testar (ordenadas por custo):

1. __
2. __
3. __

Documentar resultados em `WEEKLY-{N}.md` separado e revisar este
postmortem.
