# Landscape — aprendizado SEM backprop (scoping do próximo marco)

> Pesquisa de scoping (pós-#78, pivot decidido pelo Luis para "plasticidade/aprendizado sem backprop"). Mapeamento de 7 famílias modernas via workflow de pesquisa paralela + síntese. Fundamenta a escolha de eixo do próximo marco. **Nenhum código ainda — eixo + critério dependem da decisão do Luis.**

## Contexto: a premissa-mãe e os becos já mapeados

A premissa-mãe do Project Hebb (CONTEXT §1): *"plasticidade local diferenciável — regras de update que operam só com sinais locais (pré, pós, modulação) e podem rodar online, sem backprop end-to-end."* Nunca foi testada limpa. Dois becos **já descartados**:
- **STDP biofísico puro** (Exp 01, #1–13) → falhou: saturação, colapso de filtros; ablação #10 mostrou ~80% do sinal era *arquitetural* (pesos random saturados davam quase o mesmo).
- **Differentiable plasticity / meta-Hebbian** (C2, #17–19) → 64%, mas o outer-loop É backprop, e o termo Hebbian `A·pre·post` mostrou-se *dispensável* (#18).

**As 3 armadilhas que mataram tudo até agora** (o filtro de qualquer eixo novo): (1) sinal arquitetural disfarçado de aprendizado; (2) o componente "novo" ser dispensável; (3) backprop disfarçado.

## As 7 famílias (fit_score = adequação como próximo marco do Hebb)

| Fit | Família | Teto empírico (sem backprop) | Viabilidade 4070 | Evita os becos? |
|---|---|---|---|---|
| **8** | **Hebbiano competitivo (SoftHebb)** | CIFAR-10 80.3%, STL-10 76.2%, ImageNet 27.3% (probe) | Alta — single-pass, repo oficial PyTorch | ✅ regra local fechada (zero autograd); WTA não-dispensável por construção |
| **8** | **Forward-Forward (Hinton 2022) / SCFF** | CIFAR-10 ~80% (SCFF), Tiny-ImageNet funcional | Alta — memória < backprop | ✅ sinal contrastivo não-dispensável; ⚠️ update intra-camada usa autograd local |
| **7** | **Equilibrium Propagation** | CIFAR-10 93.9% (ResNet13) — **menor gap vs backprop** | Parcial — fases de equilíbrio 3-10× mais lentas | ⚠️ provável "backprop disfarçado via simetria Wᵀ" |
| 6 | Predictive Coding | VGG-7 ≈ backprop; degrada com profundidade | Parcial — relaxação 10-100× mais lenta | ⚠️ PC-IL converge ao gradiente do backprop |
| 6 | Perturbation / zeroth-order | Estimador puro colapsa acima de MNIST (variância) | Alta p/ versão mínima | ✅ regra de 3 fatores local+online, mas teto baixo |
| 5 | Local self-supervised (Greedy InfoMax) | **ImageNet probe ~70%, gap ~1pp** (mais escala) | Baixa — ImageNet inviável no 4070 | ⚠️ usa backprop curto intra-bloco |
| 3 | Feedback Alignment / DFA | Colapsa em CNN profunda / ImageNet | Alta | ❌ erro global projetado por matriz random — menos local que STDP |

## Os 3 eixos candidatos (com critério literal, estilo Hebb)

### ⭐ Eixo 1 — SoftHebb / Hebbiano competitivo (RECOMENDADO)
Pilha conv treinada **só por regra Hebbiana competitiva local** (pré × pós-competido, `requires_grad=False`, zero backprop), avaliada por linear-probe congelado. Reaproveita o repo oficial SoftHebb + a expertise k-WTA do C3.
- **Pergunta:** a representação Hebbiana competitiva carrega sinal REAL — supera a mesma arquitetura com pesos random congelados (+ mesma WTA + mesmo probe) por margem grande — ou o ganho é arquitetural como no STDP?
- **Critério:** SUCESSO = probe CIFAR-10 ≥75% E bate random+WTA por ≥15 p.p. E ≤15 p.p. do backprop e2e. MEDIANO = 65-75% e 8-15 p.p. FALHA = <65% OU <8 p.p. (3 seeds, IC95%, HP fixos antes).
- **Predição honesta:** Mediano provável, inclinado a Sucesso. SoftHebb reporta ~80% bruto; a incógnita é o random-control. ~55% Sucesso limpo / ~35% Mediano / ~10% random come tudo (= STDP de novo, negativo forte).
- **Baseline + controle obrigatório:** backprop e2e mesma arquitetura (teto) + **(a) random congelado + WTA + probe** (a lição do STDP — rodar ANTES de concluir) + **(b) WTA desligado**.
- **Esforço:** 4-6 sessões. **Risco:** random comer o sinal (mitigado tornando o controle gatilho de fechamento antes do experimento principal).

### Eixo 2 — Forward-Forward / SCFF
Objetivo local contrastivo por camada (goodness pos/neg), sem gradiente global, linear-probe + ablação do contraste (embaralhar pos/neg).
- **Critério:** SUCESSO = probe ≥78% E bate random por ≥15 p.p. E acc despenca ≥20 p.p. com pos/neg embaralhados. MEDIANO = 68-78%. FALHA = <68% OU shuffle quase não muda.
- **Predição:** ~45% Sucesso (com SCFF), ~45% Mediano. **Esforço:** 5-7 sessões. **Asterisco honesto:** update intra-camada é autograd de 1 camada — elimina o backward GLOBAL, não o local; greedy não é online estrito.

### Eixo 3 — Equilibrium Propagation
Convnet energy-based pequeno, com o controle adversarial central: **cosine-similarity entre Δw do EP e o gradiente do backprop** na mesma rede.
- **Critério:** SUCESSO = bate controles (fase-livre-só, random+dinâmica) E ≤5 p.p. do backprop; o cosine DECIDE o enquadramento (alto = "backprop disfarçado", negativo honesto; baixo+paridade = positivo raro). FALHA = não bate os controles.
- **Predição:** paridade numérica MAS cosine provavelmente alto = "EP = backprop com passos extras via Wᵀ". Negativo honesto valioso, não confirma a premissa. **Esforço:** 6-9 sessões (fases de equilíbrio caras — já foi tiro no pé no 2-B).

## Leitura honesta (recomendação)

**O teste mais limpo e menos auto-enganoso é o Eixo 1 (SoftHebb).** Único onde: (a) o update é regra local fechada de verdade (zero autograd, nada pra disfarçar); (b) o termo de aprendizado não é dispensável por construção — desligar a WTA literalmente vira random features, então o controle de ablação e o controle random são a MESMA pergunta, binária e honesta; (c) o controle decisivo (random+WTA+probe) é exatamente o que matou o STDP, mas a regra agora normaliza (Oja) e a competição está na dinâmica (não penalidade pós-hoc), corrigindo os dois bugs do Exp-01 — então tem chance GENUÍNA de passar, e se não passar é um negativo forte e definitivo sobre "plasticidade local pura não carrega sinal".

FF é segundo (quase empatado), mais escalável mas meio passo menos puro. EP dá o menor gap numérico mas a predição é que revele backprop disfarçado. Evitar FA/DFA (nem testa a tese) e desconfiar de PC / block-local SSL (backprop disfarçado / só escala com backprop intra-bloco).

**Fiel ao padrão do projeto:** rodar o random-control ANTES do experimento principal e registrar a predição por escrito. Um negativo honesto aqui vale mais que um positivo inflado — fecharia a porta da "plasticidade local pura" com evidência, não com narrativa.
