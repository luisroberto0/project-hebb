# Marco 3 — WEEKLY-1: SoftHebb em CIFAR-10 (primeiro aprendizado local sem backprop)

## Setup

Eixo SoftHebb (Hebbiano competitivo local, ICLR 2023). Regra fiel à implementação oficial single-file
(`demo.py` do repo NeuromorphicComputing/SoftHebb): conv Hebbian com soft-WTA (softmax + anti-Hebbian
para perdedores), regra de Oja (Δw = yx − yu·w), `requires_grad=False` na pilha (ZERO backprop nas
features), arquitetura de 3 camadas (96/384/1536), linear-probe supervisionado (backprop só no
classificador). CIFAR-10 dos parquets HF (toronto.edu estava a ~20 KB/s; HF a 30 MB/s) → `cifar10.npz`.

**Critério literal (fixado antes):** SUCESSO = probe ≥75% E supera random+WTA por ≥15 p.p. E ≤15 p.p.
do backprop. MEDIANO = 65–75% e 8–15 p.p. FALHA = <65% OU <8 p.p.

## Resultado principal (3 seeds, probe-epochs=50)

| Modo | acc | vs critério |
|---|---|---|
| **softhebb** (Hebbiano local) | **80.27% ±0.10** | ≥75% ✓ |
| **random** (controle-chave: pesos random + WTA + probe) | 68.59% ±0.33 | — |
| **backprop** (teto, mesma macro-arquitetura e2e) | 87.11% ±0.27 | — |
| **wta_off** (Hebbiano SEM competição) | 43.22% ±0.85 | colapsa |

**Deltas (bootstrap IC95%):**
- **softhebb − random = +11.67 p.p.** IC95[+11.22, +12.00] — sinal REAL, mas **MEDIANO** (8–15), não cruza os ≥15 do SUCESSO pleno.
- **softhebb − backprop = −6.84 p.p.** IC95[−7.19, −6.54] — fica a só ~7 p.p. do teto.
- **softhebb − wta_off = +37.04 p.p.** — a competição é o motor.

## Leitura honesta

**Este é o achado mais positivo do projeto — e qualitativamente o oposto do STDP.** Três fatos:

1. **A plasticidade Hebbiana competitiva carrega sinal GENUÍNO.** +11.67 p.p. sobre pesos random (IC apertado) — diferente do STDP (#10 do Exp 01), onde ~80% do sinal era arquitetural (random saturado dava quase tudo). Aqui o random dá 68.6% e o Hebbian *aprende* +11.67 a mais. Sinal real, não-arquitetural.

2. **A competição (soft-WTA) é o motor e NÃO é dispensável.** Desligá-la (wta_off) colapsa para 43.2% — **abaixo até do random (68.6%)**: sem competição, a regra Hebbiana não só não ajuda, atrapalha. Contraste com o C2 (differentiable plasticity), onde o termo Hebbian era dispensável (#18). Aqui o mecanismo é essencial por construção.

3. **Reproduz o paper (80.3%) e fica a 6.8 p.p. do backprop** — competitivo para aprendizado sem gradiente global.

**Veredicto literal:** MEDIANO-forte. 2 das 3 sub-condições em nível SUCESSO (acc ≥75% ✓, ≤15 p.p. do backprop ✓); a margem sobre random (+11.67) ficou em MEDIANO. A predição registrada (10–20 p.p.) acertou. **Não é backprop disfarçado** (regra local fechada, zero autograd na pilha), **não é arquitetural** (bate random por +11.67), **não é dispensável** (wta_off colapsa) — sobrevive aos 3 controles que mataram tudo até agora.

## Tentativa de SUCESSO pleno — via mais treino FALHOU (achado honesto)

Hipótese: mais passadas Hebbianas → pilha melhor → margem cresce (random fica fixo em 68.6%). **Refutado:**

| unsup-epochs | softhebb acc | margem vs random |
|---|---|---|
| 1 | 80.32% | +11.67 |
| 3 | 79.75% | +11.16 |
| 5 | 78.35% | +9.76 |
| 10 | 77.24% | +8.65 |

Mais treino **PIORA** — confirma que o SoftHebb é genuinamente **single-pass** (a regra satura/degrada com repetição; é o design do paper). A margem *encolhe*, não cresce. A via natural de empurrar não cruza os +15 p.p.

**Segunda via — ZCA whitening (setup Hebbian canônico, Coates 2011 / SoftHebb original) — também FALHOU:** com whitening (probe-50, seed 0), softhebb 80.32→79.02 (piora) e random 68.33→**71.96** (melhora) → margem cai para **+7.06**. O whitening ajuda o *random* mais que o Hebbiano (descorrelacionar o input fortalece features random). Esgotadas as duas vias principled de empurrar a margem.

## Veredicto final (honesto)

**SUCESSO em 2 das 3 sub-condições; MEDIANO na 3ª.** O critério literal exigia probe ≥75% (✓ 80.27%), ≤15 p.p. do backprop (✓ −6.84), E margem ≥15 p.p. sobre random (✗ +11.67, ficou na faixa MEDIANO 8–15). Não é SUCESSO *pleno* pela letra — a margem ficou **3,3 p.p. abaixo** do limiar que eu havia fixado.

**Mas é, sem ambiguidade, o primeiro positivo LIMPO da premissa-mãe do projeto.** Por quê o limiar não foi cruzado: o trade-off é estrutural — a arquitetura larga (1536 canais) que dá ao SoftHebb seus 80,3% também dá ao *random* features fortes (68,6%), comprimindo a margem. Arquitetura mais estreita aumentaria a margem mas derrubaria a acc <75%. O +11,67 com 80,3% é perto do ótimo do trade-off; não há via honesta óbvia para ter ambos (acc ≥75% E margem ≥15) nesta família — e forçar arquiteturas até cruzar um limiar arbitrário seria p-hacking, contra o ethos.

**Leitura científica:** plasticidade Hebbiana competitiva local, sem backprop, **aprende uma representação genuinamente útil** (80,3% CIFAR-10, a 6,8 p.p. do backprop), com sinal **real** (+11,67 sobre random, não-arquitetural — oposto do STDP) e **mecanismo essencial** (wta_off colapsa — oposto do C2). Sobrevive aos 3 controles que mataram tudo. É um **sucesso qualitativo da tese**, com a margem numérica honestamente reportada logo abaixo do meu limiar.
