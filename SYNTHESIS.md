# Project Hebb — Síntese da exploração pós-LLM

> Consolidação da jornada (sessões #1–#77b). Documento de capstone: reúne os 5 marcos, a narrativa transversal de k-WTA, e uma avaliação honesta de onde a abordagem bio-inspirada (não) entrega capacidades pós-LLM. Serve de esqueleto de paper, de registro final, ou de base para um pivô — conforme a decisão de rumo.

---

## 1. A missão e a régua

Project Hebb perguntou se uma **arquitetura neuro-inspirada** — plasticidade local, codificação esparsa, dinâmica spiking, idealmente **sem backprop end-to-end** (CONTEXT §1, linha 16) — poderia entregar **capacidades que LLMs não têm**: aprendizado contínuo sem esquecimento, one-shot real, eficiência radical, e raciocínio temporal.

A régua honesta tem duas camadas: (a) atingir a capacidade numericamente, e (b) atingi-la **pelo mecanismo bio-inspirado** (não por backprop convencional disfarçado). A camada (b) é o que separa "arquitetura fundamentalmente nova" de "incrementalismo".

---

## 2. As 4 capacidades — resultados

| Capacidade | Marco | Resultado | Mecanismo bio-inspirado entregou? |
|---|---|---|---|
| One-shot (in-domain) | C3 (#20) | ✅ numérico — ProtoNet+k-WTA 93.10% 5w1s (75% sparsity, −1.45 p.p.) | ❌ usa backprop end-to-end |
| One-shot inédito (cross-domain) | 2-A (#52–66) | ❌ k-WTA effect collapse; C3 cross-domain ~22% vs retreinado 34–50% | ❌ anti-transfer; encoder ≈ random |
| Aprendizado contínuo | 1 (#21–30) | ❌ bio-inspirado ≤ ProtoNet naive (80.65%) | ❌ ProtoNet já é robusto; mecanismos não agregam |
| Eficiência radical | 2-B (#67–70) | ❌ SNN não eficiente em CPU (latência 80–327× pior) | ❌ vantagem só em SynOps teórico, não realizada em von Neumann |
| Raciocínio temporal | 2-C (#71–77b) | ✅ **positivo modesto** — SHD timing +19.7 p.p. (rec 71.27%) | ✅ a dinâmica recorrente explora o timing (único caso) |

**Placar honesto:** das 4 capacidades, **3 com achado negativo rigoroso** + **1 positivo modesto** (temporal). O único "positivo numérico" extra (one-shot in-domain, C3) usa backprop — não satisfaz a camada (b).

---

## 3. Os achados, em detalhe

### One-shot in-domain (C3) — ✅ numérico, ❌ mecanístico
ProtoNet (CNN-4) + k-WTA esparso atinge 93.10% em Omniglot 5-way 1-shot com 75% de sparsity (custo de −1.45 p.p. vs ProtoNet denso). Mas treina por SGD/backprop end-to-end. Re-framing honesto: é *metric learning convencional + esparsidade biológica*, não "STDP funciona pra few-shot". Publicado como `paper_c3` (LinkedIn, decisão #36).

### One-shot inédito / cross-domain (Marco 2-A) — ❌
Encoder C3 (Omniglot, congelado) transferido para CUB-200 colapsa: todas as sparsities k-WTA ficam em 21.7–22.2% (≈ ruído acima de chance), enquanto ProtoNet retreinado em CUB chega a 34–50%. Três achados: (1) **k-WTA effect collapse** — o spread in-domain de 3.78 p.p. vira 0.52 p.p. cross-domain; (2) **anti-transfer** — encoder treinado ≈ random; (3) **Pixel kNN domina** os encoders. Paper draft completo + peer review, **arquivado** (#66).

### Aprendizado contínuo (Marco 1) — ❌
4 abordagens (naive, plasticidade meta-aprendida, trace STDP, kitchen-sink) em Split-Omniglot, todas ≤ baseline naive ProtoNet (80.65%). Achado mecanístico: ProtoNet sem classifier head é **inerentemente robusto** a forgetting; mecanismos bio-inspirados não têm o que melhorar. Encerrado (#30).

### Eficiência radical (Marco 2-B) — ❌
SNN-LIF + k-WTA temporal vs MLP denso em Fashion-MNIST, métrica dupla (SynOps + latência CPU). Nenhuma config atinge acc −2 p.p. **E** SynOps ≥5× menores **E** latência ≤ denso. Trade-off acc↔SynOps íngreme; **latência CPU 80–327× pior sempre**; inferência event-driven (sparse) é até *mais* lenta que o runtime denso (overhead de indexação > matmul BLAS). Conclusão: a eficiência neuromórfica é **co-design hardware-algoritmo** (silício dedicado), não se realiza em von Neumann. Encerrado (#70).

### Raciocínio temporal (Marco 2-C) — ✅ positivo modesto
SNN recorrente em SHD (Spiking Heidelberg Digits): **71.27%** vs baseline cego ao timing 51.56% = **+19.7 p.p.** (5 seeds, IC95%). Caracterizado em 6 frentes:
- **#73 resolução temporal:** timing *genuíno* **+10.18 p.p.** (controlando arquitetura — ~metade do +19.7 bruto era LIF+BN, não timing).
- **#74 latency coding:** a SNN extrai 50.68% só do *onset* (1 spike/canal).
- **#75 k-WTA temporal:** tolerante até **75% de sparsity (−1.50 p.p.)** — paralelo quase exato ao C3 espacial in-domain.
- **#76–77b generalização SSC:** **fraca, positiva, estável** (+4–5 p.p. vs +19.7 no SHD) — não era subtreino (platôou com 60 epochs). Magnitude dataset-específica.

**Único marco onde o mecanismo bio-inspirado (dinâmica recorrente) genuinamente agrega.** Mas modesto: a SNN faz ~30% em SSC onde redes convencionais fazem >90%.

---

## 4. Achado transversal — o comportamento do k-WTA

Emergente, não planejado — costura três marcos:

| Regime | Eixo | k-WTA a 75% sparsity |
|---|---|---|
| In-domain, Omniglot (C3) | espacial (embedding) | tolerante: **−1.45 p.p.** |
| In-domain, SHD (2-C) | temporal (timestep) | tolerante: **−1.50 p.p.** |
| Cross-domain (2-A) | espacial | **colapsa** — sparsity vira ruído |
| Esparsidade extrema (2-C, k≤8 / >96%) | temporal | **colapsa** — converge ao baseline |

**A esparsidade k-WTA é compatível in-domain em qualquer eixo (espaço ou tempo) com o mesmo custo (~1.5 p.p. a 75%), mas é frágil sob shift de domínio ou compressão extrema.** É a contribuição mais original e coesa do projeto.

---

## 5. Avaliação honesta — isso é um caminho pós-LLM?

**Provavelmente não, com tração — no regime testável.** Três razões:

1. **Não é uma capacidade que falta às LLMs.** O único positivo (timing no SHD/SSC) é classificação de áudio — que ASR/transformers fazem com >90%. A SNN explora timing, mas opera muito abaixo do estado da arte. Não supera nem faz algo único.

2. **Usa backprop.** Todo "sucesso" numérico (C3, 2-C) é surrogate-gradient/SGD — viola a premissa-mãe (plasticidade local sem backprop). Quando o mecanismo bio-inspirado foi *isolado* (STDP #1–13, termo Hebbian #18, encoder vs random #54), ele não carregou o sinal.

3. **A única vantagem estrutural única — eficiência event-driven — foi refutada** em hardware comum (2-B). Ela só existiria em silício neuromórfico (Loihi/SpiNNaker/Akida), que o projeto não tem e cujo acesso é restrito (EBRAINS gated por afiliação; Akida exige desktop Linux+PCIe).

**O que o projeto NÃO testou** e onde a tese pós-LLM verdadeira ainda poderia morar: **plasticidade local online, sem backprop** (o eixo B nunca atacado), e **hardware neuromórfico real** (onde a eficiência se materializa). Ambos são apostas muito maiores que um side-project de 5h/semana absorveu.

---

## 6. O valor real do projeto

Não foi achar o caminho pós-LLM. Foi:

- **Caracterizar com rigor onde a abordagem bio-inspirada falha e por quê** — 3 achados negativos defensáveis, cada um com mecanismo documentado.
- **Um achado positivo honesto e bem-delimitado** (timing) — com a magnitude do confound arquitetural *medida*, não assumida.
- **Uma narrativa transversal original** (k-WTA tolerante in-domain em espaço e tempo, frágil cross-domain).
- **Disciplina metodológica exemplar:** predições registradas antes, random baselines, critérios literais, peer review adversarial, e a recusa consistente de inflar resultados.

No espírito do próprio projeto: *falhas bem-documentadas valem mais que sucessos superficiais*. Este é um corpo de evidência negativa rigoroso + um positivo modesto — uma contribuição honesta sobre os limites da bio-inspiração como caminho pós-LLM no regime acessível a um pesquisador independente.

---

## 7. Estado e próximos rumos possíveis (decisão do autor)

Exploração das 4 capacidades **completa**. Caminhos, sem prescrição:
- **Publicar a jornada** — paper/post sobre a exploração das 4 capacidades + a narrativa de k-WTA. Honesto e original.
- **Consolidar e fechar** — Project Hebb como exploração documentada, encerrada num estado completo.
- **Pivotar** — para a aposta de fato pós-LLM (plasticidade sem backprop, ou hardware neuromórfico). Aposta maior.
- **Pausar** — decidir com calma.

Artefatos: `paper_c3/`, `paper_marco2a/`, `experiment_0{1..5}/`, este `SYNTHESIS.md`. Tudo reproduzível e committado.
