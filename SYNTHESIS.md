# Project Hebb — Síntese da exploração pós-LLM

> Consolidação da jornada (sessões #1–#77b). Documento de capstone: reúne os 5 marcos, a narrativa transversal de k-WTA, e uma avaliação honesta de onde a abordagem bio-inspirada (não) entrega capacidades pós-LLM. Serve de esqueleto de paper, de registro final, ou de base para um pivô — conforme a decisão de rumo.

---

## 1. A missão e a régua

Project Hebb perguntou se uma **arquitetura neuro-inspirada** — plasticidade local, codificação esparsa, dinâmica spiking, idealmente **sem backprop end-to-end** (CONTEXT §1, linha 16) — poderia entregar **capacidades que LLMs não têm**: aprendizado contínuo sem esquecimento, one-shot real, eficiência radical, e raciocínio temporal.

A régua honesta tem duas camadas: (a) atingir a capacidade numericamente, e (b) atingi-la **pelo mecanismo bio-inspirado** (não por backprop convencional disfarçado). A camada (b) é o que separa "arquitetura fundamentalmente nova" de "incrementalismo".

---

## 2. As capacidades — resultados

| Capacidade | Marco | Resultado | Mecanismo bio-inspirado entregou? |
|---|---|---|---|
| One-shot (in-domain) | C3 (#20) | ✅ numérico — ProtoNet+k-WTA 93.10% 5w1s (75% sparsity, −1.45 p.p.) | ❌ usa backprop end-to-end |
| One-shot inédito (cross-domain) | 2-A (#52–66) | ❌ k-WTA effect collapse; C3 cross-domain ~22% vs retreinado 34–50% | ❌ anti-transfer; encoder ≈ random |
| Aprendizado contínuo | 1 (#21–30) | ❌ bio-inspirado ≤ ProtoNet naive (80.65%) | ❌ ProtoNet já é robusto; mecanismos não agregam |
| Eficiência radical | 2-B (#67–70) | ❌ SNN não eficiente em CPU (latência 80–327× pior) | ❌ vantagem só em SynOps teórico, não realizada em von Neumann |
| Raciocínio temporal | 2-C (#71–78) | ⚠️ timing genuíno no SHD, **mas GRU não-spiking supera a SNN (+10.5 p.p.)** | ❌ timing é genérico de recorrência, não do spiking |
| **Plasticidade local SEM backprop** (premissa-mãe) | **3 (SoftHebb)** | ✅ **80.27% CIFAR-10, +11.67 p.p. sobre random, a 6.8 p.p. do backprop** | ✅ **SIM — 1º positivo limpo: pilha 100% local, sinal real, competição essencial** |
| **Continual learning sem esquecer** | **4 (SoftHebb sequencial)** | ⚖️ SoftHebb não esquece (BWT +0.34) e SUPERA backprop-sup (que esquece, −16.78) — **mas o autoencoder backprop-não-sup também não esquece (+4.59)** | ❌ resistência vem do **não-supervisionado**, NÃO da localidade Hebbiana (controle AE desinflou) — MEDIANO |
| **Escala (200 classes)** | **5 (Tiny-ImageNet)** | ✅ softhebb 31.67% / random 22.36% — **margem +9.31 p.p. persiste** (~63× chance) | ✅ o sinal escala; não é específico de CIFAR-10 |
| **Eficiência de treino** | **3/4 (medição)** | ✅ SoftHebb treina features **21× mais rápido** que backprop (33s vs 686s, sem labels/backprop, a −7 p.p.) | ✅ a contribuição ortogonal QUANTIFICADA — eficiência, não capacidade |

**Placar honesto (revisado pós-Marco 3):** os Marcos 1–2 (4 capacidades) deram 3 negativos rigorosos + 1 positivo desinflado pelo controle GRU — **nenhum** deu vantagem competitiva à bio-inspiração, e todos os "sucessos" numéricos usavam backprop. **Mas o Marco 3 mudou o quadro:** atacou a *premissa-mãe* (plasticidade local sem backprop, nunca testada antes) e deu o **1º positivo limpo** — o mecanismo bio-inspirado, isolado, carrega sinal genuíno (sobrevive aos 3 controles que mataram tudo). Ressalva: a margem (+11.67) ficou 3,3 p.p. abaixo do limiar pré-registrado, e o probe final usa backprop (a pilha é local). Não é vitória sobre o estado da arte — é a **prova de conceito da tese**, que faltava.

**Pós-Marco 4 (continual learning):** o SoftHebb sequencial **não esquece** (BWT +0.34) e supera o backprop supervisionado (que sofre catastrophic forgetting, −16.78) — parecia a 1ª vitória da bio-inspiração sobre o backprop. Mas o controle-chave (autoencoder backprop **não-supervisionado**) **também não esquece** (+4.59): a resistência vem do **aprendizado não-supervisionado**, não da localidade Hebbiana. **MEDIANO honesto** — mesmo padrão do GRU no Marco 2-C: o controle adversarial revela a causa real. A contribuição genuína do SoftHebb permanece **ortogonal e agora QUANTIFICADA**: atinge a mesma utilidade/resistência treinando features **21× mais rápido** que o backprop (CIFAR-10: 33s single-pass sem backprop/labels vs 686s/50-épocas, a −7 p.p.) — é vantagem de **eficiência**, não de capacidade exclusiva. **Lição transversal de 4 marcos:** o que parece vantagem bio-inspirada quase sempre se reduz, sob controle, a uma propriedade mais geral (não-supervisionado, recorrência, esparsidade) — mas a bio-inspiração frequentemente a entrega de forma mais *eficiente/local*.

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

### Raciocínio temporal (Marco 2-C) — ⚠️ timing real, mas SNN perde para GRU
SNN recorrente em SHD (Spiking Heidelberg Digits): **71.27%** vs baseline cego ao timing 51.56% = **+19.7 p.p.** (5 seeds, IC95%). Caracterizado em 7 frentes:
- **#73 resolução temporal:** timing *genuíno* **+10.18 p.p.** — mas *upper bound*: em bins=1 a recorrência é inerte, então mistura timing com a recorrência ficando operativa (achado do peer review).
- **#74 latency coding:** a SNN extrai 50.68% só do *onset* (1 spike/canal).
- **#75 k-WTA temporal:** tolerante até **75% de sparsity (−1.50 p.p.)** — paralelo numérico ao C3 espacial in-domain (coincidência sugestiva, 2 pontos).
- **#76–77b generalização SSC:** **fraca** (+4–5 p.p. vs +19.7 no SHD), dataset-específica.
- **#78 controle GRU (decisivo):** um **GRU não-spiking** no mesmo input atinge **79.64% > SNN 69.10%** (+10.5 p.p.). O timing é **genérico de recorrência, não do spiking**; a SNN é uma forma **inferior** de explorá-lo.

**O que parecia o único marco positivo foi desinflado pelo controle adversarial:** a SNN explora timing genuíno, mas (a) isso não é uma vantagem do spiking, e (b) um RNN convencional faz melhor. Não há vantagem competitiva da bio-inspiração — o 2-C junta-se de fato aos 3 negativos.

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

> **Atualização pós-Marco 3 (2026-06-18):** a premissa-mãe — *plasticidade local sem backprop* — foi **finalmente testada limpa** (eixo SoftHebb) e deu o **1º positivo limpo** do projeto: pilha conv treinada só por Hebbiano competitivo local atinge **80.3% CIFAR-10**, com sinal **real** (+11.67 p.p. sobre random) e competição **essencial**, a 6.8 p.p. do backprop. As três razões abaixo (escritas antes) permanecem para os Marcos 1–2; o Marco 3 é a exceção que muda o quadro: o mecanismo bio-inspirado, isolado, **carrega sinal genuíno** quando a regra é competitiva e normalizada (o que o STDP não era).

**Para os Marcos 1–2, a resposta era: provavelmente não, no regime testável.** Três razões:

1. **Não é uma capacidade que falta às LLMs.** O único positivo (timing no SHD/SSC) é classificação de áudio — que ASR/transformers fazem com >90%. A SNN explora timing, mas opera muito abaixo do estado da arte. Não supera nem faz algo único.

2. **Usa backprop.** Todo "sucesso" numérico (C3, 2-C) é surrogate-gradient/SGD — viola a premissa-mãe (plasticidade local sem backprop). Quando o mecanismo bio-inspirado foi *isolado* (STDP #1–13, termo Hebbian #18, encoder vs random #54), ele não carregou o sinal.

3. **A única vantagem estrutural única — eficiência event-driven — foi refutada** em hardware comum (2-B). Ela só existiria em silício neuromórfico (Loihi/SpiNNaker/Akida), que o projeto não tem e cujo acesso é restrito (EBRAINS gated por afiliação; Akida exige desktop Linux+PCIe).

**O que o projeto testou no Marco 3** e onde a tese encontrou tração: **plasticidade local sem backprop** (SoftHebb) — e funciona (ver atualização no topo da §5). **O que ainda NÃO foi testado** e onde a tese pós-LLM poderia ir mais longe: plasticidade local *verdadeiramente online* (single-pass streaming, aprendizado contínuo) e **hardware neuromórfico real** (onde a eficiência se materializa). Apostas ainda maiores, mas o Marco 3 mostra que a direção não é um beco — pela primeira vez.

---

## 6. O valor real do projeto

- **Caracterizar com rigor onde a abordagem bio-inspirada falha e por quê** — 3 achados negativos defensáveis nas capacidades pós-LLM, cada um com mecanismo documentado.
- **Provar a premissa-mãe** (Marcos 3-5): plasticidade local, sem backprop, aprende features genuínas e úteis (80% CIFAR-10), que escalam (200 classes) e resistem a forgetting — sobrevivendo aos controles adversariais que mataram tudo antes.
- **Descobrir, e quantificar, qual é de fato a contribuição:** não é capacidade nova (o controle do autoencoder mostrou que a robustez vem do não-supervisionado) — é **eficiência**: as mesmas features, **21× mais rápido** que o backprop, sem labels, single-pass, local.
- **Disciplina metodológica exemplar:** predições registradas antes, random baselines, critérios literais, peer review adversarial, e — o fio condutor — **o controle contra a própria hipótese favorita**, que vez após vez separou o desejado do real.

No espírito do projeto (*falhas bem-documentadas valem mais que sucessos superficiais*): um corpo de evidência negativa rigoroso **+ uma tese positiva honestamente delimitada** — a bio-inspiração não entrega superpoderes pós-LLM, mas entrega features competitivas por uma fração do custo. Uma contribuição honesta e original de um pesquisador independente.

---

## 7. Estado e próximos rumos possíveis (decisão do autor)

Exploração das 4 capacidades **completa** + premissa-mãe **provada** (pivot Marcos 3-5 + eficiência). Estado atual (2026-06):
- **🔵 PUBLICANDO a jornada** (decisão do Luis) — rascunho do post LinkedIn (arco de 3 atos) em `writeups/linkedin-jornada-pt.md`, pronto para revisão. É o desfecho mais forte e original do projeto.
- **⬜ Marco 6 (futuro) — hardware neuromórfico:** medir a *energia real* do SoftHebb em silício (Loihi/SpiNNaker via EBRAINS, ou Akida) — a validação física da tese de eficiência. Aguarda o Luis resolver o acesso a hardware.
- Os papers anteriores (`paper_c3`, `paper_marco2a`, `paper_marco2c`) permanecem como artefatos por-marco.

Artefatos: `paper_c3/`, `paper_marco2a/`, `experiment_0{1..5}/`, este `SYNTHESIS.md`. Tudo reproduzível e committado.
