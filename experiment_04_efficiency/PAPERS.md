# PAPERS — Marco 2-B (eficiência radical: inferência event-driven)

> Lit review #68. Foco: o que a literatura diz sobre *onde* a eficiência neuromórfica existe — e por que o achado do projeto (#69: SNN não ganha em CPU) é consistente com ela.
> Nota: números marcados "(≈, ver paper)" são de memória e devem ser conferidos antes de qualquer citação formal. Conceitos qualitativos são confiáveis.

---

## Síntese (o que importa pro marco)

A eficiência neuromórfica é uma propriedade de **co-design hardware-algoritmo**, não do algoritmo SNN isolado. Em silício neuromórfico (TrueNorth, Loihi), a comunicação é **event-driven em hardware**: energia só é gasta quando há um spike, e a esparsidade vira economia real. Em hardware **von Neumann** (CPU/GPU) com framework denso, essa esparsidade tem que ser *emulada* — e a emulação (indexação esparsa, loop sobre timesteps) custa **mais** que um matmul denso otimizado (BLAS), conforme medimos em #69b. Logo: a contagem teórica de SynOps subestima o custo real de uma SNN em CPU; a vantagem só se materializa no substrato certo.

Isso transforma o achado negativo do projeto em um achado **alinhado com a literatura**, não em contradição: ninguém afirma que SNN rate-coded em CPU é eficiente; a promessa neuromórfica é sempre condicionada ao hardware event-driven.

---

## Papers core

### 1. Merolla et al. 2014 — TrueNorth (Science)
**"A million spiking-neuron integrated circuit with a scalable communication network and interface."**
- Chip neuromórfico da IBM: ~1M neurônios, ~256M sinapses, comunicação event-driven (spikes como pacotes).
- Consumo na ordem de dezenas de mW (≈70 mW, ver paper) para cargas spiking — ordens de magnitude abaixo de CPU/GPU equivalentes.
- **Relevância:** estabelece o **SynOp** (operação sináptica disparada por evento) como unidade de energia, e mostra que a eficiência vem de *não computar* neurônios silenciosos — em silício. É a vantagem que o probe #69b NÃO conseguiu reproduzir em CPU.

### 2. Davies et al. 2018 — Loihi (IEEE Micro)
**"Loihi: A Neuromorphic Manycore Processor with On-Chip Learning."**
- Chip da Intel: cores neuromórficos, plasticidade on-chip, roteamento event-driven.
- Ganhos de energia/latência para cargas **esparsas e naturalmente spiking** (otimização, inferência esparsa) vs CPU/GPU.
- **Importante e honesto:** o próprio paper/linha reconhece que a vantagem é regime-dependente — para cargas densas e feedforward simples (ex: classificação MNIST-like rate-coded), a vantagem **encolhe ou desaparece**. Isso é exatamente o regime do Marco 2-B → coerente com a Falha medida.

### 3. Neftci, Mostafa & Zenke 2019 — Surrogate Gradient Learning (IEEE Signal Processing Magazine)
**"Surrogate Gradient Learning in Spiking Neural Networks."**
- Como treinar SNNs por BPTT apesar do spike ser não-diferenciável: substitui a derivada do degrau por uma função surrogate suave (ex: fast-sigmoid).
- **Relevância:** é o método de treino usado em `efficiency_bench.py` (`surrogate.fast_sigmoid`). Nota honesta do marco: surrogate-gradient **é backprop** — por isso o eixo A mede eficiência de *inferência*, não treino-sem-backprop.

### 4. Xiao, Rasul & Vollgraf 2017 — Fashion-MNIST (arXiv:1708.07747)
**"Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms."**
- Drop-in replacement do MNIST: 28×28 grayscale, 10 classes (roupas), mesma estrutura (60k/10k).
- Mais discriminativo que MNIST (não satura ~99%), expondo o trade-off acurácia↔eficiência. **Razão da escolha** do dataset do marco.

---

## Implicação para a régua do critério (5× SynOps)

A literatura de hardware neuromórfico não promete vantagem em CPU; promete em silício dedicado. Portanto a régua "SynOps ≥5× menores **E** latência CPU ≤ denso" é, em retrospecto, uma régua que **a física do von Neumann praticamente impede** de satisfazer com SNN rate-coded — o que reforça (não enfraquece) o valor do achado negativo: documenta o limite com números, em vez de assumi-lo.

Um possível **Marco 2-B.2** (se Luis quiser): estimar energia em hardware neuromórfico via proxy (SynOps × E_SynOp vs MACs × E_MAC, com fatores E_SynOp/E_MAC de Merolla/Davies — a verificar), caracterizando *onde* a SNN ganharia. Não demonstra eficiência radical em CPU (a régua), mas mapeia a fronteira hardware.

---

## A verificar antes de qualquer uso formal

- Número exato de consumo do TrueNorth (≈70 mW?) e densidade de neurônios/sinapses.
- Razão E_SynOp / E_MAC citável (ordem de magnitude conhecida, valor exato depende do nó tecnológico).
- Claims quantitativos de speedup/energia do Loihi por carga (são regime-específicos no paper).
