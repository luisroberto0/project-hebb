# Marco 5 — WEEKLY-1: escala da plasticidade local (Tiny-ImageNet)

## Setup

ImageNet completo inviável (150GB, rede lenta). Caminho viável: **Tiny-ImageNet** (200 classes, imagens
estilo-ImageNet, resize 32×32 dos parquets HF). Mesma regra/arquitetura SoftHebb do Marco 3, classificador
10→200. Linear-probe (probe-30), seed 0. `scale.py` reaproveita `softhebb_cifar.py`.

## Resultado (seed 0, probe-30)

| Dataset | classes | softhebb | random | margem | vs backprop |
|---|---|---|---|---|---|
| CIFAR-10 | 10 | 80,27 | 68,59 | +11,67 | −6,84 |
| CIFAR-100 | 100 | 55,36 | 43,31 | +12,05 | −7,59 |
| **Tiny-ImageNet** | **200** | **31,67** | 22,36 | **+9,31** | **−9,19** |

(chance Tiny-ImageNet = 0,5%; softhebb 31,67% ≈ **63× chance**.)

## Veredicto — a plasticidade local ESCALA (positivo)

**O SoftHebb escala em nº de classes e em complexidade de imagem, sem colapsar:**
- **A margem sobre random PERSISTE** (+9,31 p.p. em 200 classes de imagens reais) — consistente com os +11,7/+12,0 de CIFAR-10/100. O sinal real do Marco 3 não é específico de CIFAR-10; é **robusto a escala**.
- A acurácia absoluta cai (80→55→32%) com a dificuldade, como esperado — mas continua muito acima de random e de chance.
- O gap para o backprop cresce levemente com a dificuldade (−6,8 → −7,6 → −9,2): o backprop aproveita mais a escala/dados, mas o SoftHebb segue **competitivo** (32% vs 41%, single-pass sem backprop).

**Consistente com o paper SoftHebb** (que reporta 27% em ImageNet-1k, 1000 classes): aqui 32% em 200 classes 32×32. O método tem um teto claro em escala grande (não compete com SOTA), mas **não desmorona** — a margem-sobre-random, que é a evidência de aprendizado genuíno, sobrevive até 200 classes.

## Honestidade

- 1 seed, probe-30, resize 32×32 (degrada Tiny-ImageNet de 64×64). Caracteriza a tendência, não é um número de leaderboard.
- O ponto do Marco 5 era "até onde escala": **escala bem** (margem persiste), com o gap-para-backprop crescendo. Não há reviravolta de controle aqui (o random já é o controle central, e a margem persiste).

## Próximo (roadmap): Marco 6 — hardware neuromórfico

Medir energia real do SoftHebb em silício (Loihi/SpiNNaker via EBRAINS, ou Akida AKD1000 $289). Onde a eficiência local/single-pass/sem-backprop se materializa. **Requer resolver acesso a hardware** (fora da sessão).
