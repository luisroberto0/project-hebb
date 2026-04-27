"""
Validação Brian2 + STDP (spike-timing dependent plasticity).

Simula 2 neurônios pré-sinápticos disparando em padrões temporais
diferentes pra um neurônio pós-sináptico. A regra STDP deve fortalecer
a sinapse de quem dispara *antes* do pós e enfraquecer a de quem
dispara *depois*.

É o "hello world" da plasticidade local — base da sua pesquisa em
aprendizado one-shot. Se isso roda, Brian2 está OK e o conceito que
sustenta o pilar one-shot está validado na sua máquina.
"""

from __future__ import annotations

import numpy as np

try:
    from brian2 import (
        ms, mV, second, Hz,
        NeuronGroup, Synapses, SpikeGeneratorGroup,
        StateMonitor, SpikeMonitor,
        run, defaultclock, prefs,
    )
except ImportError as e:
    raise SystemExit(
        "Brian2 não instalado. Rode: pip install brian2"
    ) from e

# Brian2 gera código C++ na hora; modo "cython" é o default e funciona
# bem desde que tenha compilador (gcc no Linux, MSVC no Windows).
prefs.codegen.target = "numpy"  # 'numpy' é portátil; troque pra 'cython' depois
defaultclock.dt = 0.1 * ms

print("Brian2 carregado. Configurando experimento STDP...")

# ---------------------------------------------------------------------------
# Estímulos: dois pré-sinápticos
#   - Pré 0 dispara 5ms ANTES do pós (deve sofrer LTP — sinapse fortalece)
#   - Pré 1 dispara 5ms DEPOIS do pós (deve sofrer LTD — sinapse enfraquece)
# ---------------------------------------------------------------------------
N_PAIRS = 60
DURATION = 1000 * ms

pre0_times = np.arange(10, 1000, 16) * ms  # disparos a cada 16 ms
pre1_times = pre0_times + 10 * ms          # 10 ms depois
post_times = pre0_times + 5 * ms           # 5 ms após pre0, 5 ms antes de pre1

pre = SpikeGeneratorGroup(
    2,
    indices=np.concatenate([np.zeros(len(pre0_times)), np.ones(len(pre1_times))]),
    times=np.concatenate([pre0_times, pre1_times]),
)

# Pós sináptico: vamos forçar disparos via SpikeGeneratorGroup também
# pra controle exato do timing
post = SpikeGeneratorGroup(1, indices=np.zeros(len(post_times)), times=post_times)

# ---------------------------------------------------------------------------
# Sinapses com STDP clássico (Song, Miller & Abbott 2000)
# ---------------------------------------------------------------------------
taupre = 20 * ms
taupost = 20 * ms
A_pre = 0.01
A_post = -A_pre * 1.05  # leve assimetria pra estabilidade
wmax = 1.0

stdp_eqs = """
w : 1
dapre/dt  = -apre/taupre  : 1 (event-driven)
dapost/dt = -apost/taupost : 1 (event-driven)
"""

S = Synapses(
    pre, post,
    model=stdp_eqs,
    on_pre="""
    apre += A_pre
    w = clip(w + apost, 0, wmax)
    """,
    on_post="""
    apost += A_post
    w = clip(w + apre, 0, wmax)
    """,
)
S.connect()
S.w = 0.5  # peso inicial igual pros dois

mon = StateMonitor(S, "w", record=True)

print("Simulando 1s com STDP ativo...")
run(DURATION)

w_final = S.w[:]
print()
print(f"Peso pré 0 (dispara ANTES do pós, esperado LTP): {w_final[0]:.4f}  (inicial 0.5000)")
print(f"Peso pré 1 (dispara DEPOIS do pós, esperado LTD): {w_final[1]:.4f}  (inicial 0.5000)")
print()

if w_final[0] > 0.5 and w_final[1] < 0.5:
    print("✅ STDP funcionando: causal fortaleceu, anti-causal enfraqueceu.")
    print("   Brian2 está pronto pra experimentos sérios de plasticidade local.")
elif np.allclose(w_final, 0.5, atol=0.01):
    print("⚠️  Pesos não mudaram. STDP não disparou — confira indices/times dos disparos.")
else:
    print("⚠️  Resultado inesperado. Pode ser parâmetros, mas a stack está rodando.")

# ---------------------------------------------------------------------------
# Salvar gráfico (opcional, comente se rodando headless)
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # backend sem janela
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mon.t / ms, mon.w[0], label="pré 0 (causal, esperado ↑)")
    ax.plot(mon.t / ms, mon.w[1], label="pré 1 (anti-causal, esperado ↓)")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("tempo (ms)")
    ax.set_ylabel("peso sináptico w")
    ax.set_title("STDP: evolução de pesos por timing relativo")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("stdp_validation.png", dpi=120)
    print("\nGráfico salvo em stdp_validation.png")
except Exception as e:
    print(f"\n(Plotagem pulada: {e})")
