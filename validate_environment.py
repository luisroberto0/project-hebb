"""
Validação completa do ambiente de pesquisa neuromorfa.

Rode com:
    python validate_environment.py

Saída esperada: tudo verde. Qualquer ❌ é um problema a investigar
antes de mergulhar em código de pesquisa.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
import time

OK = "\033[92m✔\033[0m"
FAIL = "\033[91m✘\033[0m"
WARN = "\033[93m!\033[0m"


def header(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def check(label: str, fn) -> bool:
    try:
        result = fn()
    except Exception as e:
        print(f"  {FAIL} {label}: {type(e).__name__}: {e}")
        return False
    if result is None or result is True:
        print(f"  {OK} {label}")
        return True
    print(f"  {OK} {label} — {result}")
    return True


# 1. Sistema
header("1. Sistema operacional e Python")
print(f"  Python:   {sys.version.split()[0]}")
print(f"  Platform: {platform.platform()}")
print(f"  Conda:    {shutil.which('conda') or shutil.which('mamba') or '(não detectado)'}")

if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    print(f"  {WARN} Python {sys.version_info.major}.{sys.version_info.minor} fora da faixa testada (3.10–3.12). Pode dar fricção.")
else:
    print(f"  {OK} Versão de Python suportada.")


# 2. PyTorch + CUDA
header("2. PyTorch e CUDA")
try:
    import torch
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  CUDA build: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"  {OK} CUDA disponível")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"  Compute capability: {cap[0]}.{cap[1]}")
        if cap == (8, 9):
            print(f"  {OK} Ada Lovelace detectada (RTX 4070 / 4080 / 4090).")
        x = torch.randn(2048, 2048, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            y = x @ x
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 10
        gflops = (2 * 2048**3) / dt / 1e9
        print(f"  Benchmark matmul 2048² fp32: {dt*1000:.1f} ms/iter ≈ {gflops:.0f} GFLOPS")
        if gflops < 1000:
            print(f"  {WARN} GFLOPS abaixo do esperado pra 4070 (>5000 fp32). Verifique se a GPU é mesmo a 4070 e não uma integrada.")
    else:
        print(f"  {FAIL} CUDA NÃO disponível. PyTorch vai rodar só em CPU.")
        print(f"     Causa comum: driver NVIDIA desatualizado ou rodando em WSL1.")
except ImportError:
    print(f"  {FAIL} PyTorch não instalado. Rode: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")


# 3. Frameworks de Spiking Neural Networks
header("3. Frameworks neuromorfos")

frameworks = {
    "snntorch": "snnTorch (PyTorch-based SNN)",
    "norse": "Norse (PyTorch SNN, biologically faithful)",
    "brian2": "Brian2 (CPU SNN simulator, gera C++)",
    "nest": "NEST (large-scale neural simulator) — opcional",
}

for module, label in frameworks.items():
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", "?")
        print(f"  {OK} {label}: {version}")
    except ImportError:
        if module == "nest":
            print(f"  {WARN} {label}: não instalado (opcional pra começar)")
        else:
            print(f"  {FAIL} {label}: não instalado")


# 4. Bibliotecas auxiliares
header("4. Bibliotecas auxiliares")

aux = [
    "numpy", "scipy", "matplotlib", "pandas",
    "sklearn", "tqdm", "einops",
    "jupyterlab", "tensorboard",
]

for module in aux:
    try:
        m = importlib.import_module(module)
        print(f"  {OK} {module}: {getattr(m, '__version__', '?')}")
    except ImportError:
        print(f"  {FAIL} {module}: não instalado")


# 5. Bibliotecas pra raciocínio causal e meta-learning
header("5. Causal / meta-learning (pilares 2 e 3)")

advanced = {
    "pyro": "Pyro (probabilistic programming)",
    "dowhy": "DoWhy (Microsoft)",
    "causalnex": "CausalNex (QuantumBlack)",
    "learn2learn": "learn2learn (meta-learning)",
}

for module, label in advanced.items():
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", "?")
        print(f"  {OK} {label}: {version}")
    except ImportError:
        print(f"  {WARN} {label}: não instalado (opcional na fase atual)")


header("Resumo")
print("""
Próximo passo:
  python validate_snn_minimal.py     # roda uma SNN mínima end-to-end
  python validate_brian2_stdp.py     # confirma Brian2 + STDP
""".rstrip())
