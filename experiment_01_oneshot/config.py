"""
Configuração central do experimento.

Centraliza hiperparâmetros pra que todos os scripts (data, model, train,
evaluate) puxem do mesmo lugar. Ao mudar uma escolha de design, edite
aqui em vez de caçar em vários arquivos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Configuração de dataset Omniglot."""
    root: Path = Path("./data/omniglot")
    image_size: int = 28          # downsample de 105 → 28 pra acelerar
    augment_rotations: tuple = (0, 90, 180, 270)  # data augmentation clássica do Omniglot


@dataclass
class SpikeConfig:
    """Codificação imagem → trem de spikes."""
    timesteps: int = 100          # T: comprimento da janela de simulação
    encoding: str = "poisson"     # 'poisson' (rate) ou 'temporal' (TTFS)
    max_rate_hz: float = 100.0    # taxa máxima pra pixel branco
    dt_ms: float = 1.0


@dataclass
class STDPConfig:
    """Hiperparâmetros da regra STDP (Song/Miller/Abbott 2000, Diehl/Cook 2015)."""
    tau_pre_ms: float = 20.0
    tau_post_ms: float = 20.0
    A_pre: float = 0.01
    A_post: float = -0.0105       # leve assimetria pra estabilidade
    w_min: float = 0.0
    w_max: float = 1.0
    w_init_low: float = 0.0
    w_init_high: float = 0.3
    lateral_inhibition: float = 0.5  # força da inibição entre filtros (winner-take-all soft)


@dataclass
class LIFConfig:
    """Neurônio Leaky Integrate-and-Fire."""
    tau_mem_ms: float = 20.0
    v_thresh: float = 1.0
    v_reset: float = 0.0
    refractory_ms: float = 5.0


@dataclass
class ArchitectureConfig:
    """Topologia da rede convolucional STDP."""
    conv1_filters: int = 8
    conv1_kernel: int = 5
    conv1_pool: int = 2
    conv2_filters: int = 16
    conv2_kernel: int = 5
    conv2_pool: int = 2
    embedding_dim: int = 64       # dimensão final pré-memória; flatten + projeção opcional


@dataclass
class MemoryConfig:
    """Memória episódica Hopfield Moderna."""
    beta: float = 8.0              # temperatura inversa do softmax (Ramsauer 2020)
    distance: str = "cosine"       # 'cosine' ou 'euclidean'
    normalize_keys: bool = True


@dataclass
class TrainConfig:
    """Pretreino STDP não-supervisionado."""
    epochs: int = 1
    batch_size: int = 1            # STDP é online, batch=1 é o natural
    n_pretrain_images: int = 24000  # imagens do split background
    log_every: int = 100
    save_every: int = 1000


@dataclass
class EvalConfig:
    """Avaliação few-shot."""
    n_episodes: int = 1000
    ways: tuple = (5, 20)          # N-way: classes por episódio
    shots: tuple = (1, 5)          # K-shot: exemplos de support por classe
    queries_per_class: int = 5
    seed: int = 42


@dataclass
class Config:
    """Configuração raiz."""
    data: DataConfig = field(default_factory=DataConfig)
    spike: SpikeConfig = field(default_factory=SpikeConfig)
    stdp: STDPConfig = field(default_factory=STDPConfig)
    lif: LIFConfig = field(default_factory=LIFConfig)
    arch: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    device: str = "cuda"           # 'cuda' ou 'cpu'
    seed: int = 42
    log_dir: Path = Path("./logs")
    checkpoint_dir: Path = Path("./checkpoints")


def default_config() -> Config:
    """Configuração de referência. Use como ponto de partida e edite o que quiser."""
    return Config()
