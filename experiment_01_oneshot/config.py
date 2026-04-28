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
    """Hiperparâmetros da regra STDP (Song/Miller/Abbott 2000, Diehl/Cook 2015).

    NOTA 2026-04-27 (Etapa 2 sessão): tentativas de rebalancear A_pre/A_post
    pra compensar regime esparso de k=1 WTA falharam. Razão pré:pós empírica
    = 10.1 (tests/test_spike_balance.py). Trade-off observado:
      R≈1 (paper original): LTD domina → pesos morrem → 17.76% acurácia
      R≈10 (balance ideal): LTP domina → "rich-get-richer", 1 filtro vence
                            sempre → 11.51% acurácia
      R=3  (meio termo):    rich-get-richer ainda persiste → 11.36% acurácia
    Conclusão: o gargalo NÃO é só ratio LTP/LTD. Diehl & Cook usam adaptive
    threshold homeostático (não implementado aqui ainda) que força cada
    filtro a disparar aproximadamente igualmente. Sem isso, qualquer regime
    LTP>LTD colapsa filtros pra 1; qualquer LTP<LTD mata pesos.
    Mantido valores originais do paper (= melhor estado conhecido até aqui).
    """
    tau_pre_ms: float = 20.0
    tau_post_ms: float = 20.0
    A_pre: float = 0.0001         # Sessão #7 (Etapa 1 Semana 2): A_pre=0.01 do paper saturou
                                  # layer 1 em 10 timesteps com conv real (output denso de k=1
                                  # WTA × 28×28 posições). Pesos atingiam max=1 em t=10, oscilavam,
                                  # zeravam em t=70 (theta locked-out). Reduzido 100× pra dar STDP
                                  # gentle compatível com regime denso de spikes.
    A_post: float = -0.000105     # Mantido razão R≈1.05 do paper (calibração de balance).
    w_min: float = 0.0
    w_max: float = 1.0
    w_init_low: float = 0.0
    w_init_high: float = 0.3
    lateral_inhibition: float = 0.5  # força da inibição entre filtros (winner-take-all soft)
    # --- Homeostasis (adaptive threshold) — Diehl & Cook 2015 §2.3 ---
    # Cada filtro tem theta_i adicional ao threshold base. Theta cresce com
    # cada spike próprio (theta_plus) e decai lentamente com tempo (tau_theta).
    # Função: forçar todos os filtros a disparar aproximadamente igualmente,
    # quebrando rich-get-richer do k-WTA + STDP. Sessão #3 isolou este como
    # causa raiz do colapso de filtros.
    theta_plus: float = 0.0005     # Restaurado pós-H_theta_omn (sessão #8): tuning não
                                   # destrava acurácia. Ver WEEKLY-2.md tabela H_theta_omn.
                                   # Causa raiz isolada: tau_theta=1e7 não permite decay
                                   # efetivo no nosso regime → theta cresce monotonicamente,
                                   # qualquer theta_plus gera trade-off (silenciar OU saturar).
    tau_theta_ms: float = 1e7      # Revertido sessão #10 — A3 mostrou que sinal de 35.98%
                                   # era ~80% arquitetural (magnitude alta + Poisson + Hopfield),
                                   # não STDP. Decisão arquitetural sessão #9 foi prematura: o
                                   # que destravou sinal foi a saturação dos pesos (qualquer
                                   # config que satura produz isso), não a calibração de
                                   # tau_theta. Theta treinada é inerte (A3b). Estado conhecido
                                   # do paper restaurado.
    # H_norm (sessão #12): normalização de Σ|w| por (out, in) sub-kernel pós-STDP-update.
    # None = desativado (default; estado pré-#12 e pós-#12).
    # Float = alvo de mean(|w|) por sub-kernel.
    # Sessão #12: testado com 0.3 → destravou diversidade (centered cosine L2: 0.55 → 0.04)
    # MAS magnitude 0.3 é insuficiente pra gerar spikes pós-LIF → acc 20.04% (chance).
    # Sweet spot a investigar em sessão futura (target maior, ex 0.6, 0.8).
    # Código de normalização permanece em model.py:ConvSTDPLayer.stdp_update.
    norm_target_mean: float | None = None


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
