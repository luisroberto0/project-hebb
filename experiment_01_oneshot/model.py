"""
Arquitetura: Conv-STDP + Memória Episódica Hopfield Moderna.

STDP convolucional vetorizado em PyTorch puro, baseado em
Kheradpisheh et al. (2018) e Diehl & Cook (2015). A vetorização usa
F.unfold pra extrair patches da entrada na resolução receptiva de cada
neurônio de saída — o que permite calcular Δw[c_out, c_in, kh, kw] em
uma única chamada einsum, sem loops Python.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# ---------------------------------------------------------------------------
# Camada LIF + STDP
# ---------------------------------------------------------------------------
class ConvSTDPLayer(nn.Module):
    """
    Camada convolucional com neurônios LIF e atualização STDP local.

    `forward` processa um único timestep. `stdp_update` aplica a regra
    de plasticidade local olhando spikes pré/pós e seus traços
    exponenciais. Pesos não passam por backprop em momento algum
    (`requires_grad=False`).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, bias=False,
        )
        self.conv.weight.requires_grad_(False)
        with torch.no_grad():
            self.conv.weight.uniform_(cfg.stdp.w_init_low, cfg.stdp.w_init_high)
        # Traços alocados sob demanda em reset_traces.
        self.apre: torch.Tensor | None = None
        self.apost: torch.Tensor | None = None

    def reset_traces(self, batch_size: int, in_shape: tuple, out_shape: tuple, device: torch.device) -> None:
        self.apre = torch.zeros(batch_size, *in_shape, device=device)
        self.apost = torch.zeros(batch_size, *out_shape, device=device)

    def forward(self, spikes_t: torch.Tensor, mem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Um único timestep com k-WTA (k=1) lateral inhibition.
        spikes_t: (B, C_in, H, W)
        mem:      (B, C_out, H', W')

        Inibição lateral implementada como winner-take-all por posição espacial:
        apenas o filtro com maior membrana em cada (B, H, W) pode disparar.

        MELHOR RESULTADO EM TESTES: distribui filtros entre todas as classes,
        acurácia 17.76% (vs 9.80% sem WTA, 10.89% com k=5).
        """
        I = self.conv(spikes_t)
        decay = torch.exp(torch.tensor(-self.cfg.spike.dt_ms / self.cfg.lif.tau_mem_ms, device=spikes_t.device))
        mem = decay * mem + I

        # Spikes brutos (sem inibição lateral)
        spikes_raw = (mem >= self.cfg.lif.v_thresh).float()

        # k-WTA (k=1): só o filtro com maior membrana por posição espacial dispara
        max_filter_idx = mem.argmax(dim=1, keepdim=True)  # (B, 1, H', W')
        wta_mask = torch.zeros_like(mem).scatter_(1, max_filter_idx, 1.0)
        spikes_out = spikes_raw * wta_mask

        # Reset de membrana apenas nos neurônios que dispararam
        mem = mem * (1 - spikes_out) + self.cfg.lif.v_reset * spikes_out
        return spikes_out, mem

    @torch.no_grad()
    def stdp_update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> None:
        """
        STDP convolucional vetorizado.

        Regra (Diehl & Cook 2015 / Kheradpisheh 2018, normalizada):
          LTP (causal — pré antes de pós):
            Δw[c_out, c_in, kh, kw] += A_pre * Σ_{b,i,j} post[b,c_out,i,j] · apre[b,c_in,i+kh-p,j+kw-p]
          LTD (anti-causal — pós antes de pré):
            Δw[c_out, c_in, kh, kw] += A_post * Σ_{b,i,j} pre[b,c_in,i+kh-p,j+kw-p] · apost[b,c_out,i,j]

        Convenção de sinal: A_pre > 0 (potenciação), A_post < 0 (depressão).

        Vetorização: F.unfold extrai patches (kH×kW) de apre/pre na grade
        de saída (com padding p), produzindo tensor (B, C_in*kH*kW, Hp*Wp)
        que se contrai com post/apost flat (B, C_out, Hp*Wp) via einsum.

        Ordem importante: traços atualizam ANTES das contribuições serem
        lidas, seguindo a convenção de Diehl & Cook.
        """
        if self.apre is None or self.apre.shape != pre_spikes.shape:
            self.apre = torch.zeros_like(pre_spikes)
        if self.apost is None or self.apost.shape != post_spikes.shape:
            self.apost = torch.zeros_like(post_spikes)

        cfg = self.cfg.stdp
        dt = self.cfg.spike.dt_ms
        device = pre_spikes.device
        decay_pre = torch.exp(torch.tensor(-dt / cfg.tau_pre_ms, device=device))
        decay_post = torch.exp(torch.tensor(-dt / cfg.tau_post_ms, device=device))
        self.apre = decay_pre * self.apre + pre_spikes
        self.apost = decay_post * self.apost + post_spikes

        kH = kW = self.kernel_size
        pad = self.padding
        C_out = self.out_channels

        apre_patches = F.unfold(self.apre, kernel_size=(kH, kW), padding=pad)   # (B, C_in*kH*kW, Hp*Wp)
        pre_patches = F.unfold(pre_spikes, kernel_size=(kH, kW), padding=pad)   # (B, C_in*kH*kW, Hp*Wp)
        post_flat = post_spikes.flatten(start_dim=2)                            # (B, C_out, Hp*Wp)
        apost_flat = self.apost.flatten(start_dim=2)                            # (B, C_out, Hp*Wp)

        delta_LTP = torch.einsum("bop,bip->oi", post_flat, apre_patches)        # (C_out, C_in*kH*kW)
        delta_LTD = torch.einsum("bop,bip->oi", apost_flat, pre_patches)
        delta_LTP = delta_LTP.view(C_out, self.in_channels, kH, kW)
        delta_LTD = delta_LTD.view(C_out, self.in_channels, kH, kW)

        self.conv.weight.data += cfg.A_pre * delta_LTP + cfg.A_post * delta_LTD
        # Clip pra manter pesos no intervalo definido (inibição lateral agora é via k-WTA no forward)
        self.conv.weight.data.clamp_(cfg.w_min, cfg.w_max)

    def clip_weights(self) -> None:
        with torch.no_grad():
            self.conv.weight.clamp_(self.cfg.stdp.w_min, self.cfg.stdp.w_max)


# ---------------------------------------------------------------------------
# Pool espacial
# ---------------------------------------------------------------------------
class SpatialMaxPool(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.pool = nn.MaxPool2d(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


# ---------------------------------------------------------------------------
# Modern Hopfield Network como memória episódica
# ---------------------------------------------------------------------------
class HopfieldMemory(nn.Module):
    """
    Memória associativa Hopfield Moderna (Ramsauer et al. 2020).
    K e V são as memórias armazenadas; predição = softmax(β · sim) · V.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    def store(self, embeddings: torch.Tensor, labels: torch.Tensor, n_classes: int) -> None:
        if self.cfg.memory.normalize_keys:
            embeddings = F.normalize(embeddings, dim=-1)
        self.keys = embeddings
        self.values = F.one_hot(labels, num_classes=n_classes).float()

    def query(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.keys is None or self.values is None:
            raise RuntimeError("Memória vazia. Chame store() antes de query().")
        if self.cfg.memory.normalize_keys:
            embeddings = F.normalize(embeddings, dim=-1)
        if self.cfg.memory.distance == "cosine":
            sims = embeddings @ self.keys.T
        elif self.cfg.memory.distance == "euclidean":
            sims = -torch.cdist(embeddings, self.keys)
        else:
            raise ValueError(self.cfg.memory.distance)
        weights = F.softmax(self.cfg.memory.beta * sims, dim=-1)
        return weights @ self.values


# ---------------------------------------------------------------------------
# Modelo completo
# ---------------------------------------------------------------------------
class STDPHopfieldModel(nn.Module):
    """
    Pipeline:
      spike encoding → conv-STDP layer 1 → pool → conv-STDP layer 2 → pool
        → flatten → projeção ortogonal → memória Hopfield → predição
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        a = cfg.arch
        self.layer1 = ConvSTDPLayer(1, a.conv1_filters, a.conv1_kernel, cfg)
        self.pool1 = SpatialMaxPool(a.conv1_pool)
        self.layer2 = ConvSTDPLayer(a.conv1_filters, a.conv2_filters, a.conv2_kernel, cfg)
        self.pool2 = SpatialMaxPool(a.conv2_pool)
        self.memory = HopfieldMemory(cfg)
        self._proj: nn.Linear | None = None
        self._shape_cache: dict[tuple, tuple] = {}

    def _ensure_projection(self, flat_dim: int, device: torch.device) -> None:
        if self._proj is None or self._proj.in_features != flat_dim:
            proj = nn.Linear(flat_dim, self.cfg.arch.embedding_dim, bias=False)
            with torch.no_grad():
                nn.init.orthogonal_(proj.weight)
            proj.weight.requires_grad_(False)
            self._proj = proj.to(device)

    def _compute_shapes(self, image_size: tuple) -> tuple:
        if image_size in self._shape_cache:
            return self._shape_cache[image_size]
        H, W = image_size
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.zeros(1, 1, H, W, device=device)
            o1 = self.layer1.conv(x); p1 = self.pool1(o1)
            o2 = self.layer2.conv(p1); p2 = self.pool2(o2)
        shapes = (o1.shape[1:], o2.shape[1:], p1.shape[1:], p2.shape[1:])
        self._shape_cache[image_size] = shapes
        return shapes

    def extract_features(self, images: torch.Tensor, train_stdp: bool = False) -> torch.Tensor:
        """
        images: (B, H, W) em [0,1]. Retorna embeddings (B, embedding_dim).
        Se train_stdp=True, aplica stdp_update em cada timestep.
        """
        from data import encode

        B, H, W = images.shape
        device = images.device
        x = images.unsqueeze(1)
        spikes_in = encode(x, self.cfg)
        T = spikes_in.shape[0]

        out1_shape, out2_shape, pool1_shape, pool2_shape = self._compute_shapes((H, W))
        mem1 = torch.zeros((B,) + out1_shape, device=device)
        mem2 = torch.zeros((B,) + out2_shape, device=device)
        spike_count_final = torch.zeros((B,) + pool2_shape, device=device)

        if train_stdp:
            # Reset de traços por imagem (regra Diehl & Cook: cada amostra é episódio independente)
            self.layer1.apre = None; self.layer1.apost = None
            self.layer2.apre = None; self.layer2.apost = None

        for t in range(T):
            pre1 = spikes_in[t]
            spk1, mem1 = self.layer1(pre1, mem1)
            if train_stdp:
                self.layer1.stdp_update(pre1, spk1)
            spk1_pool = self.pool1(spk1)
            spk2, mem2 = self.layer2(spk1_pool, mem2)
            if train_stdp:
                self.layer2.stdp_update(spk1_pool, spk2)
            spk2_pool = self.pool2(spk2)
            spike_count_final = spike_count_final + spk2_pool

        rate = spike_count_final / T
        flat = rate.flatten(start_dim=1)
        self._ensure_projection(flat.shape[-1], device)
        return self._proj(flat)

    def forward(self, support: torch.Tensor, support_labels: torch.Tensor,
                query: torch.Tensor, n_classes: int) -> torch.Tensor:
        sup_emb = self.extract_features(support, train_stdp=False)
        self.memory.store(sup_emb, support_labels, n_classes)
        qry_emb = self.extract_features(query, train_stdp=False)
        return self.memory.query(qry_emb)
