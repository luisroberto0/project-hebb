"""
Marco 2-C (#72) — dataloader do SHD (Spiking Heidelberg Digits) via h5py.

SHD HDF5 (Zenke lab): grupo 'spikes' com vlen 'times' (s) e 'units' (canal 0-699),
e 'labels' (0-19). Convertemos cada amostra (lista esparsa de spikes) em um tensor
denso (n_bins, n_units) por binning temporal — a entrada das SNNs.

Baseline cego ao timing: bag_of_spikes(x) = soma sobre o tempo -> (n_units,), destrói
a ordem temporal preservando a contagem por canal (espectro médio).

Uso:
    python shd_data.py   # smoke: shapes, densidade, sanity
"""
from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
N_UNITS = 700
N_CLASSES = 20
MAX_T = 1.4   # s — SHD spikes vão até ~1.4 s


class SHDDataset(Dataset):
    """Carrega spikes esparsos na memória (SHD é pequeno); binning denso lazy no __getitem__."""

    def __init__(self, split: str, n_bins: int = 100, coding: str = "rate",
                 dataset: str = "shd", max_per_class: int | None = None, lazy: bool = False):
        path = os.path.join(DATA_DIR, f"{dataset}_{split}.h5")
        self.lazy = lazy  # lazy: lê do HDF5 por índice (SSC completo ~75k não cabe na RAM)
        if lazy:
            self._f = h5py.File(path, "r")
            self._td = self._f["spikes"]["times"]
            self._ud = self._f["spikes"]["units"]
            self.labels = np.asarray(self._f["labels"][:], dtype=np.int64)
        else:
            with h5py.File(path, "r") as f:
                labels_all = np.asarray(f["labels"][:], dtype=np.int64)
                if max_per_class is not None:
                    # subset balanceado (SSC é grande: ~75k -> usa N por classe)
                    sel = []
                    for c in np.unique(labels_all):
                        sel.extend(np.where(labels_all == c)[0][:max_per_class].tolist())
                    sel = sorted(sel)
                    td, ud = f["spikes"]["times"], f["spikes"]["units"]
                    self.times = [np.asarray(td[i], dtype=np.float32) for i in sel]
                    self.units = [np.asarray(ud[i], dtype=np.int64) for i in sel]
                    self.labels = labels_all[sel]
                else:
                    self.times = [np.asarray(t, dtype=np.float32) for t in f["spikes"]["times"]]
                    self.units = [np.asarray(u, dtype=np.int64) for u in f["spikes"]["units"]]
                    self.labels = labels_all
        self.n_bins = n_bins
        self.coding = coding  # "rate" (contagem por bin) | "latency" (time-to-first-spike)
        self.n_classes = int(self.labels.max()) + 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.lazy:
            t = np.asarray(self._td[i], dtype=np.float32)
            u = np.asarray(self._ud[i], dtype=np.int64)
        else:
            t, u = self.times[i], self.units[i]
        x = torch.zeros(self.n_bins, N_UNITS)
        if t.size:
            b = np.clip((t / MAX_T * self.n_bins).astype(np.int64), 0, self.n_bins - 1)
            uu = np.clip(u, 0, N_UNITS - 1)
            if self.coding == "rate":
                np.add.at(x.numpy(), (b, uu), 1.0)  # contagem de spikes por (bin, unit)
            else:  # latency: 1 spike por canal, no bin do PRIMEIRO spike (time-to-first-spike)
                order = np.argsort(t, kind="stable")
                uo, first = np.unique(uu[order], return_index=True)
                x.numpy()[b[order][first], uo] = 1.0
        return x, int(self.labels[i])


def get_shd_loaders(batch=128, n_bins=100):
    tr = SHDDataset("train", n_bins)
    te = SHDDataset("test", n_bins)
    return (DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0),
            DataLoader(te, batch_size=batch, shuffle=False, num_workers=0))


def bag_of_spikes(x: torch.Tensor) -> torch.Tensor:
    """Baseline cego ao timing: soma sobre o eixo temporal. (B, T, U) -> (B, U)."""
    return x.sum(dim=1)


if __name__ == "__main__":
    print("Carregando SHD test...")
    ds = SHDDataset("test", n_bins=100)
    print(f"  n_amostras={len(ds)}  n_classes={len(np.unique(ds.labels))}")
    x, y = ds[0]
    print(f"  amostra 0: x.shape={tuple(x.shape)} (bins, units)  label={y}  spikes_totais={x.sum().item():.0f}")
    dens = (x > 0).float().mean().item()
    print(f"  densidade (frac bins-unit ativos): {dens:.4f}")
    bag = bag_of_spikes(x.unsqueeze(0))
    print(f"  bag_of_spikes (baseline cego): shape={tuple(bag.shape)} canais_ativos={(bag[0]>0).sum().item()}/{N_UNITS}")
    # distribuição de labels
    u, c = np.unique(ds.labels, return_counts=True)
    print(f"  labels: {u.tolist()}  counts min/max={c.min()}/{c.max()}")
