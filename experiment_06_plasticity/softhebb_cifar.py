"""
Marco 3 — Plasticidade local sem backprop (eixo SoftHebb). Harness com CONTROLES.

Baseado na implementação oficial single-file dos autores (NeuromorphicComputing/SoftHebb, demo.py),
a regra SoftHebb e a arquitetura de 3 camadas (96/384/1536) são fiéis ao paper (ICLR 2023, ~80% CIFAR-10).
Aqui estruturado em MODOS para o critério literal do Marco 3:

  --mode softhebb : pilha treinada SO por Hebbian competitivo local (sem backprop) + linear-probe
  --mode random   : MESMA arquitetura, pesos RANDOM congelados (sem treino unsup) + linear-probe   [CONTROLE-CHAVE]
  --mode wta_off  : Hebbian SEM competição (sem o anti-Hebbian dos perdedores) + linear-probe        [CONTROLE b]
  --mode backprop : MESMA macro-arquitetura treinada end-to-end por backprop (TETO)

A pilha Hebbian/random é congelada; só o classificador linear é treinado (backprop) — é o linear-probe.
Mede se a regra Hebbian competitiva carrega sinal REAL vs a arquitetura sozinha (a lição do STDP, #10).

Uso:
    python softhebb_cifar.py --mode softhebb --seed 0 --probe-epochs 50
    python softhebb_cifar.py --mode random   --seed 0 --probe-epochs 50
"""
from __future__ import annotations
import argparse, math, os, warnings, json
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.optim.lr_scheduler import StepLR

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ============================ SoftHebb layer (fiel ao demo oficial) ============================
class SoftHebbConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, t_invert=12., competitive=True):
        super().__init__()
        assert groups == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = 'reflect'
        self.F_padding = (padding, padding, padding, padding)
        weight_range = 25 / math.sqrt((in_channels / groups) * kernel_size * kernel_size)
        self.weight = nn.Parameter(weight_range * torch.randn((out_channels, in_channels // groups, *self.kernel_size)))
        self.t_invert = torch.tensor(t_invert)
        self.competitive = competitive  # False = WTA off (controle b): Hebbian sem anti-Hebbian dos perdedores

    def forward(self, x):
        x = F.pad(x, self.F_padding, self.padding_mode)
        weighted_input = F.conv2d(x, self.weight, None, self.stride, 0, self.dilation, self.groups)
        if self.training:
            batch_size, out_channels, height_out, width_out = weighted_input.shape
            flat_weighted_inputs = weighted_input.transpose(0, 1).reshape(out_channels, -1)
            flat_softwta_activs = torch.softmax(self.t_invert * flat_weighted_inputs, dim=0)
            if self.competitive:
                flat_softwta_activs = - flat_softwta_activs  # anti-Hebbian para todos
                win_neurons = torch.argmax(flat_weighted_inputs, dim=0)
                competing_idx = torch.arange(flat_weighted_inputs.size(1))
                flat_softwta_activs[win_neurons, competing_idx] = - flat_softwta_activs[win_neurons, competing_idx]  # winner -> Hebbian
            # se competitive=False: todos os neuronios recebem update Hebbian (softmax positivo), sem competicao
            softwta_activs = flat_softwta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)
            yx = F.conv2d(x.transpose(0, 1), softwta_activs.transpose(0, 1), padding=0,
                          stride=self.dilation, dilation=self.stride, groups=1).transpose(0, 1)
            yu = torch.sum(torch.mul(softwta_activs, weighted_input), dim=(0, 2, 3))
            delta_weight = yx - yu.view(-1, 1, 1, 1) * self.weight
            delta_weight.div_(torch.abs(delta_weight).amax() + 1e-30)
            self.weight.grad = delta_weight  # usado pelo TensorLRSGD (lr negativo = ascent Hebbian)
        return weighted_input


class Triangle(nn.Module):
    def __init__(self, power: float = 1, inplace: bool = True):
        super().__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, x):
        x = x - torch.mean(x.data, axis=1, keepdims=True)
        return F.relu(x, inplace=self.inplace) ** self.power


class DeepSoftHebb(nn.Module):
    """Pilha Hebbian de 3 camadas (fiel ao demo) + classificador linear (linear-probe)."""
    def __init__(self, competitive=True):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = SoftHebbConv2d(3, 96, 5, padding=2, t_invert=1., competitive=competitive)
        self.activ1 = Triangle(power=0.7)
        self.pool1 = nn.MaxPool2d(4, 2, 1)
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.conv2 = SoftHebbConv2d(96, 384, 3, padding=1, t_invert=0.65, competitive=competitive)
        self.activ2 = Triangle(power=1.4)
        self.pool2 = nn.MaxPool2d(4, 2, 1)
        self.bn3 = nn.BatchNorm2d(384, affine=False)
        self.conv3 = SoftHebbConv2d(384, 1536, 3, padding=1, t_invert=0.25, competitive=competitive)
        self.activ3 = Triangle(power=1.)
        self.pool3 = nn.AvgPool2d(2, 2, 0)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(24576, 10)
        self.classifier.weight.data = 0.11048543456039805 * torch.rand(10, 24576)
        self.dropout = nn.Dropout(0.5)

    def features(self, x):
        out = self.pool1(self.activ1(self.conv1(self.bn1(x))))
        out = self.pool2(self.activ2(self.conv2(self.bn2(out))))
        out = self.pool3(self.activ3(self.conv3(self.bn3(out))))
        return self.flatten(out)

    def forward(self, x):
        return self.classifier(self.dropout(self.features(x)))

    def hebbian_convs(self):
        return [self.conv1, self.conv2, self.conv3]

    def freeze_features(self):
        for m in [self.conv1, self.conv2, self.conv3, self.bn1, self.bn2, self.bn3]:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


class DeepBackpropCNN(nn.Module):
    """MESMA macro-arquitetura (96/384/1536, mesmos kernels/pools) treinada e2e por backprop = TETO."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, 5, padding=2), nn.BatchNorm2d(96), nn.ReLU(), nn.MaxPool2d(4, 2, 1),
            nn.Conv2d(96, 384, 3, padding=1), nn.BatchNorm2d(384), nn.ReLU(), nn.MaxPool2d(4, 2, 1),
            nn.Conv2d(384, 1536, 3, padding=1), nn.BatchNorm2d(1536), nn.ReLU(), nn.AvgPool2d(2, 2, 0),
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(24576, 10),
        )

    def forward(self, x):
        return self.net(x)


# ============================ optimizers/schedulers (fiéis ao demo) ============================
class TensorLRSGD(optim.SGD):
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.add_(-group['lr'] * p.grad)
        return None


class WeightNormDependentLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, power_lr, last_epoch=-1):
        self.optimizer = optimizer
        self.initial_lr_groups = [g['lr'] for g in optimizer.param_groups]
        self.power_lr = power_lr
        super().__init__(optimizer, last_epoch, False)

    def get_lr(self):
        new_lr = []
        for i, group in enumerate(self.optimizer.param_groups):
            for param in group['params']:
                norm_diff = torch.abs(torch.linalg.norm(param.view(param.shape[0], -1), dim=1, ord=2) - 1) + 1e-10
                new_lr.append(self.initial_lr_groups[i] * (norm_diff ** self.power_lr)[:, None, None, None])
        return new_lr

    def step(self, epoch=None):
        # torch 2.6: base step faz lr.fill_(escalar); aqui o lr e tensor 4D por-filtro -> atribuicao direta
        self.last_epoch += 1
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group['lr'] = lr


class CustomStepLR(StepLR):
    def __init__(self, optimizer, nb_epochs):
        ratios = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.step_thresold = [int(nb_epochs * r) for r in ratios]
        super().__init__(optimizer, -1, False)

    def get_lr(self):
        if self.last_epoch in self.step_thresold:
            return [g['lr'] * 0.5 for g in self.optimizer.param_groups]
        return [g['lr'] for g in self.optimizer.param_groups]


class CIFAR10NPZ(torch.utils.data.Dataset):
    """Carrega CIFAR-10 do cifar10.npz (decodificado dos parquets HF). Pre-carrega no device (GPU)."""
    def __init__(self, train=True, device="cpu"):
        d = np.load(os.path.join(DATA_DIR, "cifar10.npz"))
        x = d["train_x"] if train else d["test_x"]
        y = d["train_y"] if train else d["test_y"]
        self.data = torch.tensor(x, dtype=torch.float, device=device).div_(255).movedim(-1, 1).contiguous()
        self.targets = torch.tensor(y, device=device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


# ============================ treino/eval ============================
def train_unsup(model, loader, device):
    """1 passada Hebbian local (sem backprop). lr negativo = ascent na regra."""
    opt = TensorLRSGD([
        {"params": model.conv1.parameters(), "lr": -0.08},
        {"params": model.conv2.parameters(), "lr": -0.005},
        {"params": model.conv3.parameters(), "lr": -0.01},
    ], lr=0)
    sched = WeightNormDependentLR(opt, power_lr=0.5)
    model.train()
    for inputs, _ in loader:
        inputs = inputs.to(device)
        opt.zero_grad()
        with torch.no_grad():
            model(inputs)
        opt.step()
        sched.step()


def train_probe(model, loader, device, epochs):
    """Linear-probe: congela features, treina SO o classificador (backprop)."""
    model.freeze_features()
    opt = optim.Adam(model.classifier.parameters(), lr=0.001)
    sched = CustomStepLR(opt, nb_epochs=epochs)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.classifier.train(); model.dropout.train()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            loss = crit(model(inputs), labels)
            loss.backward()
            opt.step()
        sched.step()


def train_backprop(model, loader, device, epochs):
    """Teto: treina a macro-arquitetura inteira e2e por backprop."""
    opt = optim.Adam(model.parameters(), lr=0.001)
    sched = CustomStepLR(opt, nb_epochs=epochs)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            loss = crit(model(inputs), labels)
            loss.backward()
            opt.step()
        sched.step()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        _, pred = torch.max(model(inputs), 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100.0 * correct / total


def run(mode, seed, probe_epochs, device):
    torch.manual_seed(seed)
    trainset = CIFAR10NPZ(train=True, device=device)
    testset = CIFAR10NPZ(train=False, device=device)
    unsup_loader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
    sup_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

    if mode == "backprop":
        model = DeepBackpropCNN().to(device)
        train_backprop(model, sup_loader, device, probe_epochs)
    else:
        competitive = (mode != "wta_off")
        model = DeepSoftHebb(competitive=competitive).to(device)
        if mode in ("softhebb", "wta_off"):
            train_unsup(model, unsup_loader, device)
        # mode == "random": pula o treino unsup (pesos random congelados)
        train_probe(model, sup_loader, device, probe_epochs)

    acc = evaluate(model, test_loader, device)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["softhebb", "random", "wta_off", "backprop"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-epochs", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    acc = run(args.mode, args.seed, args.probe_epochs, dev)
    line = f"mode={args.mode} seed={args.seed} probe_epochs={args.probe_epochs} acc={acc:.2f}"
    print(line)
    with open(os.path.join(os.path.dirname(__file__), "results_softhebb.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
