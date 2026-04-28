"""
Auditoria das funções `assign_labels` e `evaluate` em sanity_mnist.py.

Pergunta sob teste: dada distribuição de filtros [36,40,6,11,18,19,17,43,2,8]
e acurácia observada de 9.94% (chance), o problema está em (A) features STDP
sem sinal discriminativo OU (B) bug em assignment/evaluate?

Estratégia: mockar `forward_image` retornando spike_counts sintéticos com
estrutura conhecida e medir acurácia esperada. Se o caso "features perfeitas"
não der 100%, achamos bug no pipeline de classificação.

Uso:
    cd experiment_01_oneshot && python tests/test_assignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite importar sanity_mnist quando rodado de tests/
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset

from sanity_mnist import assign_labels, evaluate


# ---------------------------------------------------------------------------
# Mock de SanityNet: expõe n_filters e forward_image que retorna features fixas
# ---------------------------------------------------------------------------
class FakeNet:
    """Retorna spike_count pré-definido por classe — não roda STDP/LIF."""

    def __init__(self, n_filters: int, response_per_class: torch.Tensor):
        """
        response_per_class: (n_classes, n_filters) — spike_count médio que cada
        filtro deve produzir pra cada classe. Imagens dentro da mesma classe
        retornam o mesmo vetor (sem ruído).
        """
        self.n_filters = n_filters
        self.response_per_class = response_per_class

    def eval(self):
        return self

    def forward_image(self, image: torch.Tensor, train_stdp: bool) -> torch.Tensor:
        # image: (B, 1, 28, 28); usamos só o batch size, ignoramos pixels
        # Cada imagem precisa carregar seu label pra retornar a resposta certa.
        # Como o DataLoader passa só image, precisamos truque: o "image" aqui
        # carrega o label codificado no primeiro pixel.
        labels = image[:, 0, 0, 0].long()  # (B,)
        return self.response_per_class[labels]  # (B, n_filters)


def make_fake_loader(labels: torch.Tensor, batch_size: int = 16) -> DataLoader:
    """Cria DataLoader onde cada 'imagem' é um tensor (1,28,28) com o label
    codificado no pixel [0,0,0] — reaproveitado pelo FakeNet."""
    n = labels.shape[0]
    fake_imgs = torch.zeros(n, 1, 28, 28)
    fake_imgs[:, 0, 0, 0] = labels.float()
    ds = TensorDataset(fake_imgs, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Casos de teste
# ---------------------------------------------------------------------------
def test_perfect_features():
    """
    Caso 1: features perfeitas — cada filtro f responde forte só pra classe (f mod 10).
    Distribuição imitando Config A: [36,40,6,11,18,19,17,43,2,8] = 200 filtros.
    Esperado: 100% acurácia (sinal sem ambiguidade).
    """
    print("\n[Test 1] Features perfeitas com distribuição Config A")
    n_classes = 10
    distribution = [36, 40, 6, 11, 18, 19, 17, 43, 2, 8]
    n_filters = sum(distribution)
    assert n_filters == 200

    # Cada filtro f está atribuído a uma classe específica conforme distribution
    # filter_to_class[f] = qual classe esse filtro "deveria" representar
    filter_to_class = []
    for c, count in enumerate(distribution):
        filter_to_class.extend([c] * count)
    filter_to_class = torch.tensor(filter_to_class)

    # response_per_class[c, f] = 10.0 se filter_to_class[f] == c, else 0.0
    response_per_class = torch.zeros(n_classes, n_filters)
    for f in range(n_filters):
        response_per_class[filter_to_class[f], f] = 10.0

    net = FakeNet(n_filters, response_per_class)

    # Loader balanceado: 100 imagens por classe
    labels = torch.arange(n_classes).repeat_interleave(100)
    loader = make_fake_loader(labels)

    device = torch.device("cpu")
    learned_labels = assign_labels(net, loader, device, n_classes)
    print(f"  Distribuição inferida: {torch.bincount(learned_labels, minlength=10).tolist()}")
    print(f"  Esperada:              {distribution}")

    acc = evaluate(net, loader, learned_labels, device, n_classes)
    print(f"  Acurácia: {acc*100:.2f}%")

    assert torch.equal(torch.bincount(learned_labels, minlength=10),
                       torch.tensor(distribution)), \
        "BUG: assign_labels não recuperou a distribuição esperada"
    assert acc >= 0.99, f"BUG: acurácia {acc*100:.2f}% << 100% com features perfeitas"
    print("  ✓ PASSOU — assign_labels e evaluate funcionam com features perfeitas")
    return True


def test_random_features():
    """
    Caso 2: features ruído puro — todos filtros respondem igual a todas as classes.
    Esperado: ~chance (10% ± ruído).
    """
    print("\n[Test 2] Features sem sinal discriminativo (ruído)")
    n_classes = 10
    n_filters = 200

    # Todos filtros têm resposta uniforme em todas as classes
    response_per_class = torch.ones(n_classes, n_filters) * 5.0

    net = FakeNet(n_filters, response_per_class)
    labels = torch.arange(n_classes).repeat_interleave(100)
    loader = make_fake_loader(labels)

    device = torch.device("cpu")
    learned_labels = assign_labels(net, loader, device, n_classes)
    print(f"  Distribuição inferida: {torch.bincount(learned_labels, minlength=10).tolist()}")

    acc = evaluate(net, loader, learned_labels, device, n_classes)
    print(f"  Acurácia: {acc*100:.2f}%")
    # Sem sinal, argmax sobre tensor de 1s vai sempre escolher classe 0 (tie-break)
    # Isso resulta em acurácia = % de classe 0 no test set = 10%.
    assert 0.05 <= acc <= 0.15, f"Acurácia {acc*100:.2f}% fora do range esperado de chance"
    print("  ✓ PASSOU — assignment correctly produces chance accuracy with no signal")
    return True


def test_weak_signal():
    """
    Caso 3: sinal fraco realista — filtros têm viés sutil pra sua classe atribuída,
    mas disparam 80% do que disparam pra outras. Mais perto do que STDP real produz.
    Esperado: > chance, < 100% (digamos 30-90%).
    """
    print("\n[Test 3] Features com sinal fraco (80% baseline + 20% boost na classe certa)")
    n_classes = 10
    distribution = [36, 40, 6, 11, 18, 19, 17, 43, 2, 8]
    n_filters = sum(distribution)

    filter_to_class = []
    for c, count in enumerate(distribution):
        filter_to_class.extend([c] * count)
    filter_to_class = torch.tensor(filter_to_class)

    # Baseline 5.0 pra todas as combinações; boost +1.0 quando filter_to_class[f] == c
    response_per_class = torch.ones(n_classes, n_filters) * 5.0
    for f in range(n_filters):
        response_per_class[filter_to_class[f], f] += 1.0

    net = FakeNet(n_filters, response_per_class)
    labels = torch.arange(n_classes).repeat_interleave(100)
    loader = make_fake_loader(labels)

    device = torch.device("cpu")
    learned_labels = assign_labels(net, loader, device, n_classes)
    print(f"  Distribuição inferida: {torch.bincount(learned_labels, minlength=10).tolist()}")

    acc = evaluate(net, loader, learned_labels, device, n_classes)
    print(f"  Acurácia: {acc*100:.2f}%")
    assert acc > 0.5, f"Sinal fraco com boost de 20% deveria dar > 50%, deu {acc*100:.2f}%"
    print("  ✓ PASSOU — sinal fraco recuperado corretamente")
    return True


# ---------------------------------------------------------------------------
# Diagnóstico final
# ---------------------------------------------------------------------------
def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 70)
    print("Auditoria de assign_labels e evaluate em sanity_mnist.py")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")

    results = []
    for name, fn in [("perfect", test_perfect_features),
                     ("random", test_random_features),
                     ("weak_signal", test_weak_signal)]:
        try:
            ok = fn()
            results.append((name, ok))
        except AssertionError as e:
            print(f"  ✗ FALHOU: {e}")
            results.append((name, False))

    print("\n" + "=" * 70)
    print("Resumo:")
    for name, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")

    all_passed = all(ok for _, ok in results)
    print()
    if all_passed:
        print("CONCLUSÃO: assign_labels e evaluate estão CORRETOS.")
        print("9.94% em Config A é sinal de que features STDP não carregam")
        print("informação discriminativa — não é bug no pipeline de classificação.")
        print("Próximo passo: atacar dinâmica STDP (LTP/LTD desbalance).")
        return 0
    else:
        print("CONCLUSÃO: BUG encontrado em assign_labels ou evaluate.")
        print("Corrigir antes de mexer em hiperparâmetros STDP.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
