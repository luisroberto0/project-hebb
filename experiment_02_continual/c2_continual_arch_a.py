"""
Possibilidade A — STDP biofísico nas camadas iniciais + C2 nas camadas finais.

Status: SCAFFOLD (não implementado). Sessão #27 criou estrutura; implementação
fica pra sessão posterior se Possibilidade B sanity passar e B completo
mostrar promessa.

Arquitetura:
  Input (1, 28, 28)
    → Conv layer 1: STDP biofísico (Diehl & Cook 2015) + LIF + k-WTA
    → Conv layer 2: STDP biofísico + LIF + k-WTA
    → Pool
    → Linear projection → 32D embedding via plasticidade C2
      (A·pre·post + B·pre + C·post + D, A/B/C/D meta-aprendidos)
    → Prototype-based classification (cosine sim, β=8)

Hipótese específica:
  STDP captura features genéricas resilientes a forgetting (updates locais,
  sem gradiente cross-task). C2 captura especificidade da task atual via
  meta-learning. Combinação: features estáveis + adaptação rápida.

Risco principal:
  STDP biofísico mostrou barreira em sessões #1-#13 (matched filter trivial).
  Pode contribuir 0 ou negativamente em CL também.

Métricas-alvo (vs baseline naive ProtoNet 80.65% / BWT -9.26):
  Sucesso: ACC ≥ 84% E BWT ≥ -5
  Mediano: ACC 81-84% (margem dentro do IC95% do baseline)
  Falha: ACC < 81%

Ablações pré-definidas (executar se A passar baseline):
  A1: Remover STDP layers (substituir por random fixos) — testa se STDP
      contribui ou se features fixas + C2 já entregam
  A2: Inverter ordem (C2 early + STDP late) — testa direcionalidade
  A3: Variar tau_pre/post do STDP (10, 20, 40 ms)

Inner/outer loop:
  Inner: STDP atualiza camadas 1-2 durante apresentação dos 5 supports
         (n_inner=5 forward passes). C2 inner também aplica.
  Outer: cross-entropy do query backprop através de C2 (gradient flui)
         e através de STDP layers (mas STDP não tem requires_grad → pesos
         de STDP são CONSEQUÊNCIA da apresentação, não meta-aprendidos).

Não implementado nesta sessão. Sessão #28+ se Possibilidade B passar sanity.
"""
# TODO: implementar após validação de Possibilidade B (sessão #27 scaffold)


def main():
    raise NotImplementedError(
        "Possibilidade A scaffold (sessão #27). "
        "Implementação posterior dependente de Possibilidade B passar sanity."
    )


if __name__ == "__main__":
    main()
