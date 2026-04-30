"""
Possibilidade C — STDP within-task + C2 between-task (two-timescale).

Status: SCAFFOLD (não implementado). Sessão #27 criou estrutura.

Arquitetura two-timescale:
  Within-task (apresentação dos 14 chars do alfabeto sequencialmente):
    → STDP atualiza pesos do encoder com plasticidade local rápida.
    → Sem gradiente. Pesos drifam baseado em pre/post correlations locais.
    → Captura adaptação à task atual de forma online.

  Between-task (após cada task t completa):
    → C2 outer-loop gradient consolida ajustes.
    → Cross-entropy do query da task atual flui pra meta-parâmetros.
    → Meta-parâmetros (A, B, C, D) ajustados pra fazer STDP futuro
      preservar features das tasks anteriores.

Hipótese específica:
  STDP rápido captura "o que esta task pede"; C2 lento previne forgetting
  via consolidação do que ficou estável entre tasks. Inspirado em
  Complementary Learning Systems (CLS, McClelland 1995): hipocampo
  rápido + córtex lento.

Risco principal:
  Two-timescale é difícil de tunar:
  - Taxa relativa STDP/C2 (dois etas)
  - Quando "consolidar" (após cada chars? após cada task?)
  - Gradiente do C2 não pode interferir nos pesos do STDP diretamente
    (senão vira backprop padrão); precisa interface clara.

Métricas-alvo (vs baseline 80.65% / -9.26):
  Sucesso: ACC ≥ 84% E BWT ≥ -5
  Mediano: ACC 81-84%
  Falha: ACC < 81%

Ablações pré-definidas:
  C1: Remover STDP within-task (mantém só C2 between) — testa se
      decoupling temporal agrega
  C2_ablacao: Remover C2 between (mantém só STDP within) — testa se
      consolidação between-task é necessária
  C3_ablacao: Variar quando consolidar (após cada char, cada metade da
      task, cada task)

Não implementado nesta sessão. Sessão #35+ se Possibilidades A e B
passarem sanity e mostrarem direção promissora.
"""
# TODO: implementar após validação de Possibilidades B e A (sessão #27 scaffold)


def main():
    raise NotImplementedError(
        "Possibilidade C scaffold (sessão #27). "
        "Implementação posterior dependente de A e B mostrarem direção viável."
    )


if __name__ == "__main__":
    main()
