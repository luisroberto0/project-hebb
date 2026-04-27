# run_all.ps1 — pipeline automatizado das 6 semanas do experimento 01
#
# Uso:
#   pwsh -File experiment_01_oneshot/run_all.ps1
#
# Opções via env vars:
#   $env:HEBB_QUICK="1"    # debug rápido (subset reduzido em todas as fases)
#   $env:HEBB_SKIP_DONE="1"   # pula etapa cujo output já existe (idempotente)
#
# Saída: logs/run_<timestamp>/ com stdout de cada fase + RESULTS.md final.
#
# IMPORTANTE: o script assume que evaluate.py e baselines.py foram
# adaptados pra suportar `--json-out <path>` produzindo o formato que
# analysis.py espera (ver topo de analysis.py). Se ainda não foram, o
# Claude Code CLI no host pode adicionar essa flag — é apertar uma
# linha em cada script.

$ErrorActionPreference = "Stop"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "logs/run_$ts"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path "checkpoints" | Out-Null
New-Item -ItemType Directory -Force -Path "data" | Out-Null

Write-Host "=== Project Hebb — Experimento 01: pipeline automatizado ==="
Write-Host "Logs em: $logDir"
Write-Host ""

$quick = ($env:HEBB_QUICK -eq "1")
$skipDone = ($env:HEBB_SKIP_DONE -eq "1")

function Run-Phase {
    param([string]$name, [string]$cmd, [string]$skipIfExists = "")
    if ($skipDone -and $skipIfExists -and (Test-Path $skipIfExists)) {
        Write-Host "  [skip] $name (output já existe: $skipIfExists)"
        return
    }
    Write-Host "  [run]  $name"
    $logFile = "$logDir/$($name -replace '[^\w]', '_').log"
    $start = Get-Date
    Invoke-Expression "$cmd 2>&1 | Tee-Object -FilePath '$logFile'"
    $elapsed = (Get-Date) - $start
    Write-Host "         tempo: $([math]::Round($elapsed.TotalSeconds, 1))s"
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Fase 0 — Validação de ambiente
# ---------------------------------------------------------------------------
Write-Host "=== Fase 0: validação de ambiente ==="
Run-Phase "validate_env" "python ..\validate_environment.py"

# ---------------------------------------------------------------------------
# Pré-Semana-1 — Pipeline + baselines (sem STDP)
# ---------------------------------------------------------------------------
Write-Host "=== Pré-Semana-1: pipeline end-to-end + baselines ==="
$episodes = if ($quick) { 100 } else { 1000 }
$trainEps = if ($quick) { 500 } else { 5000 }

Run-Phase "eval_random_5w1s" `
    "python evaluate.py --ways 5 --shots 1 --episodes 100"

Run-Phase "baseline_pixel_knn" `
    "python baselines.py --baseline pixel_knn --ways 5 --shots 1 --episodes $episodes"

Run-Phase "baseline_proto_net" `
    "python baselines.py --baseline proto_net --ways 5 --shots 1 --episodes $episodes --train-episodes $trainEps"

# ---------------------------------------------------------------------------
# Semana 1 — Sanity check Diehl & Cook 2015 em MNIST
# ---------------------------------------------------------------------------
Write-Host "=== Semana 1: sanity check (Diehl & Cook 2015 em MNIST) ==="
$nImgsSanity = if ($quick) { 1000 } else { 10000 }
Run-Phase "sanity_mnist" `
    "python sanity_mnist.py --n-images $nImgsSanity --epochs 1" `
    "checkpoints/sanity_mnist.pt"

# ---------------------------------------------------------------------------
# Semanas 2-3 — Pretreino STDP em Omniglot
# ---------------------------------------------------------------------------
Write-Host "=== Semanas 2-3: pretreino STDP em Omniglot ==="
$nImgsPretrain = if ($quick) { 500 } else { 24000 }
Run-Phase "pretrain_stdp" `
    "python train.py --n-images $nImgsPretrain --epochs 1 --log-dir $logDir/tb" `
    "checkpoints/stdp_model.pt"

# Visualizações pós-pretreino
Run-Phase "viz_filters" `
    "python -m utils.visualize --checkpoint checkpoints/stdp_model.pt --out-dir $logDir/figs"

# ---------------------------------------------------------------------------
# Semanas 4-5 — Avaliação completa em 4 configs
# ---------------------------------------------------------------------------
Write-Host "=== Semanas 4-5: avaliação completa N-way K-shot ==="
$evalEpisodes = if ($quick) { 100 } else { 1000 }

foreach ($cfg in @(@(5,1), @(5,5), @(20,1), @(20,5))) {
    $w = $cfg[0]; $s = $cfg[1]
    $jsonOut = "$logDir/eval_${w}w${s}s.json"
    Run-Phase "eval_${w}w${s}s" `
        "python evaluate.py --checkpoint checkpoints/stdp_model.pt --ways $w --shots $s --episodes $evalEpisodes --json-out $jsonOut" `
        $jsonOut
}

# ---------------------------------------------------------------------------
# Semana 6 — Análise + RESULTS.md
# ---------------------------------------------------------------------------
Write-Host "=== Semana 6: análise e RESULTS.md ==="
Run-Phase "analysis" `
    "python analysis.py --logs-dir $logDir --out RESULTS.md"

Write-Host ""
Write-Host "=========================================="
Write-Host "Pipeline concluído. Outputs:"
Write-Host "  - Logs:      $logDir/"
Write-Host "  - RESULTS:   RESULTS.md"
Write-Host "  - Filtros:   $logDir/figs/"
Write-Host "  - TensorBoard: $logDir/tb/"
Write-Host "=========================================="
Write-Host ""
Write-Host "Próximo: ler RESULTS.md, decidir entre paper rascunho / NEXT.md / POSTMORTEM.md."
