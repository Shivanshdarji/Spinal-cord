# Benchmark currently running llama-server via OpenAI-compatible chat endpoint.
# Run this script twice:
#   1) with draft ON  -> dashboard\run_dashboard_llama_scaffold.bat
#   2) with draft OFF -> dashboard\run_dashboard_llama_scaffold_brain_only.bat
# Compare printed means.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\benchmark_llama_server.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\benchmark_llama_server.ps1 -Runs 8 -MaxTokens 128

param(
    [string] $BindHost = "127.0.0.1",
    [int] $Port = 8080,
    [string] $ModelId = "",
    [int] $Runs = 6,
    [int] $MaxTokens = 128,
    [double] $Temperature = 0.2,
    [double] $RepeatPenalty = 1.15,
    [string] $Prompt = "Explain recursion in three short bullet points."
)

function Resolve-ModelId($base, $forcedId) {
    if ($forcedId -and $forcedId.Trim().Length -gt 0) { return $forcedId.Trim() }
    $m = Invoke-RestMethod -Uri "$base/v1/models" -Method Get -TimeoutSec 10
    $ids = @()
    if ($m.data) {
        foreach ($e in $m.data) {
            if ($e.id) { $ids += [string]$e.id }
            elseif ($e.name) { $ids += [string]$e.name }
            elseif ($e.model) { $ids += [string]$e.model }
        }
    } elseif ($m.models) {
        foreach ($e in $m.models) {
            if ($e.id) { $ids += [string]$e.id }
            elseif ($e.name) { $ids += [string]$e.name }
            elseif ($e.model) { $ids += [string]$e.model }
        }
    }
    if (-not $ids -or $ids.Count -eq 0) { throw "No model id found at $base/v1/models" }
    foreach ($id in $ids) {
        if ($id.ToLower().Contains("brain") -and -not $id.ToLower().Contains("draft")) { return $id }
    }
    foreach ($id in $ids) {
        if (-not $id.ToLower().Contains("draft")) { return $id }
    }
    return $ids[0]
}

$base = "http://${BindHost}:${Port}"
Write-Host "Benchmarking $base ..." -ForegroundColor Cyan

try {
    $resolvedId = Resolve-ModelId -base $base -forcedId $ModelId
} catch {
    Write-Host "Model resolution failed: $_" -ForegroundColor Red
    exit 1
}
Write-Host "Using model id: $resolvedId" -ForegroundColor Green

$wall = @()
$predTokPerSec = @()
$promptTokPerSec = @()
$outputTok = @()

for ($i = 1; $i -le $Runs; $i++) {
    $bodyObj = @{
        model = $resolvedId
        messages = @(@{ role = "user"; content = $Prompt })
        stream = $false
        temperature = $Temperature
        max_tokens = $MaxTokens
        repeat_penalty = $RepeatPenalty
    }
    $json = $bodyObj | ConvertTo-Json -Compress

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $json -TimeoutSec 240
    } catch {
        Write-Host "Run $i failed: $_" -ForegroundColor Red
        exit 2
    } finally {
        $sw.Stop()
    }

    $w = [double]$sw.Elapsed.TotalSeconds
    $wall += $w

    if ($resp.timings -and $resp.timings.predicted_per_second) {
        $predTokPerSec += [double]$resp.timings.predicted_per_second
    }
    if ($resp.timings -and $resp.timings.prompt_per_second) {
        $promptTokPerSec += [double]$resp.timings.prompt_per_second
    }
    if ($resp.usage -and $resp.usage.completion_tokens) {
        $outputTok += [double]$resp.usage.completion_tokens
    }

    $preview = ""
    try { $preview = [string]$resp.choices[0].message.content } catch {}
    if ($preview.Length -gt 80) { $preview = $preview.Substring(0, 80) + "..." }
    Write-Host ("Run {0}/{1}  wall={2:N2}s  outTok={3}  preview={4}" -f $i, $Runs, $w, ($resp.usage.completion_tokens), $preview)
}

function Mean($arr) {
    if (-not $arr -or $arr.Count -eq 0) { return 0.0 }
    $sum = 0.0
    foreach ($x in $arr) { $sum += [double]$x }
    return $sum / [double]$arr.Count
}

function Stdev($arr) {
    if (-not $arr -or $arr.Count -lt 2) { return 0.0 }
    $m = Mean $arr
    $acc = 0.0
    foreach ($x in $arr) {
        $d = [double]$x - $m
        $acc += $d * $d
    }
    return [Math]::Sqrt($acc / ($arr.Count - 1))
}

$wallMean = Mean $wall
$wallSd = Stdev $wall
$outMean = Mean $outputTok
$predMean = Mean $predTokPerSec
$predSd = Stdev $predTokPerSec
$promptMean = Mean $promptTokPerSec
$wallTokPerSec = if ($wallMean -gt 0) { $outMean / $wallMean } else { 0.0 }

Write-Host ""
Write-Host "=== llama-server benchmark summary ===" -ForegroundColor Yellow
Write-Host ("Model:                {0}" -f $resolvedId)
Write-Host ("Runs:                 {0}" -f $Runs)
Write-Host ("Completion tok mean:  {0:N1}" -f $outMean)
Write-Host ("Wall sec mean±sd:     {0:N2} ± {1:N2}" -f $wallMean, $wallSd)
Write-Host ("Wall tok/s (mean):    {0:N1}" -f $wallTokPerSec)
Write-Host ("Predicted tok/s mean: {0:N1} ± {1:N1}" -f $predMean, $predSd)
Write-Host ("Prompt tok/s mean:    {0:N1}" -f $promptMean)
Write-Host ""
Write-Host "Now run this same script in the other mode (draft on/off) and compare wall tok/s + predicted tok/s."
