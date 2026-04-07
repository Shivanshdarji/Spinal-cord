# Quick check: llama-server on localhost responds for /v1/models and POST chat.
# Usage: .\scripts\verify_llama_server.ps1
#        .\scripts\verify_llama_server.ps1 -Port 8080 -ModelId "scbrain_1b.gguf"

param(
    [string] $BindHost = "127.0.0.1",
    [int] $Port = 8080,
    [string] $ModelId = "scbrain_1b.gguf"
)

$base = "http://${BindHost}:${Port}"
Write-Host "Checking $base ..." -ForegroundColor Cyan

try {
    $models = Invoke-RestMethod -Uri "$base/v1/models" -Method Get -TimeoutSec 10
    Write-Host "GET /v1/models OK" -ForegroundColor Green
    $models | ConvertTo-Json -Depth 6 | Write-Host
} catch {
    Write-Host "GET /v1/models FAILED: $_" -ForegroundColor Red
    exit 1
}

# Lower temperature + repeat_penalty reduce junk loops on small checkpoints (same knobs as dashboard).
$body = @{
    model            = $ModelId
    messages         = @(@{ role = "user"; content = "Reply with exactly: OK" })
    stream           = $false
    temperature      = 0.2
    max_tokens       = 64
    repeat_penalty   = 1.15
} | ConvertTo-Json -Compress

try {
    $chat = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post `
        -ContentType "application/json; charset=utf-8" -Body $body -TimeoutSec 120
    Write-Host "POST /v1/chat/completions OK" -ForegroundColor Green
    $chat | ConvertTo-Json -Depth 8 | Write-Host
} catch {
    Write-Host "POST /v1/chat/completions FAILED: $_" -ForegroundColor Red
    if ($_.Exception.Response) {
        $r = $_.Exception.Response
        Write-Host "Status:" $r.StatusCode.value__
    }
    exit 2
}

Write-Host "Done." -ForegroundColor Green
