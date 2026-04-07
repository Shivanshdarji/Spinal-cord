# Run elevated (Administrator) to allow inbound TCP to llama-server.
param(
  [int]$Port = 8080
)
New-NetFirewallRule -DisplayName "SpinalCord llama-server TCP $Port" `
  -Direction Inbound -LocalPort $Port -Protocol TCP -Action Allow -ErrorAction Stop
Write-Host "Opened inbound TCP $Port"
