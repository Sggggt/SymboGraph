param(
  [int]$FrontendPort = 3000,
  [int]$BackendPort = 8000,
  [switch]$Production
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutputRoot = Join-Path $Root "output"
$PidFile = Join-Path $OutputRoot "web-dev.pid.json"

$nodeVersion = (& node --version 2>$null)
if ($LASTEXITCODE -ne 0 -or $nodeVersion -notmatch '^v(\d+)\.(\d+)\.') {
  throw "Node.js 20.9 or newer is required for Next.js 16.2.4."
}
if ([int]$Matches[1] -lt 20 -or ([int]$Matches[1] -eq 20 -and [int]$Matches[2] -lt 9)) {
  throw "Node.js 20.9 or newer is required; found $nodeVersion."
}
if (-not (Test-Path -LiteralPath (Join-Path $Root "node_modules\next\package.json") -PathType Leaf)) {
  & npm.cmd ci
  if ($LASTEXITCODE -ne 0) {
    throw "npm ci failed."
  }
}
if (Test-Path -LiteralPath $PidFile -PathType Leaf) {
  & (Join-Path $Root "stop-app.ps1") -FrontendPort $FrontendPort -KeepBackend
}
$listener = Get-NetTCPConnection -LocalPort $FrontendPort -State Listen -ErrorAction SilentlyContinue
if ($listener) {
  throw "Port $FrontendPort is already occupied."
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdout = Join-Path $OutputRoot "web-$stamp.stdout.log"
$stderr = Join-Path $OutputRoot "web-$stamp.stderr.log"
$env:NEXT_PUBLIC_API_BASE_URL = "http://127.0.0.1:$BackendPort/api"
$scriptName = if ($Production) { "start" } else { "dev" }
$arguments = @("run", $scriptName, "--workspace", "web", "--", "--hostname", "127.0.0.1", "--port", [string]$FrontendPort)
$process = Start-Process -FilePath "npm.cmd" -ArgumentList $arguments -WorkingDirectory $Root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
@{
  protocol_version = "symbograph_native_web_pid_v1"
  pid = $process.Id
  root = $Root
  port = $FrontendPort
  started_at = (Get-Date).ToUniversalTime().ToString("o")
  mode = $scriptName
  stdout = $stdout
  stderr = $stderr
} | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $PidFile

$deadline = (Get-Date).AddSeconds(60)
$ready = $false
while ((Get-Date) -lt $deadline) {
  try {
    $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$FrontendPort/" -TimeoutSec 2
    if ($response.StatusCode -eq 200) {
      $ready = $true
      break
    }
  } catch {
  }
  Start-Sleep -Milliseconds 500
}
if (-not $ready) {
  throw "Native Web did not become ready. Inspect $stderr"
}
Write-Host "Native Web is ready at http://127.0.0.1:$FrontendPort/ (PID $($process.Id), mode $scriptName)." -ForegroundColor Green
