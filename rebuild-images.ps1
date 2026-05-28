param(
  [switch]$NoCache
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$InfraComposeFile = Join-Path $Root "infra\docker-compose.yml"
$EnvFile = Join-Path $Root ".env"

function Invoke-Compose {
  param(
    [string[]]$Arguments
  )

  $effectiveArguments = $Arguments
  if ($Arguments.Length -gt 0 -and $Arguments[0] -eq "compose" -and (Test-Path $EnvFile)) {
    $remainingArguments = @()
    if ($Arguments.Length -gt 1) {
      $remainingArguments = $Arguments[1..($Arguments.Length - 1)]
    }
    $effectiveArguments = @("compose", "--env-file", $EnvFile) + $remainingArguments
  }

  & docker @effectiveArguments
  if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose build failed."
  }
}

if (-not (Test-Path $InfraComposeFile)) {
  throw "Docker Compose file not found: $InfraComposeFile"
}

$buildArgs = @("compose", "-f", $InfraComposeFile, "build")
if ($NoCache) {
  $buildArgs += "--no-cache"
}
$buildArgs += @("api", "worker", "web")

Write-Host "Rebuilding application images (api, worker, web)..." -ForegroundColor Cyan
Invoke-Compose -Arguments $buildArgs
Write-Host "Done. Images rebuilt. Start services with start-app.bat" -ForegroundColor Green
