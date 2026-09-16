param(
  [int]$FrontendPort = 3000,
  [string]$ComposeProjectName = "knowledgegraph-dev-20260820",
  [switch]$KeepBackend
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path $Root "output\web-dev.pid.json"
$EnvFile = Join-Path $Root ".env"
$ComposeFile = Join-Path $Root "infra\docker-compose.yml"

function Stop-RecordedWebProcess {
  if (-not (Test-Path -LiteralPath $PidFile -PathType Leaf)) {
    Write-Host "Native Web pid record is absent; nothing to stop." -ForegroundColor DarkGray
    return
  }
  $record = Get-Content -Raw -Encoding UTF8 -LiteralPath $PidFile | ConvertFrom-Json
  $processId = [int]$record.pid
  if (
    [string]$record.protocol_version -ne "symbograph_native_web_pid_v1" -or
    [string]$record.root -ne $Root -or
    [int]$record.port -ne $FrontendPort
  ) {
    throw "Refusing to use a native Web pid record from another workspace or port."
  }
  $owner = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
  if ($null -eq $owner) {
    Remove-Item -LiteralPath $PidFile -Force
    return
  }
  $commandLine = [string]$owner.CommandLine
  if ($commandLine -notmatch "npm" -or $commandLine -notmatch "workspace\s+web") {
    throw "Refusing to stop PID $processId because it is not the recorded npm Web launcher."
  }
  $descendants = @()
  $frontier = @($processId)
  while ($frontier.Count -gt 0) {
    $parentId = [int]$frontier[0]
    $frontier = @($frontier | Select-Object -Skip 1)
    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $parentId" -ErrorAction SilentlyContinue)
    foreach ($child in $children) {
      $descendants += [int]$child.ProcessId
      $frontier += [int]$child.ProcessId
    }
  }
  [array]::Reverse($descendants)
  foreach ($childId in @($descendants | Select-Object -Unique)) {
    Stop-Process -Id $childId -Force -ErrorAction SilentlyContinue
  }
  Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $PidFile -Force
  Write-Host "Native Web process stopped." -ForegroundColor Green
}

Stop-RecordedWebProcess

if (-not $KeepBackend) {
  $arguments = @("compose")
  if (Test-Path -LiteralPath $EnvFile -PathType Leaf) {
    $arguments += @("--env-file", $EnvFile)
  }
  $arguments += @(
    "--project-name", $ComposeProjectName,
    "-f", $ComposeFile,
    "--profile", "model-bridge",
    "down", "--remove-orphans"
  )
  & docker @arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose shutdown failed."
  }
  Write-Host "Backend containers stopped." -ForegroundColor Green
}
