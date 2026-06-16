param(
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 3000,
  [string]$OpenPath = "/graph",
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile = Join-Path $Root ".env"
$InfraComposeFile = Join-Path $Root "infra\docker-compose.yml"

function Get-DotEnvValue {
  param(
    [string]$Key,
    [string]$DefaultValue
  )

  if (-not (Test-Path $EnvFile)) {
    return $DefaultValue
  }

  $prefix = "$Key="
  foreach ($line in Get-Content -Encoding UTF8 -LiteralPath $EnvFile) {
    $cleanLine = $line.TrimStart([char]0xFEFF).Trim()
    if (-not $cleanLine -or $cleanLine.StartsWith("#")) {
      continue
    }
    if ($cleanLine.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
      return $cleanLine.Substring($prefix.Length).Trim().Trim('"').Trim("'")
    }
  }

  return $DefaultValue
}

function Test-Url {
  param([string]$Url)
  try {
    $null = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
    return $true
  } catch {
    return $false
  }
}

function Wait-Url {
  param(
    [string]$Url,
    [string]$Name,
    [int]$TimeoutSeconds = 120
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    if (Test-Url $Url) {
      Write-Host "$Name is ready: $Url" -ForegroundColor Green
      return
    }
    Start-Sleep -Seconds 1
  }

  throw "$Name did not become ready within $TimeoutSeconds seconds: $Url"
}

function Wait-ContainerHealthy {
  param(
    [string]$ContainerName,
    [string]$Name,
    [int]$TimeoutSeconds = 120
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    $status = (& docker inspect --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}" $ContainerName 2>$null)
    if ($LASTEXITCODE -eq 0 -and ($status -eq "healthy" -or $status -eq "running")) {
      Write-Host "$Name is ready: $ContainerName" -ForegroundColor Green
      return
    }
    Start-Sleep -Seconds 1
  }

  throw "$Name did not become ready within $TimeoutSeconds seconds: $ContainerName"
}

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
    throw "Docker Compose failed. If an application image is missing, build it first using the README commands."
  }
}

function Get-DotEnvBool {
  param(
    [string]$Key,
    [bool]$DefaultValue
  )

  $rawValue = (Get-DotEnvValue -Key $Key -DefaultValue ($(if ($DefaultValue) { "true" } else { "false" }))).ToLowerInvariant()
  if ($rawValue -in @("true", "1", "yes", "on")) {
    return $true
  }
  if ($rawValue -in @("false", "0", "no", "off")) {
    return $false
  }
  throw "Unsupported $Key='$rawValue'. Use true or false."
}

if (-not (Test-Path $InfraComposeFile)) {
  throw "Docker Compose file not found: $InfraComposeFile"
}

$modelBridgeEnabled = Get-DotEnvBool -Key "MODEL_BRIDGE_ENABLED" -DefaultValue $false
$modelBridgePortRaw = Get-DotEnvValue -Key "MODEL_BRIDGE_PORT" -DefaultValue "8765"
$chatBaseUrl = Get-DotEnvValue -Key "CHAT_BASE_URL" -DefaultValue "https://api.openai.com/v1"
$chatResolveIp = Get-DotEnvValue -Key "CHAT_RESOLVE_IP" -DefaultValue ""
$embeddingBaseUrl = Get-DotEnvValue -Key "EMBEDDING_BASE_URL" -DefaultValue ""
$embeddingResolveIp = Get-DotEnvValue -Key "EMBEDDING_RESOLVE_IP" -DefaultValue ""
$modelBridgeAdminToken = Get-DotEnvValue -Key "MODEL_BRIDGE_ADMIN_TOKEN" -DefaultValue "local-model-bridge-admin"
try {
  $modelBridgePort = [int]$modelBridgePortRaw
} catch {
  throw "Unsupported MODEL_BRIDGE_PORT='$modelBridgePortRaw'. Use an integer port."
}
if ($modelBridgeEnabled -and ($modelBridgePort -lt 1 -or $modelBridgePort -gt 65535)) {
  throw "Unsupported MODEL_BRIDGE_PORT='$modelBridgePort'. Use a port between 1 and 65535."
}

$BackendUrl = "http://127.0.0.1:$BackendPort/api/health"
$FrontendUrl = "http://127.0.0.1:$FrontendPort$OpenPath"
$env:API_HOST_PORT = [string]$BackendPort
$env:WEB_HOST_PORT = [string]$FrontendPort
$env:CHAT_BASE_URL = $chatBaseUrl
$env:CHAT_RESOLVE_IP = $chatResolveIp
$env:EMBEDDING_BASE_URL = $embeddingBaseUrl
$env:EMBEDDING_RESOLVE_IP = $embeddingResolveIp
$env:MODEL_BRIDGE_ADMIN_TOKEN = $modelBridgeAdminToken

function Normalize-BridgeBaseUrl {
  param([string]$Value)
  if ([string]::IsNullOrWhiteSpace($Value)) {
    return ""
  }
  return $Value.Trim().TrimEnd("/")
}

function Normalize-BridgeResolveIp {
  param([string]$Value)
  if ([string]::IsNullOrWhiteSpace($Value) -or $Value -eq "__none__") {
    return ""
  }
  return $Value.Trim()
}

function Test-ModelBridgeSelfTarget {
  param(
    [string]$Url,
    [int]$Port
  )
  if ([string]::IsNullOrWhiteSpace($Url)) {
    return $false
  }
  try {
    $uri = [System.Uri]$Url
  } catch {
    return $false
  }
  $hostName = $uri.Host.ToLowerInvariant()
  if ($hostName -notin @("host.docker.internal", "127.0.0.1", "localhost", "::1", "0.0.0.0")) {
    return $false
  }
  return $uri.Port -eq $Port
}

function Invoke-BridgeAdminJson {
  param(
    [string]$Uri,
    [string]$Method = "GET",
    [object]$Body = $null
  )
  $headers = @{ "X-Bridge-Admin-Token" = $modelBridgeAdminToken }
  if ($null -eq $Body) {
    return Invoke-RestMethod -Uri $Uri -Method $Method -Headers $headers -TimeoutSec 5
  }
  $jsonBody = $Body | ConvertTo-Json -Depth 6
  return Invoke-RestMethod -Uri $Uri -Method $Method -Headers $headers -ContentType "application/json" -Body $jsonBody -TimeoutSec 10
}

function Stop-ModelBridgeOnPort {
  param([int]$Port)
  try {
    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($connection in $connections) {
      if ($connection.OwningProcess) {
        Write-Host "Stopping existing model bridge process on port ${Port}: PID $($connection.OwningProcess)" -ForegroundColor Yellow
        Stop-Process -Id $connection.OwningProcess -Force -ErrorAction SilentlyContinue
      }
    }
    Start-Sleep -Milliseconds 500
  } catch {
    Write-Host "Could not stop existing bridge process on port $Port automatically: $($_.Exception.Message)" -ForegroundColor Yellow
  }
}

function Start-ModelBridge {
  param(
    [string]$BridgeScript,
    [object]$PythonCommand,
    [int]$Port
  )

  if ([string]::IsNullOrWhiteSpace($chatBaseUrl)) {
    throw "MODEL_BRIDGE_ENABLED=true requires CHAT_BASE_URL."
  }
  if ([string]::IsNullOrWhiteSpace($embeddingBaseUrl)) {
    throw "MODEL_BRIDGE_ENABLED=true requires EMBEDDING_BASE_URL. The bridge no longer reuses CHAT_BASE_URL for embeddings."
  }
  if (Test-ModelBridgeSelfTarget -Url $chatBaseUrl -Port $Port) {
    throw "CHAT_BASE_URL must be the real chat provider URL, not the model bridge URL."
  }
  if (Test-ModelBridgeSelfTarget -Url $embeddingBaseUrl -Port $Port) {
    throw "EMBEDDING_BASE_URL must be the real embedding provider URL, not the model bridge URL."
  }

  $bridgeArgs = @(
    $BridgeScript,
    "--host", "127.0.0.1",
    "--port", [string]$Port,
    "--chat-target-base-url", $chatBaseUrl,
    "--embedding-target-base-url", $embeddingBaseUrl,
    "--admin-token", $modelBridgeAdminToken
  )
  if ($chatResolveIp -and $chatResolveIp -ne "__none__") {
    $bridgeArgs += @("--chat-resolve-ip", $chatResolveIp)
  }
  if ($embeddingResolveIp -and $embeddingResolveIp -ne "__none__") {
    $bridgeArgs += @("--embedding-resolve-ip", $embeddingResolveIp)
  }
  Start-Process -WindowStyle Hidden -FilePath $PythonCommand.Source -ArgumentList $bridgeArgs
}

function Sync-ModelBridge {
  param(
    [string]$BridgeScript,
    [object]$PythonCommand,
    [int]$Port
  )

  $BridgeHealthUrl = "http://127.0.0.1:$Port/health"
  $BridgeAdminConfigUrl = "http://127.0.0.1:$Port/admin/config"
  $BridgeAdminReloadUrl = "http://127.0.0.1:$Port/admin/reload"
  if (Test-ModelBridgeSelfTarget -Url $chatBaseUrl -Port $Port) {
    throw "CHAT_BASE_URL must be the real chat provider URL, not the model bridge URL."
  }
  if (Test-ModelBridgeSelfTarget -Url $embeddingBaseUrl -Port $Port) {
    throw "EMBEDDING_BASE_URL must be the real embedding provider URL, not the model bridge URL."
  }
  $desiredPayload = @{
    chat_target_base_url = $chatBaseUrl
    chat_resolve_ip = (Normalize-BridgeResolveIp $chatResolveIp)
    embedding_target_base_url = $embeddingBaseUrl
    embedding_resolve_ip = (Normalize-BridgeResolveIp $embeddingResolveIp)
  }

  if (-not (Test-Url $BridgeHealthUrl)) {
    Start-ModelBridge -BridgeScript $BridgeScript -PythonCommand $PythonCommand -Port $Port
    Wait-Url -Url $BridgeHealthUrl -Name "Model bridge" -TimeoutSeconds 20
    return
  }

  $currentConfig = $null
  try {
    $currentConfig = Invoke-BridgeAdminJson -Uri $BridgeAdminConfigUrl
  } catch {
    Write-Host "Existing model bridge does not support admin reload or token mismatch; restarting bridge." -ForegroundColor Yellow
    Stop-ModelBridgeOnPort -Port $Port
    Start-ModelBridge -BridgeScript $BridgeScript -PythonCommand $PythonCommand -Port $Port
    Wait-Url -Url $BridgeHealthUrl -Name "Model bridge" -TimeoutSeconds 20
    return
  }

  $needsReload =
    (Normalize-BridgeBaseUrl $currentConfig.chat_target_base_url) -ne (Normalize-BridgeBaseUrl $chatBaseUrl) -or
    (Normalize-BridgeBaseUrl $currentConfig.embedding_target_base_url) -ne (Normalize-BridgeBaseUrl $embeddingBaseUrl) -or
    (Normalize-BridgeResolveIp $currentConfig.chat_resolve_ip) -ne (Normalize-BridgeResolveIp $chatResolveIp) -or
    (Normalize-BridgeResolveIp $currentConfig.embedding_resolve_ip) -ne (Normalize-BridgeResolveIp $embeddingResolveIp)

  if ($needsReload) {
    Write-Host "Reloading model bridge target configuration" -ForegroundColor Yellow
    $null = Invoke-BridgeAdminJson -Uri $BridgeAdminReloadUrl -Method "POST" -Body $desiredPayload
  }
}

if ($modelBridgeEnabled) {
  $BridgeScript = Join-Path $Root "infra\model-bridge\model_bridge.py"
  if (-not (Test-Path $BridgeScript)) {
    throw "Model bridge script not found: $BridgeScript"
  }
  $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
  if (-not $pythonCommand) {
    throw "MODEL_BRIDGE_ENABLED=true requires Python on the Windows host PATH."
  }
  if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw "MODEL_BRIDGE_ENABLED=true requires Windows curl.exe on PATH."
  }

  Sync-ModelBridge -BridgeScript $BridgeScript -PythonCommand $pythonCommand -Port $modelBridgePort

  $env:API_CHAT_BASE_URL = "http://host.docker.internal:$modelBridgePort"
  $env:API_CHAT_RESOLVE_IP = "__none__"
} else {
  $env:API_CHAT_BASE_URL = $chatBaseUrl
  $env:API_CHAT_RESOLVE_IP = $chatResolveIp
}

Write-Host "Course Knowledge Base Docker launcher" -ForegroundColor Cyan
Write-Host "Root: $Root"
Write-Host "Model bridge enabled: $modelBridgeEnabled"
if ($modelBridgeEnabled) {
  Write-Host "Model bridge: http://127.0.0.1:$modelBridgePort"
}
Write-Host "API: http://127.0.0.1:$BackendPort/api"
Write-Host "Web: http://127.0.0.1:$FrontendPort"

# Stop heavy leftover ablation/experiment containers to free up system memory and prevent OOM
try {
  $runningAblations = & docker ps --filter "status=running" --format "{{.Names}}" 2>$null
  if ($LASTEXITCODE -eq 0 -and $runningAblations) {
    foreach ($containerName in $runningAblations) {
      if ($containerName -match "ablation" -or $containerName -match "experiment") {
        Write-Host "Stopping heavy background service to free up memory: $containerName" -ForegroundColor Yellow
        & docker stop $containerName > $null
      }
    }
  }
} catch {
  # Ignore failures to avoid blocking startup if docker is not running or other issues
}

Invoke-Compose -Arguments @(
  "compose",
  "-f", $InfraComposeFile,
  "down", "--remove-orphans"
)

Invoke-Compose -Arguments @(
  "compose",
  "-f", $InfraComposeFile,
  "up", "-d",
  "postgres", "redis", "qdrant", "api", "worker", "web"
)
$stopCommand = "docker compose -f infra/docker-compose.yml down"


Wait-Url -Url $BackendUrl -Name "Backend"
Wait-Url -Url "http://127.0.0.1:$FrontendPort" -Name "Frontend"

if (-not $NoBrowser) {
  Write-Host "Opening $FrontendUrl"
  Start-Process $FrontendUrl
}

Write-Host "Done. Stop services with: $stopCommand" -ForegroundColor Green
