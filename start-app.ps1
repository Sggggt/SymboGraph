param(
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 3000,
  [string]$OpenPath = "/graph",
  # Frozen because Docker prefixes the persistent symbograph-data volume with this identity.
  [string]$ComposeProjectName = "knowledgegraph-dev-20260820",
  [string]$ApiImage = "course-kg-api:local",
  [switch]$SkipBuild,
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile = Join-Path $Root ".env"
$EnvExampleFile = Join-Path $Root ".env.example"
$SettingsFile = Join-Path $Root "settings.json"
$SettingsExampleFile = Join-Path $Root "settings.example.json"
$InfraComposeFile = Join-Path $Root "infra\docker-compose.yml"

if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) {
  if (-not (Test-Path -LiteralPath $EnvExampleFile -PathType Leaf)) {
    throw "Missing .env and .env.example."
  }
  Copy-Item -LiteralPath $EnvExampleFile -Destination $EnvFile
}
if (-not (Test-Path -LiteralPath $SettingsFile -PathType Leaf)) {
  if (-not (Test-Path -LiteralPath $SettingsExampleFile -PathType Leaf)) {
    throw "Missing settings.json and settings.example.json."
  }
  Copy-Item -LiteralPath $SettingsExampleFile -Destination $SettingsFile
}

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
    $effectiveArguments = @(
      "compose",
      "--env-file", $EnvFile,
      "--project-name", $ComposeProjectName
    ) + $remainingArguments
  } elseif ($Arguments.Length -gt 0 -and $Arguments[0] -eq "compose") {
    $remainingArguments = @()
    if ($Arguments.Length -gt 1) {
      $remainingArguments = $Arguments[1..($Arguments.Length - 1)]
    }
    $effectiveArguments = @(
      "compose",
      "--project-name", $ComposeProjectName
    ) + $remainingArguments
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

if ($ComposeProjectName -cnotmatch '^[a-z0-9][a-z0-9_-]*$') {
  throw "Unsupported ComposeProjectName='$ComposeProjectName'. Use lowercase letters, digits, underscores or hyphens."
}
if ([string]::IsNullOrWhiteSpace($ApiImage)) {
  throw "ApiImage must be a non-empty Docker image reference."
}
if (-not $SkipBuild) {
  foreach ($buildImage in @(
    @{ Name = "ApiImage"; Value = $ApiImage }
  )) {
    $buildValue = [string]$buildImage.Value
    if ($buildValue.Contains("@") -or $buildValue.StartsWith("sha256:", [StringComparison]::OrdinalIgnoreCase)) {
      throw "$($buildImage.Name)='$buildValue' is digest-qualified and cannot be used as a Docker build output tag. Pass a mutable tag or use -SkipBuild to run an existing digest."
    }
  }
}
$modelBridgeEnabled = Get-DotEnvBool -Key "MODEL_BRIDGE_ENABLED" -DefaultValue $false
$modelBridgePortRaw = Get-DotEnvValue -Key "MODEL_BRIDGE_PORT" -DefaultValue "8765"
$chatApiProtocol = (Get-DotEnvValue -Key "CHAT_API_PROTOCOL" -DefaultValue "openai").ToLowerInvariant()
$chatBaseUrl = Get-DotEnvValue -Key "CHAT_BASE_URL" -DefaultValue ""
$chatResolveIp = Get-DotEnvValue -Key "CHAT_RESOLVE_IP" -DefaultValue ""
$graphApiProtocol = (Get-DotEnvValue -Key "GRAPH_API_PROTOCOL" -DefaultValue "openai").ToLowerInvariant()
$graphBaseUrl = Get-DotEnvValue -Key "GRAPH_BASE_URL" -DefaultValue ""
$graphResolveIp = Get-DotEnvValue -Key "GRAPH_RESOLVE_IP" -DefaultValue ""
$embeddingApiProtocol = Get-DotEnvValue -Key "EMBEDDING_API_PROTOCOL" -DefaultValue "openai"
$embeddingBaseUrl = Get-DotEnvValue -Key "EMBEDDING_BASE_URL" -DefaultValue ""
$embeddingResolveIp = Get-DotEnvValue -Key "EMBEDDING_RESOLVE_IP" -DefaultValue ""
$modelFallbackEnabled = Get-DotEnvBool -Key "ENABLE_MODEL_FALLBACK" -DefaultValue $false
$databaseFallbackEnabled = Get-DotEnvBool -Key "ENABLE_DATABASE_FALLBACK" -DefaultValue $false
$modelBridgeAdminToken = Get-DotEnvValue -Key "MODEL_BRIDGE_ADMIN_TOKEN" -DefaultValue ""
$modelBridgeAdminTokenValidationValue = $modelBridgeAdminToken
if ($modelBridgeAdminTokenValidationValue.Length -ge 2) {
  $firstTokenCharacter = $modelBridgeAdminTokenValidationValue[0]
  $lastTokenCharacter = $modelBridgeAdminTokenValidationValue[
    $modelBridgeAdminTokenValidationValue.Length - 1
  ]
  if (
    ($firstTokenCharacter -eq '"' -or $firstTokenCharacter -eq "'") -and
    $lastTokenCharacter -eq $firstTokenCharacter
  ) {
    $modelBridgeAdminTokenValidationValue = $modelBridgeAdminTokenValidationValue.Substring(
      1,
      $modelBridgeAdminTokenValidationValue.Length - 2
    )
  }
}
try {
  $modelBridgePort = [int]$modelBridgePortRaw
} catch {
  throw "Unsupported MODEL_BRIDGE_PORT='$modelBridgePortRaw'. Use an integer port."
}
if ($modelBridgeEnabled -and ($modelBridgePort -lt 1 -or $modelBridgePort -gt 65535)) {
  throw "Unsupported MODEL_BRIDGE_PORT='$modelBridgePort'. Use a port between 1 and 65535."
}
if ($chatApiProtocol -notin @("openai", "anthropic")) {
  throw "Unsupported CHAT_API_PROTOCOL='$chatApiProtocol'. Use openai or anthropic."
}
if ($graphApiProtocol -notin @("openai", "anthropic")) {
  throw "Unsupported GRAPH_API_PROTOCOL='$graphApiProtocol'. Use openai or anthropic."
}
if ($embeddingApiProtocol -cne "openai") {
  throw "Unsupported EMBEDDING_API_PROTOCOL='$embeddingApiProtocol'. Use openai. Standard Anthropic Messages has no embedding contract."
}
if ($modelFallbackEnabled -or $databaseFallbackEnabled) {
  throw "The source-mounted local stack requires ENABLE_MODEL_FALLBACK=false and ENABLE_DATABASE_FALLBACK=false."
}
$modelBridgeAdminTokenHasControlCharacter = $false
if ($null -ne $modelBridgeAdminToken) {
  foreach ($character in $modelBridgeAdminToken.ToCharArray()) {
    if ([char]::IsControl($character)) {
      $modelBridgeAdminTokenHasControlCharacter = $true
      break
    }
  }
}
if ($modelBridgeEnabled -and (
  [string]::IsNullOrEmpty($modelBridgeAdminToken) -or
  $modelBridgeAdminToken -ne $modelBridgeAdminToken.Trim() -or
  [string]::IsNullOrEmpty($modelBridgeAdminTokenValidationValue) -or
  $modelBridgeAdminTokenValidationValue -ne $modelBridgeAdminTokenValidationValue.Trim() -or
  $modelBridgeAdminTokenHasControlCharacter
)) {
  throw "MODEL_BRIDGE_ENABLED=true requires a non-empty MODEL_BRIDGE_ADMIN_TOKEN without whitespace padding or control characters."
}
if ($modelBridgeEnabled -and @(
  "change-me",
  "changeme",
  "default",
  "local-model-bridge-admin",
  "model-bridge-admin-token"
) -contains $modelBridgeAdminTokenValidationValue.ToLowerInvariant()) {
  throw "MODEL_BRIDGE_ENABLED=true requires a non-default MODEL_BRIDGE_ADMIN_TOKEN."
}

$BackendUrl = "http://127.0.0.1:$BackendPort/api/ready"
$FrontendUrl = "http://127.0.0.1:$FrontendPort$OpenPath"
$env:COMPOSE_PROJECT_NAME = $ComposeProjectName
$env:API_IMAGE = $ApiImage
$env:API_HOST_PORT = [string]$BackendPort
$env:NEXT_PUBLIC_API_BASE_URL = "http://127.0.0.1:$BackendPort/api"
$env:CHAT_BASE_URL = $chatBaseUrl
$env:CHAT_API_PROTOCOL = $chatApiProtocol
$env:CHAT_RESOLVE_IP = $chatResolveIp
$env:GRAPH_BASE_URL = $graphBaseUrl
$env:GRAPH_API_PROTOCOL = $graphApiProtocol
$env:GRAPH_RESOLVE_IP = $graphResolveIp
$env:EMBEDDING_BASE_URL = $embeddingBaseUrl
$env:EMBEDDING_API_PROTOCOL = $embeddingApiProtocol
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
    "--chat-api-protocol", $chatApiProtocol,
    "--embedding-api-protocol", $embeddingApiProtocol,
    "--chat-target-base-url", $chatBaseUrl,
    "--embedding-target-base-url", $embeddingBaseUrl
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
    chat_api_protocol = $chatApiProtocol
    chat_target_base_url = $chatBaseUrl
    chat_resolve_ip = (Normalize-BridgeResolveIp $chatResolveIp)
    embedding_api_protocol = $embeddingApiProtocol
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
    ([string]$currentConfig.chat_api_protocol).ToLowerInvariant() -ne $chatApiProtocol -or
    ([string]$currentConfig.embedding_api_protocol) -cne $embeddingApiProtocol -or
    (Normalize-BridgeBaseUrl $currentConfig.chat_target_base_url) -ne (Normalize-BridgeBaseUrl $chatBaseUrl) -or
    (Normalize-BridgeBaseUrl $currentConfig.embedding_target_base_url) -ne (Normalize-BridgeBaseUrl $embeddingBaseUrl) -or
    (Normalize-BridgeResolveIp $currentConfig.chat_resolve_ip) -ne (Normalize-BridgeResolveIp $chatResolveIp) -or
    (Normalize-BridgeResolveIp $currentConfig.embedding_resolve_ip) -ne (Normalize-BridgeResolveIp $embeddingResolveIp)

  if ($needsReload) {
    Write-Host "Reloading model bridge target configuration" -ForegroundColor Yellow
    $null = Invoke-BridgeAdminJson -Uri $BridgeAdminReloadUrl -Method "POST" -Body $desiredPayload
  }
}

function Stop-NativeWebProcess {
  param([int]$Port)

  $pidFile = Join-Path $Root "output\web-dev.pid.json"
  if (-not (Test-Path -LiteralPath $pidFile -PathType Leaf)) {
    return
  }
  try {
    $record = Get-Content -Raw -Encoding UTF8 -LiteralPath $pidFile | ConvertFrom-Json
    $processId = [int]$record.pid
    if ([string]$record.root -ne $Root -or [int]$record.port -ne $Port) {
      throw "Native Web pid record does not match this workspace and port."
    }
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
    if ($null -eq $owner) {
      Remove-Item -LiteralPath $pidFile -Force
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
    foreach ($childId in @($descendants | Select-Object -Unique | Sort-Object -Descending)) {
      Stop-Process -Id $childId -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $pidFile -Force
  } catch {
    throw "Could not stop the recorded native Web process safely: $($_.Exception.Message)"
  }
}

function Start-NativeWebProcess {
  param([int]$Port)

  $nodeVersion = (& node --version 2>$null)
  if ($LASTEXITCODE -ne 0 -or $nodeVersion -notmatch '^v(\d+)\.(\d+)\.') {
    throw "Node.js 20.9 or newer is required for Next.js 16.2.4."
  }
  if ([int]$Matches[1] -lt 20 -or ([int]$Matches[1] -eq 20 -and [int]$Matches[2] -lt 9)) {
    throw "Node.js 20.9 or newer is required; found $nodeVersion."
  }
  if (-not (Test-Path -LiteralPath (Join-Path $Root "node_modules\next\package.json") -PathType Leaf)) {
    Write-Host "Installing native Web dependencies with npm ci..." -ForegroundColor Cyan
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) {
      throw "npm ci failed."
    }
  }
  $outputRoot = Join-Path $Root "output"
  New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
  Stop-NativeWebProcess -Port $Port
  $stdout = Join-Path $outputRoot "web-dev.stdout.log"
  $stderr = Join-Path $outputRoot "web-dev.stderr.log"
  $process = Start-Process -FilePath "npm.cmd" `
    -ArgumentList @("run", "dev", "--workspace", "web", "--", "--hostname", "127.0.0.1", "--port", [string]$Port) `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru
  @{
    protocol_version = "symbograph_native_web_pid_v1"
    pid = $process.Id
    root = $Root
    port = $Port
    started_at = (Get-Date).ToUniversalTime().ToString("o")
  } | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $outputRoot "web-dev.pid.json")
}

Write-Host "SymboGraph backend Docker and native Web launcher" -ForegroundColor Cyan
Write-Host "Root: $Root"
Write-Host "Compose project: $ComposeProjectName"
Write-Host "API image: $ApiImage"
Write-Host "Web runtime: native Node.js / workspace node_modules"
Write-Host "Environment file: $EnvFile"
Write-Host "Runtime settings file: $SettingsFile"
Write-Host "Model bridge enabled: $modelBridgeEnabled"
if ($modelBridgeEnabled) {
  Write-Host "Model bridge: http://127.0.0.1:$modelBridgePort"
}
Write-Host "API: http://127.0.0.1:$BackendPort/api"
Write-Host "Web: http://127.0.0.1:$FrontendPort"

Invoke-Compose -Arguments @(
  "compose",
  "-f", $InfraComposeFile,
  "config", "--quiet"
)

if (-not $SkipBuild) {
  $RebuildImagesScript = Join-Path $Root "rebuild-images.ps1"
  if (-not (Test-Path -LiteralPath $RebuildImagesScript -PathType Leaf)) {
    throw "Image rebuild script not found: $RebuildImagesScript"
  }
  Write-Host "Building the latest source-mounted local images..." -ForegroundColor Cyan
  & $RebuildImagesScript `
    -ApiBuildTag $ApiImage `
    -ComposeProjectName $ComposeProjectName
} else {
  foreach ($requiredImage in @($ApiImage)) {
    & docker image inspect $requiredImage *> $null
    if ($LASTEXITCODE -ne 0) {
      throw "Required image '$requiredImage' is missing. Re-run without -SkipBuild."
    }
  }
}

Invoke-Compose -Arguments @(
  "compose",
  "-f", $InfraComposeFile,
  "--profile", "model-bridge",
  "down", "--remove-orphans"
)

Invoke-Compose -Arguments @(
  "compose",
  "-f", $InfraComposeFile,
  "up", "-d",
  "postgres", "redis", "qdrant"
)
Wait-ContainerHealthy -ContainerName "course-kg-postgres" -Name "PostgreSQL"
Wait-ContainerHealthy -ContainerName "course-kg-redis" -Name "Redis"
Wait-Url -Url "http://127.0.0.1:6333/readyz" -Name "Qdrant"

try {
  Invoke-Compose -Arguments @(
    "compose",
    "-f", $InfraComposeFile,
    "run", "--rm", "--no-deps",
    "api", "python", "-m", "app.core.migration_safety", "preflight", "--target-revision", "head"
  )
} catch {
  throw "Database migration preflight blocked startup. Review the exact targets above. Run 'python scripts/manage_migrations.py --compose-run preflight head', then use 'upgrade head --allow-destructive' only after review."
}

try {
  Invoke-Compose -Arguments @(
    "compose",
    "-f", $InfraComposeFile,
    "run", "--rm", "--no-deps",
    "api", "python", "-c", "from app.db import ensure_schema; ensure_schema()"
  )
} catch {
  throw "Database schema initialization failed before the runtime identity gate."
}

if ($modelBridgeEnabled) {
  $BridgeScript = Join-Path $Root "infra\model-bridge\model_bridge.py"
  if (-not (Test-Path $BridgeScript)) {
    throw "Model bridge script not found: $BridgeScript"
  }
  $env:MODEL_BRIDGE_CLIENT_HOST = "model-bridge"
  Invoke-Compose -Arguments @(
    "compose",
    "-f", $InfraComposeFile,
    "--profile", "model-bridge",
    "up", "-d", "--force-recreate",
    "model-bridge"
  )
} else {
  $env:MODEL_BRIDGE_CLIENT_HOST = "model-bridge"
}

Invoke-Compose -Arguments @(
  "compose",
  "-f", $InfraComposeFile,
  "up", "-d", "--force-recreate",
  "api", "worker", "beat"
)
$stopCommand = ".\stop-app.ps1 -ComposeProjectName $ComposeProjectName -FrontendPort $FrontendPort"


Wait-Url -Url $BackendUrl -Name "Backend"
Start-NativeWebProcess -Port $FrontendPort
Wait-Url -Url "http://127.0.0.1:$FrontendPort" -Name "Frontend"
Wait-ContainerHealthy -ContainerName "course-kg-worker" -Name "Worker"
Wait-ContainerHealthy -ContainerName "course-kg-beat" -Name "Beat"
if ($modelBridgeEnabled) {
  Wait-ContainerHealthy -ContainerName "course-kg-model-bridge" -Name "Model bridge"
}

if (-not $NoBrowser) {
  Write-Host "Opening $FrontendUrl"
  Start-Process $FrontendUrl
}

Write-Host "Done. Stop services with: $stopCommand" -ForegroundColor Green
