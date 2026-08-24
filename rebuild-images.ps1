param(
  [string]$ApiBuildTag = "course-kg-api:local",
  [string]$WebBuildTag = "course-kg-web:local",
  # Must match start-app.ps1 so image rebuilds address the existing local stack.
  [string]$ComposeProjectName = "knowledgegraph-dev-20260820",
  [switch]$NoCache,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$InfraComposeFile = Join-Path $Root "infra\docker-compose.yml"
$EnvFile = Join-Path $Root ".env"

function Assert-BuildableImageTag {
  param(
    [string]$Name,
    [string]$Value
  )

  if ([string]::IsNullOrWhiteSpace($Value)) {
    throw "$Name must be a non-empty Docker image tag."
  }
  if ($Value.Contains("@") -or $Value.StartsWith("sha256:", [StringComparison]::OrdinalIgnoreCase)) {
    throw "$Name='$Value' is digest-qualified. Docker build outputs require a mutable image tag such as 'course-kg-api:local'; use a digest only as a runtime image with start-app.ps1 -SkipBuild."
  }
}

function Invoke-Compose {
  param(
    [string[]]$Arguments
  )

  $remainingArguments = @()
  if ($Arguments.Length -gt 1) {
    $remainingArguments = $Arguments[1..($Arguments.Length - 1)]
  }
  $effectiveArguments = @("compose")
  if (Test-Path $EnvFile) {
    $effectiveArguments += @("--env-file", $EnvFile)
  }
  $effectiveArguments += @("--project-name", $ComposeProjectName)
  $effectiveArguments += $remainingArguments

  if ($DryRun) {
    Write-Output ("PLAN docker " + ($effectiveArguments -join " "))
    return
  }

  & docker @effectiveArguments
  if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose build failed."
  }
}

if (-not (Test-Path $InfraComposeFile)) {
  throw "Docker Compose file not found: $InfraComposeFile"
}
if ($ComposeProjectName -cnotmatch '^[a-z0-9][a-z0-9_-]*$') {
  throw "Unsupported ComposeProjectName='$ComposeProjectName'. Use lowercase letters, digits, underscores or hyphens."
}
Assert-BuildableImageTag -Name "ApiBuildTag" -Value $ApiBuildTag
Assert-BuildableImageTag -Name "WebBuildTag" -Value $WebBuildTag

# Shell environment has higher Compose interpolation precedence than --env-file.
# This intentionally prevents a locked runtime API_IMAGE=name@sha256:... in .env
# from becoming the output tag of a local build.
$env:API_IMAGE = $ApiBuildTag
$env:WEB_IMAGE = $WebBuildTag

Write-Host "API build tag: $ApiBuildTag"
Write-Host "Web build tag: $WebBuildTag"

Invoke-Compose -Arguments @(
  "compose",
  "-f", $InfraComposeFile,
  "config", "--quiet"
)

$buildArgs = @("compose", "-f", $InfraComposeFile, "build")
if ($NoCache) {
  $buildArgs += "--no-cache"
}
# worker uses the exact API image/tag and does not need a duplicate build.
$buildArgs += @("api", "web")

Write-Host "Rebuilding application images (api, web; worker reuses api)..." -ForegroundColor Cyan
Invoke-Compose -Arguments $buildArgs

if (-not $DryRun) {
  foreach ($requiredImage in @($ApiBuildTag, $WebBuildTag)) {
    & docker image inspect $requiredImage *> $null
    if ($LASTEXITCODE -ne 0) {
      throw "Build completed without the expected image tag '$requiredImage'."
    }
  }
}

Write-Host "Done. Images rebuilt. Start services with start-app.bat" -ForegroundColor Green
