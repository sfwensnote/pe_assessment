param(
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8001,
    [int]$WebPort = 5173,
    [switch]$BackendReload,
    [switch]$FrontendPreview,
    [string]$AdminToken = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$backendScript = Join-Path $PSScriptRoot "start_backend.ps1"
$frontendScript = Join-Path $PSScriptRoot "start_frontend.ps1"
$apiBase = "http://{0}:{1}" -f $ApiHost, $ApiPort

$backendArgs = @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", $backendScript,
    "-BindHost", $ApiHost,
    "-Port", $ApiPort.ToString()
)

if ($BackendReload) {
    $backendArgs += "-Reload"
}

if ($AdminToken) {
    $backendArgs += @("-AdminToken", $AdminToken)
}

$frontendArgs = @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", $frontendScript,
    "-ApiBase", $apiBase,
    "-BindHost", $ApiHost,
    "-Port", $WebPort.ToString()
)

if ($FrontendPreview) {
    $frontendArgs += "-Preview"
}

Start-Process powershell.exe -ArgumentList $backendArgs -WorkingDirectory $projectRoot
Start-Process powershell.exe -ArgumentList $frontendArgs -WorkingDirectory $projectRoot

Write-Host "Backend launching at $apiBase"
Write-Host "Frontend launching at http://$ApiHost`:$WebPort"
