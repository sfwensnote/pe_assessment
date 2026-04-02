param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8001,
    [switch]$Reload,
    [string]$AdminToken = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Missing Python runtime: $pythonExe"
}

$env:YOLO_CONFIG_DIR = Join-Path $projectRoot ".ultralytics"
$env:PYTHONIOENCODING = "utf-8"

if ($AdminToken) {
    $env:ADMIN_TOKEN = $AdminToken
}

$uvicornArgs = @(
    "-m", "uvicorn",
    "app.main:app",
    "--host", $BindHost,
    "--port", $Port.ToString()
)

if ($Reload) {
    $uvicornArgs += "--reload"
}

Set-Location $projectRoot
& $pythonExe @uvicornArgs
