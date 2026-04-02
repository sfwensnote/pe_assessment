param(
    [string]$ApiBase = "http://127.0.0.1:8001",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 5173,
    [switch]$Preview
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$webRoot = Join-Path $projectRoot "web"

if (-not (Test-Path $webRoot)) {
    throw "Missing frontend directory: $webRoot"
}

$env:VITE_API_BASE = $ApiBase
Set-Location $webRoot

if ($Preview) {
    & npm.cmd run build
    & npm.cmd run preview -- --host $BindHost --port $Port
    exit $LASTEXITCODE
}

& npm.cmd run dev -- --host $BindHost --port $Port
exit $LASTEXITCODE
