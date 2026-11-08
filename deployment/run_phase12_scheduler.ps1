param(
    [string]$ProjectRoot = "F:\Projects\Steel Frame Optimization"
)

$ErrorActionPreference = "Stop"
$env:HTTP_PROXY = "socks5h://127.0.0.1:10808"
$env:HTTPS_PROXY = "socks5h://127.0.0.1:10808"
$env:ALL_PROXY = "socks5h://127.0.0.1:10808"
$env:UV_PYTHON_DOWNLOADS = "never"

Set-Location -LiteralPath $ProjectRoot
$stateRoot = Join-Path $ProjectRoot "outputs\runs\phase12_scheduler"
New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
$supervisorLog = Join-Path $stateRoot "supervisor.log"

while (-not (Test-Path -LiteralPath (Join-Path $stateRoot "STOP"))) {
    $started = Get-Date -Format o
    Add-Content -LiteralPath $supervisorLog -Value "$started scheduler starting"
    & uv run --frozen --offline --no-sync python `
        deployment/phase12_multiplatform_scheduler.py `
        --project-root $ProjectRoot `
        --state-root $stateRoot *>> $supervisorLog
    $exitCode = $LASTEXITCODE
    Add-Content -LiteralPath $supervisorLog -Value "$(Get-Date -Format o) scheduler exited $exitCode"
    if ($exitCode -eq 0) {
        break
    }
    Start-Sleep -Seconds 30
}
