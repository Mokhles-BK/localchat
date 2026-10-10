# Start backend + frontend together. Ctrl+C stops everything, including llama-server.
$ErrorActionPreference = 'Stop'
$Root     = Split-Path -Parent $PSScriptRoot
$VenvPy   = Join-Path $Root 'backend\.venv\Scripts\python.exe'
$Frontend = Join-Path $Root 'frontend'
if (-not (Test-Path $VenvPy)) { throw 'Run .\scripts\setup.ps1 first.' }

foreach ($port in 8000, 8080, 5173) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $port is already in use. Stop whatever is using it and retry."
    }
}

$pm = if (Get-Command pnpm -ErrorAction SilentlyContinue) { 'pnpm' } else { 'npm run' }

$backend = Start-Process -FilePath $VenvPy -WorkingDirectory $Root -PassThru -NoNewWindow `
    -ArgumentList '-m','uvicorn','main:app','--app-dir','backend','--host','127.0.0.1','--port','8000'
$frontend = Start-Process -FilePath 'cmd.exe' -WorkingDirectory $Frontend -PassThru -NoNewWindow `
    -ArgumentList '/c', "$pm dev"

Write-Host "Backend:  http://127.0.0.1:8000/api/health" -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173  (model loads in ~10-60 s)" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop everything." -ForegroundColor Yellow

try {
    while ($true) {
        Start-Sleep -Seconds 1
        if ($backend.HasExited -or $frontend.HasExited) { Write-Host 'A process exited, shutting down.'; break }
    }
} finally {
    foreach ($p in $backend, $frontend) {
        if ($p -and -not $p.HasExited) { & taskkill /PID $p.Id /T /F | Out-Null }
    }
    Get-Process llama-server -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -like "$Root\data\bin\*" } | Stop-Process -Force
    Write-Host 'Stopped.'
}
