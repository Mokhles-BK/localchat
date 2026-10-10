# LocalChat setup (Windows / PowerShell). Safe to re-run: skips what already exists.
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root      = Split-Path -Parent $PSScriptRoot
$BinDir    = Join-Path $Root 'data\bin'
$ModelDir  = Join-Path $Root 'data\models'
$Backend   = Join-Path $Root 'backend'
$Frontend  = Join-Path $Root 'frontend'
$ModelFile = Join-Path $ModelDir 'Llama-3.2-3B-Instruct-Q4_K_M.gguf'
$ModelSize = 2019377696
$ModelUrl  = 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf'

function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }

function Download($url, $out) {
    & curl.exe -L -C - --retry 5 --retry-delay 3 -o $out $url
    if ($LASTEXITCODE -ne 0) { throw "Download failed (curl exit $LASTEXITCODE): $url" }
}

# ---- Python 3.11+ ----
Step 'Checking Python 3.11+'
$Python = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($v in '3.11','3.12','3.13') {
        try {
            $exe = (& py "-$v" -c 'import sys; print(sys.executable)').Trim()
            if ($LASTEXITCODE -eq 0 -and $exe) { $Python = $exe; break }
        } catch { }
    }
}
if (-not $Python -and (Get-Command python -ErrorAction SilentlyContinue)) {
    try {
        $ok = (& python -c 'import sys; print(sys.version_info >= (3, 11))').Trim()
        if ($ok -eq 'True') { $Python = (& python -c 'import sys; print(sys.executable)').Trim() }
    } catch { }
}
if (-not $Python) { throw 'Python 3.11 or newer was not found. Install it and re-run.' }
Write-Host "Using $Python"

# ---- Node 18+ ----
Step 'Checking Node 18+'
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw 'Node.js not found. Install Node 18 or newer.' }
$nodeMajor = [int]((node --version).TrimStart('v').Split('.')[0])
if ($nodeMajor -lt 18) { throw "Node 18+ required, found $(node --version)" }
Write-Host "Node $(node --version)"

# ---- GPU ----
Step 'Checking NVIDIA GPU'
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw 'nvidia-smi not found. Install the NVIDIA driver.' }
& nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed' }

# ---- llama.cpp ----
Step 'llama.cpp (llama-server, CUDA 12.4)'
New-Item -ItemType Directory -Force $BinDir | Out-Null
$ServerExe = Join-Path $BinDir 'llama-server.exe'
$CudartDll = Join-Path $BinDir 'cudart64_12.dll'
if (-not (Test-Path $ServerExe) -or -not (Test-Path $CudartDll)) {
    $headers  = @{ 'User-Agent' = 'localchat-setup' }
    $releases = Invoke-RestMethod -Headers $headers -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15'
    $rel = $releases | Where-Object { $_.assets | Where-Object { $_.name -match '^llama-.*-bin-win-cuda-12\.4-x64\.zip$' } } | Select-Object -First 1
    if (-not $rel) { throw 'No recent llama.cpp release has a win-cuda-12.4-x64 build.' }
    Write-Host "Release: $($rel.tag_name)"
    $wanted = @(
        ($rel.assets | Where-Object { $_.name -match '^llama-.*-bin-win-cuda-12\.4-x64\.zip$' } | Select-Object -First 1),
        ($rel.assets | Where-Object { $_.name -match '^cudart-llama-bin-win-cuda-12\.4-x64\.zip$' } | Select-Object -First 1)
    )
    foreach ($a in $wanted) {
        if (-not $a) { throw 'Expected llama.cpp asset missing from release.' }
        $zip = Join-Path $BinDir $a.name
        if (-not (Test-Path $zip) -or (Get-Item $zip).Length -ne $a.size) {
            Write-Host "Downloading $($a.name)"
            Download $a.browser_download_url $zip
        }
        if ((Get-Item $zip).Length -ne $a.size) { throw "Size mismatch for $($a.name)" }
        Write-Host "Extracting $($a.name)"
        Expand-Archive -Path $zip -DestinationPath $BinDir -Force
        Remove-Item $zip
    }
}
if (-not (Test-Path $ServerExe)) { throw 'llama-server.exe is missing after extraction.' }
& $ServerExe --version
if ($LASTEXITCODE -ne 0) { throw 'llama-server.exe --version failed' }

# ---- Backend ----
Step 'Backend virtualenv + dependencies'
$VenvPy = Join-Path $Backend '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPy)) {
    & $Python -m venv (Join-Path $Backend '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
& $VenvPy -m pip install -r (Join-Path $Backend 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }

# ---- Frontend ----
Step 'Frontend dependencies'
Push-Location $Frontend
try {
    if (Get-Command pnpm -ErrorAction SilentlyContinue) { pnpm install } else { npm install }
    if ($LASTEXITCODE -ne 0) { throw 'frontend install failed' }
} finally { Pop-Location }

# ---- Model ----
Step 'Default model (Llama 3.2 3B Instruct Q4_K_M)'
New-Item -ItemType Directory -Force $ModelDir | Out-Null
if (-not (Test-Path $ModelFile) -or (Get-Item $ModelFile).Length -ne $ModelSize) {
    Download $ModelUrl $ModelFile
}
$size = (Get-Item $ModelFile).Length
if ($size -ne $ModelSize) { throw "Model size is $size, expected $ModelSize. Delete the file and re-run." }
Write-Host "Model OK ($size bytes)"

Write-Host "`nSetup complete. Start the app with: .\scripts\dev.ps1" -ForegroundColor Green
