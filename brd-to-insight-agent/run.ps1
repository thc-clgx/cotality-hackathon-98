param([switch]$Setup, [int]$Port = 8511)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Setup) {
    if (-not (Test-Path ".venv\Scripts\python.exe")) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
    }
    & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Environment missing. Run this script with -Setup once."
}

& .\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port $Port
