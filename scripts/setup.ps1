# One-shot setup for Windows PowerShell.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment ..."
    python -m venv .venv
}

$python = ".\.venv\Scripts\python.exe"

Write-Host "Upgrading pip ..."
& $python -m pip install --upgrade pip

Write-Host "Installing CPU-only PyTorch (this downloads ~200 MB) ..."
& $python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

Write-Host "Installing Nestor and its dependencies ..."
& $python -m pip install -e ".[dev]"

Write-Host ""
Write-Host "Done. Activate the environment with: .\.venv\Scripts\Activate.ps1"
Write-Host "Then run: nestor download-sample; nestor transcribe samples\example.wav"
