# Setup + execução do app (Windows PowerShell). Uso: .\run.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    Write-Host "Criando ambiente virtual..."
    python -m venv .venv
}

Write-Host "Instalando dependências..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Arquivo .env criado — preencha GEMINI_API_KEY para usar a recomendação via Gemini."
}

Write-Host "Subindo o Streamlit..."
.\.venv\Scripts\python.exe -m streamlit run app/Home.py
