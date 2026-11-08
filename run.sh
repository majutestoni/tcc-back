#!/usr/bin/env bash
# Setup + execução do app (macOS / Linux). Uso: bash run.sh
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Criando ambiente virtual..."
  python3 -m venv .venv
fi

echo "Instalando dependências..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Arquivo .env criado — preencha GEMINI_API_KEY para usar a recomendação via Gemini."
fi

echo "Subindo o Streamlit..."
.venv/bin/python -m streamlit run app/Home.py
