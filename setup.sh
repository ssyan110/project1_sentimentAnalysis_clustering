#!/bin/bash
# SETUP SCRIPT FOR ITviec Reviews Project

echo "🔹 Creating Python virtual environment..."
python3 -m venv .venv

echo "🔹 Activating virtual environment..."
source .venv/bin/activate

echo "🔹 Upgrading pip..."
pip install --upgrade pip

echo "🔹 Installing required packages..."
pip install -r requirements.txt

echo "✅ Setup complete! To activate your environment again, run: source .venv/bin/activate"

mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
