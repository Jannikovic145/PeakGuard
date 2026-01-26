#!/bin/bash
set -e

# ins Repo-Verzeichnis wechseln (Ordner der .command)
cd "$(dirname "$0")"

# Prüfe ob venv existiert, falls nicht: erstelle es
if [ ! -d ".venv" ]; then
    echo "Virtual Environment wird erstellt..."
    python3 -m venv .venv
    source ".venv/bin/activate"
    echo "Installiere Dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source ".venv/bin/activate"
fi

# Streamlit starten
echo "Starte PeakGuard Report Generator..."
python -m streamlit run app.py
