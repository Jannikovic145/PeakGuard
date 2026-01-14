#!/bin/bash
set -e

# ins Repo-Verzeichnis wechseln (Ordner der .command)
cd "$(dirname "$0")"

# venv aktivieren
source ".venv/bin/activate"

# Streamlit starten
python -m streamlit run app.py
