@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ========================================
echo    PeakGuard Report Generator
echo ========================================
echo.

:: Pruefen ob Python installiert ist
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [FEHLER] Python wurde nicht gefunden!
    echo Bitte installieren Sie Python von https://www.python.org/downloads/
    echo Stellen Sie sicher, dass "Add Python to PATH" aktiviert ist.
    pause
    exit /b 1
)

:: Python-Version anzeigen
echo [OK] Python gefunden:
python --version
echo.

:: Pruefen ob venv existiert, sonst erstellen
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Erstelle virtuelle Umgebung...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [FEHLER] Konnte virtuelle Umgebung nicht erstellen!
        pause
        exit /b 1
    )
    echo [OK] Virtuelle Umgebung erstellt.
    echo.

    :: Aktivieren und Abhaengigkeiten installieren
    call ".venv\Scripts\activate.bat"

    echo [INFO] Installiere Abhaengigkeiten...
    python -m pip install --upgrade pip >nul 2>nul
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [FEHLER] Konnte Abhaengigkeiten nicht installieren!
        pause
        exit /b 1
    )
    echo [OK] Abhaengigkeiten installiert.
    echo.
) else (
    :: Aktiviere vorhandenes venv
    call ".venv\Scripts\activate.bat"
)

:: Streamlit starten
echo [INFO] Starte PeakGuard...
echo.
echo ----------------------------------------
echo Die App oeffnet sich im Browser unter:
echo http://localhost:8501
echo ----------------------------------------
echo.
echo Zum Beenden: Strg+C druecken oder Fenster schliessen.
echo.

python -m streamlit run app.py --server.headless true

endlocal
