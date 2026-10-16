@echo off
REM One-click launcher for the web dashboard (Windows). Double-click this file.
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv || (echo Python not found. Install Python 3.9+ from python.org and tick "Add to PATH" & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt || (pause & exit /b 1)
echo Starting the dashboard... your browser will open automatically.
python app.py
pause
