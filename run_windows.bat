@echo off
REM One-click setup + run for Windows. Double-click or run from inside this folder.
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv || (echo Python not found. Install Python 3.9+ from python.org and tick "Add to PATH" & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || (pause & exit /b 1)
python main.py || (pause & exit /b 1)
python -m unittest discover -s tests -v
echo.
echo Done. Open the images folder and the outputs folder to see the results.
pause
