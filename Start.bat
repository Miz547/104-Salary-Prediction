@echo off
setlocal
cd /d "%~dp0"

title AI Salary Prediction System

echo ============================================
echo   AI Salary Prediction System
echo   Setup and Start
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10 or later.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo [OK] Detected %PYVER%
echo.

if exist ".venv\Scripts\activate.bat" (
    echo [OK] Virtual environment already exists.
) else (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)
echo.

echo [2/3] Installing packages...
call ".venv\Scripts\activate.bat"

python -m pip install --upgrade pip
python -m pip install -r requirements_server.txt

if errorlevel 1 (
    echo [ERROR] Package installation failed.
    echo Please check your internet connection or requirements_server.txt.
    pause
    exit /b 1
)
echo.

echo [3/3] Starting server...
echo Open: http://127.0.0.1:5001
echo.
start "" "http://127.0.0.1:5001"

python salary_server.py

echo.
echo Server stopped.
pause