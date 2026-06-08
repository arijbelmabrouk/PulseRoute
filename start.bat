@echo off
title PulseRoute
color 0F

echo ================================================
echo  PulseRoute - Teleconsultation Dashboard
echo ================================================
echo.

REM ── Check if rppg_env exists ──────────────────────
if not exist "%~dp0rppg_env\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found.
    echo Please run setup first.
    pause
    exit /b 1
)

REM ── Check if npm is available ─────────────────────
where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: npm not found. Please install Node.js.
    pause
    exit /b 1
)

REM ── Start FastAPI server in new window ────────────
echo [1/3] Starting dashboard server...
start "PulseRoute - Server" cmd /k "cd /d %~dp0 && call rppg_env\Scripts\activate.bat && uvicorn step12_display.server:app --port 8000"

REM ── Wait for server to be ready ───────────────────
timeout /t 3 /nobreak >nul

REM ── Start React frontend in new window ───────────
echo [2/3] Starting web interface...
start "PulseRoute - Frontend" cmd /k "cd /d %~dp0step12_display\frontend && npm run dev"

REM ── Wait for frontend to be ready ────────────────
timeout /t 4 /nobreak >nul

REM ── Open browser ─────────────────────────────────
echo [3/3] Opening dashboard in browser...
start "" "http://localhost:5173/patient"
timeout /t 1 /nobreak >nul
start "" "http://localhost:5173/doctor"

echo.
echo ================================================
echo  Dashboard is running.
echo.
echo  Patient view : http://localhost:5173/patient
echo  Doctor view  : http://localhost:5173/doctor
echo.
echo  Press any key to start the measurement pipeline.
echo ================================================
echo.
pause >nul

REM ── Run pipeline in this window ───────────────────
call rppg_env\Scripts\activate.bat
python run_web.py

echo.
echo ================================================
echo  Measurement complete. Press any key to exit.
echo ================================================
pause >nul