@echo off
echo ============================================
echo   Bet261 Prediction Engine - Start All
echo ============================================
echo.

:: Start Backend in new window
echo Starting Backend API on port 8000...
start "Backend API" cmd /c "cd /d %~dp0backend && py -m uvicorn app.main:app --reload --port 8000"

:: Wait for backend to start
timeout /t 3 /nobreak > nul

:: Start Frontend in new window
echo Starting Frontend Dashboard on port 3000...
start "Frontend Dashboard" cmd /c "cd /d %~dp0frontend && npm run dev"

echo.
echo ============================================
echo   Services Started!
echo ============================================
echo.
echo   Backend API:    http://localhost:8000
echo   API Docs:       http://localhost:8000/docs
echo   Frontend:       http://localhost:3000
echo.
echo   To start the autonomous engine, run:
echo   start_orchestrator.bat
echo.
echo ============================================
pause
