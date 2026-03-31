@echo off
echo ============================================
echo   Bet261 Prediction Engine - Start Complete
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

:: Wait for frontend to start
timeout /t 5 /nobreak > nul

:: Start Orchestrator in new window
echo Starting Prediction Orchestrator...
start "Orchestrator" cmd /c "cd /d %~dp0backend && py orchestrator.py"

echo.
echo ============================================
echo   All Services Started!
echo ============================================
echo.
echo   Backend API:    http://localhost:8000
echo   API Docs:       http://localhost:8000/docs
echo   Frontend:       http://localhost:3000
echo.
echo   3 Windows opened:
echo   - Backend API
echo   - Frontend Dashboard  
echo   - Orchestrator (with scraper)
echo.
echo ============================================
pause
