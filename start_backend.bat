@echo off
cd /d %~dp0backend
echo Starting Bet261 Prediction Engine Backend...
py -m uvicorn app.main:app --reload --port 8000
pause
