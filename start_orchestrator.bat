@echo off
cd /d %~dp0backend
echo Starting Bet261 Prediction Engine Orchestrator...
echo This will run the autonomous betting system.
py -m app.core.orchestrator
pause
