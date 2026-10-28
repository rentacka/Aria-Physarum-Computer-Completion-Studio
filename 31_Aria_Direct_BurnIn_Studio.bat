@echo off
chcp 63001 > nul
title Aria Studio 35 - RSI-5/7 Cumulative Direct Burn-In Studio

cd /d "%~dp0"

echo ===============================================================================
echo   Studio 35: Aria Cumulative Algebraic Direct Burn-In Studio (RSI-5/7)
echo ===============================================================================
echo [INFO] In-place direct synpatic engraving into 27M DeltaLoop weights.
echo [INFO] Disk footprint increase: 0 bytes. Execution time: ~3 seconds.
echo [INFO] RSI-7 EWC cumulative covariance protection + RSI-5 auto rollback.
echo ===============================================================================

set PYTHON_EXE=H:\DiffusionQwen_env\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

echo [Step 1] Running Cumulative Algebraic Burn-In...
"%PYTHON_EXE%" aria_cumulative_burnin.py %*

echo.
echo ===============================================================================
echo [DONE] Direct Burn-In Complete! You can now chat in 29_Chat_Aria_Trinity_DeltaLoop.bat
echo ===============================================================================
pause
