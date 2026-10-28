@echo off
rem ================================================================================
rem Studio 35: Aria Trinity DeltaLoop Interactive Chat CLI (7-bit ASCII Sentinel)
rem Trinity Biomorphic Intelligence: Gemma-4 ENGRAFT + Physarum + Task LoRA + RLT
rem ================================================================================

chcp 65001 > nul
set "PYTHONIOENCODING=utf-8"

title Aria Trinity DeltaLoop Biomorphic Chat CLI v2.0 Apex

echo ================================================================================
echo [Studio 35: Aria Trinity DeltaLoop Biomorphic Interactive Chat CLI v2.0 Apex]
echo   Trinity Architecture: Gemma-4 ENGRAFT + Physarum + Task LoRA + RLT
echo ================================================================================
echo.

set "ENV_PY=H:\DiffusionQwen_env\Scripts\python.exe"

if not exist "%ENV_PY%" (
    echo [ERROR] Python environment not found: %ENV_PY%
    pause
    exit /b 1
)

echo [INFO] Initializing Biomorphic Cortex and Starting Chat Session...
echo.

"%ENV_PY%" "%~dp0chat_deltaloop_cli.py"

echo.
echo ================================================================================
echo [INFO] Chat Session Ended. Thank you, Master!
echo ================================================================================
pause
