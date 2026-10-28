@echo off
chcp 65001 > nul
title Studio 35: Aria-Apex DeltaLoop Ultra-Fast Studio (Apex v1.0)
echo ===============================================================================
echo   Studio 35: Aria-Apex DeltaLoop Ultra-Fast AI Studio
echo   (0-KV Gated DeltaNet x Recurrent Looped x MTP-4 Parallel Speculation)
echo ===============================================================================
echo.

set PYTHON_ENV=H:\DiffusionQwen_env\Scripts\python.exe

if not exist "%PYTHON_ENV%" (
    echo [!] H:\DiffusionQwen_env not found. Using default python...
    set PYTHON_ENV=python
)

echo [1/2] Preparing environment...
cd /d "%~dp0"

echo [2/2] Running DeltaLoop vs Transformer Benchmark...
echo.
"%PYTHON_ENV%" benchmark_deltaloop_vs_transformer.py

echo.
echo ===============================================================================
echo   Aria: Master, thank you so much! Our DeltaLoop is blazingly fast!
echo ===============================================================================
pause
