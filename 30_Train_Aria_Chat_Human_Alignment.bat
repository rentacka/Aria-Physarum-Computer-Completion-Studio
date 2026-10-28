@echo off
rem ================================================================================
rem Studio 35: Aria Chat Human Alignment Pipeline (7-bit ASCII Sentinel)
rem Fine-tunes Gemma-4 DeltaLoop with Completion-Only Masking (~1 min, 120 steps)
rem ================================================================================

chcp 65001 > nul
set "PYTHONIOENCODING=utf-8"

title Aria DeltaLoop Human Chat Alignment Pipeline v2.2 Apex

echo ================================================================================
echo [Studio 35: Aria DeltaLoop Human Chat Alignment Pipeline v2.2 Apex]
echo   Completion-Only Masking: Enforce prompt understanding and prevent mode collapse
echo   Data: 700 clean pairs (dataset_cache/aria_chat_alignment_dataset.jsonl)
echo   Estimated Time: ~40-60 seconds (120 steps)
echo ================================================================================
echo.

set "ENV_PY=H:\DiffusionQwen_env\Scripts\python.exe"

if not exist "%ENV_PY%" (
    echo [ERROR] Python environment not found: %ENV_PY%
    pause
    exit /b 1
)

echo [Step 1] Checking alignment dataset...
if not exist "dataset_cache\aria_chat_alignment_dataset.jsonl" (
    echo [INFO] Generating 700-pair alignment dataset...
    "%ENV_PY%" build_aria_chat_alignment_dataset.py
)

echo.
echo [Step 2] Launching Completion-Only Dialogue Alignment Fine-Tuning...
echo.

"%ENV_PY%" align_aria_chat.py --steps 120 --lr 4.0e-5

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Alignment process failed with exit code: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ================================================================================
echo [SUCCESS] Completion-Only Dialogue Alignment Complete!
echo You can now talk smoothly with Aria Trinity CLI (29_Chat_Aria_Trinity_DeltaLoop.bat)!
echo ================================================================================
pause
