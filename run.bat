@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_demo.py %*
) else (
    echo Virtual environment not found.
    pause
)
