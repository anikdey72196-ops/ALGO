@echo off
cd /d "%~dp0"
title ALGO Trading Command Center

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python run_dashboard.py
) else if exist "C:\Users\anikd\AppData\Local\Programs\Python\Python312\python.exe" (
    "C:\Users\anikd\AppData\Local\Programs\Python\Python312\python.exe" run_dashboard.py
) else if exist "C:\Python312\python.exe" (
    "C:\Python312\python.exe" run_dashboard.py
) else (
    py run_dashboard.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred. Press any key to close...
    pause >nul
)
