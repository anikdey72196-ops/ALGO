@echo off
cd /d "%~dp0"
title ALGO Trading Command Center

"C:\Users\anikd\AppData\Local\Programs\Python\Python312\python.exe" run_dashboard.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred. Press any key to close...
    pause >nul
)
