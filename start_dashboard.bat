@echo off
cd /d "%~dp0"
title ALGO Trading Command Center

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" run_dashboard.py
    goto :done
)

py -3.12 -c "import loguru" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py -3.12 run_dashboard.py
    goto :done
)

python -c "import loguru" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python run_dashboard.py
    goto :done
)

py run_dashboard.py

:done
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred. Press any key to close...
    pause >nul
)
