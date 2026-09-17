@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title ALGO Trading Command Center

echo ======================================================
echo   Starting ALGO Command Center...
echo ======================================================

:: 1. Try standard 'python' command
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python run_dashboard.py
    goto :done
)

:: 2. Try 'py' launcher
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py -3.12 run_dashboard.py 2>nul
    if !ERRORLEVEL! EQU 0 goto :done
    py run_dashboard.py
    goto :done
)

:: 3. Try standard installation paths
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" run_dashboard.py
    goto :done
)

if exist "C:\Python312\python.exe" (
    "C:\Python312\python.exe" run_dashboard.py
    goto :done
)

if exist "C:\Program Files\Python312\python.exe" (
    "C:\Program Files\Python312\python.exe" run_dashboard.py
    goto :done
)

echo.
echo [ERROR] Python was not found on your system PATH.
echo Please install Python 3.12 and make sure "Add Python to PATH" is checked.
echo.

:done
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred. Press any key to close...
    pause >nul
)
