@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title ALGO Trading Command Center

echo ======================================================
echo   Starting ALGO Command Center...
echo ======================================================

:: 1. Check Virtual Environment in project directory
if exist ".venv\Scripts\python.exe" (
    echo Using project virtual environment (.venv)...
    ".venv\Scripts\python.exe" run_dashboard.py
    goto :done
)
if exist "venv\Scripts\python.exe" (
    echo Using project virtual environment (venv)...
    "venv\Scripts\python.exe" run_dashboard.py
    goto :done
)

:: 2. Try 'py' launcher with 3.12 or latest 3.x
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py -3.12 -c "import sys" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo Using Python launcher (py -3.12)...
        py -3.12 run_dashboard.py
        goto :done
    )
    py -3 -c "import sys" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo Using Python launcher (py -3)...
        py -3 run_dashboard.py
        goto :done
    )
    echo Using Python launcher (py)...
    py run_dashboard.py
    goto :done
)

:: 3. Try standard 'python' command
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using system 'python'...
    python run_dashboard.py
    goto :done
)

:: 4. Try common Python installation paths
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    echo Using Python from %LOCALAPPDATA%\Programs\Python\Python312...
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" run_dashboard.py
    goto :done
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    echo Using Python from %LOCALAPPDATA%\Programs\Python\Python311...
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" run_dashboard.py
    goto :done
)

if exist "C:\Python312\python.exe" (
    echo Using Python from C:\Python312...
    "C:\Python312\python.exe" run_dashboard.py
    goto :done
)

if exist "C:\Python311\python.exe" (
    echo Using Python from C:\Python311...
    "C:\Python311\python.exe" run_dashboard.py
    goto :done
)

if exist "C:\Program Files\Python312\python.exe" (
    echo Using Python from C:\Program Files\Python312...
    "C:\Program Files\Python312\python.exe" run_dashboard.py
    goto :done
)

if exist "C:\Program Files\Python311\python.exe" (
    echo Using Python from C:\Program Files\Python311...
    "C:\Program Files\Python311\python.exe" run_dashboard.py
    goto :done
)

echo.
echo [ERROR] Python was not found on this system.
echo Please ensure Python 3.11 or 3.12 is installed and added to PATH.
echo.

:done
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred. Press any key to close...
    pause >nul
)
