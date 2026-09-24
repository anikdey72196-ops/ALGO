@echo off
setlocal
cd /d "%~dp0"
title ALGO Trading Command Center

echo ======================================================
echo   Starting ALGO Command Center...
echo ======================================================

:: 1. Check Virtual Environments
if exist "myenv\Scripts\python.exe" (
    set "PY_EXE=myenv\Scripts\python.exe"
    goto :run
)
if exist ".venv\Scripts\python.exe" (
    set "PY_EXE=.venv\Scripts\python.exe"
    goto :run
)
if exist "venv\Scripts\python.exe" (
    set "PY_EXE=venv\Scripts\python.exe"
    goto :run
)

:: 2. Try py launcher
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py -3.12 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PY_EXE=py -3.12"
        goto :run
    )
    py -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PY_EXE=py -3"
        goto :run
    )
    set "PY_EXE=py"
    goto :run
)

:: 3. Try python on PATH
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY_EXE=python"
    goto :run
)

:: 4. Try standard installation locations
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :run
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :run
)
if exist "C:\Python312\python.exe" (
    set "PY_EXE=C:\Python312\python.exe"
    goto :run
)
if exist "C:\Python311\python.exe" (
    set "PY_EXE=C:\Python311\python.exe"
    goto :run
)
if exist "C:\Program Files\Python312\python.exe" (
    set "PY_EXE=C:\Program Files\Python312\python.exe"
    goto :run
)
if exist "C:\Program Files\Python311\python.exe" (
    set "PY_EXE=C:\Program Files\Python311\python.exe"
    goto :run
)

echo.
echo [ERROR] Python was not found on this system.
echo Please ensure Python 3.11 or 3.12 is installed and added to PATH.
echo.
pause
exit /b 1

:run
echo Using interpreter: %PY_EXE%
%PY_EXE% run_dashboard.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred. Press any key to close...
    pause >nul
)
