@echo off
REM Launch Algorithmic Trading Web Dashboard using Python 3.12
echo ======================================================
echo   Launching Algorithmic Trading Control Station
echo   URL: http://127.0.0.1:8000
echo ======================================================
"C:\Users\anikd\AppData\Local\Programs\Python\Python312\python.exe" -m uvicorn web_app:app --host 127.0.0.1 --port 8000 --reload
pause
