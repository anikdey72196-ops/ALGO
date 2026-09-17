# run_dashboard.py - Clean entrypoint for launching Web Control Station
import os
import sys
import time
import socket
import webbrowser
import threading
from pathlib import Path

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)

    print("======================================================")
    print("  ALGO COMMAND CENTER - WEB CONTROL STATION")
    print("  URL: http://127.0.0.1:8000")
    print("======================================================")

    if is_port_in_use(8000):
        print("Port 8000 is already active.")
        print("Opening http://127.0.0.1:8000 in your browser...")
        webbrowser.open("http://127.0.0.1:8000")
        sys.exit(0)

    threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
