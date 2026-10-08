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

def kill_process_on_port(port: int):
    try:
        import subprocess
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess"],
            capture_output=True, text=True
        )
        pids = set(result.stdout.strip().split())
        for pid_str in pids:
            if pid_str.isdigit() and int(pid_str) > 0:
                pid = int(pid_str)
                print(f"Terminating existing process on port {port} (PID {pid})...")
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        time.sleep(1.5)
    except Exception as e:
        print(f"Error terminating process on port {port}: {e}")

def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)

    # 1. Ensure execution within virtual environment if available
    venv_py = script_dir / "myenv" / "Scripts" / "python.exe"
    if venv_py.exists():
        curr_py = Path(sys.executable).resolve()
        if curr_py != venv_py.resolve():
            import subprocess
            sys.exit(subprocess.call([str(venv_py), str(Path(__file__).resolve())] + sys.argv[1:]))

    print("======================================================")
    print("  ALGO COMMAND CENTER - WEB CONTROL STATION")
    print("  URL: http://127.0.0.1:8000")
    print("======================================================")

    force_restart = "--restart" in sys.argv or "-r" in sys.argv

    if force_restart:
        if is_port_in_use(8000):
            print("Restarting server on port 8000...")
            kill_process_on_port(8000)
    elif is_port_in_use(8000):
        print("\n[ACTIVE] Port 8000 is ALREADY RUNNING and serving traffic in the background!")
        print("          The trading engine and web dashboard are live at http://127.0.0.1:8000\n")
        try:
            choice = input("Press [R] to restart in this console, or [Enter] to keep running in background: ").strip().lower()
            if choice in ('r', 'restart', 'yes', 'y'):
                print("Stopping background instance and restarting here...")
                kill_process_on_port(8000)
            else:
                print("Keeping existing background server active. Opening dashboard in browser...")
                webbrowser.open("http://127.0.0.1:8000")
                sys.exit(0)
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)

    threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
