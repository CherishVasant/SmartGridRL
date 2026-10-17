"""Keeps Windows from idle-sleeping while any training run (run_notebook.py) is alive, then exits.
Uses SetThreadExecutionState, a per-process request that Windows drops when this process ends (no settings change).
Usage: python keep_awake.py   (run detached; it exits 10 minutes after the last run is gone)."""
import ctypes
import subprocess
import time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def runs_alive():
    out = subprocess.run(['powershell', '-NoProfile', '-Command',
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.CommandLine }"],
                         capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW).stdout
    return 'run_notebook.py' in out


ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
idle_since = None
while True:
    if runs_alive():
        idle_since = None
    elif idle_since is None:
        idle_since = time.time()
    elif time.time() - idle_since > 600:
        break
    time.sleep(60)
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
