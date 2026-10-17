"""Runs the Stage A notebooks (via run_notebook.py) with at most MAX_PARALLEL at once, only when enough commit memory
is free, and restarts any run that dies (it resumes from its checkpoint). Exits when all finish or one fails 3 times."""
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parent.parent   # 02_multi_kpi_adaptive_reward
PYTHON = str(PHASE_DIR.parent / 'venv' / 'Scripts' / 'python.exe')
RUNNER = str(Path(__file__).parent / 'run_notebook.py')
QUEUE = ['M1_dynamic_sharp', 'M2_multikpi_fixed', 'M4_per_building_dynamic', 'M3_multikpi_adaptive',
         'M5_per_building_adaptive_shared', 'M6_per_building_adaptive_own', 'B1_cost_only', 'B3_static_matched', 'B4_dynamic']
MAX_PARALLEL = int(sys.argv[1]) if len(sys.argv) > 1 else 6
if len(sys.argv) > 2:   # optional comma-separated subset, so a second supervisor can run other notebooks
    QUEUE = sys.argv[2].split(',')
MIN_FREE_COMMIT_GB = 1.6
# Optional override, re-read every loop, so the free-memory reserve can be changed without restarting the supervisor
MIN_FREE_FILE = Path(__file__).parent / 'min_free_commit_gb.txt'
MAX_ATTEMPTS = 3
LOGS = PHASE_DIR / 'results' / 'logs'


def min_free_gb():
    try:
        return float(MIN_FREE_FILE.read_text().strip())
    except (OSError, ValueError):
        return MIN_FREE_COMMIT_GB


def running_elsewhere():
    """Queue notebooks already running in a process this supervisor did not start (e.g. after a supervisor restart)."""
    out = subprocess.run(['powershell', '-NoProfile', '-Command',
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.CommandLine }"],
                         capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW).stdout
    return {nb for nb in QUEUE if f'run_notebook.py {nb}.ipynb' in out}


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def free_commit_gb():
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullAvailPageFile / 1024 ** 3


def notebook_seeds(nb):
    """The METHOD and SEEDS set in a method notebook's code cells (None if not found)."""
    import json, re
    src = '\n'.join(''.join(c['source']) for c in json.load(open(PHASE_DIR / f'{nb}.ipynb', encoding='utf-8'))['cells']
                    if c['cell_type'] == 'code')
    method, seeds = re.search(r"METHOD\s*=\s*'([^']+)'", src), re.search(r"SEEDS\s*=\s*\[([^\]]*)\]", src)
    if not (method and seeds):
        return None, None
    return method.group(1), [int(s) for s in seeds.group(1).split(',') if s.strip()]


def finished(nb):
    """Done when the log says so, or when every seed in the notebook has its summary.json (the log may lag)."""
    log = LOGS / f'{nb}.log'
    if log.exists() and 'Finished:' in log.read_text(encoding='utf-8', errors='ignore'):
        return True
    method, seeds = notebook_seeds(nb)
    return bool(seeds) and all((PHASE_DIR / 'results' / method / f'seed_{s}' / 'summary.json').exists() for s in seeds)


def launch(nb, attempt):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    env.pop('SMARTGRID_SMOKE_TEST', None)
    out = open(LOGS / f'{nb}.log', 'a', encoding='utf-8')
    err = open(LOGS / f'{nb}.err.log', 'a', encoding='utf-8')
    out.write(f"\n===== supervisor start, attempt {attempt}, {time.strftime('%H:%M')} =====\n")
    out.flush()
    return subprocess.Popen([PYTHON, RUNNER, f'{nb}.ipynb'], cwd=PHASE_DIR, stdout=out, stderr=err,
                            creationflags=subprocess.CREATE_NO_WINDOW)


# Keep Windows from idle-sleeping while the supervisor runs (a per-process request, dropped when it exits)
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

running, attempts = {}, {nb: 0 for nb in QUEUE}
while True:
    for nb, proc in list(running.items()):
        if proc.poll() is not None:
            del running[nb]
            if finished(nb):
                print(f"{time.strftime('%H:%M')} FINISHED {nb}", flush=True)
            else:
                print(f"{time.strftime('%H:%M')} DIED {nb} (exit {proc.returncode}); will resume", flush=True)
                if attempts[nb] >= MAX_ATTEMPTS:
                    print(f"GAVE UP on {nb} after {MAX_ATTEMPTS} attempts", flush=True)
                    sys.exit(1)

    # Never start a second writer for a notebook that is already running outside this supervisor
    elsewhere = running_elsewhere() - set(running)
    pending = [nb for nb in QUEUE if nb not in running and nb not in elsewhere and not finished(nb)]
    if not pending and not running and not elsewhere:
        print(f"{time.strftime('%H:%M')} ALL DONE", flush=True)
        sys.exit(0)
    for nb in pending:
        if len(running) + len(elsewhere) >= MAX_PARALLEL or free_commit_gb() < min_free_gb():
            break
        attempts[nb] += 1
        running[nb] = launch(nb, attempts[nb])
        print(f"{time.strftime('%H:%M')} started {nb} (attempt {attempts[nb]}), free commit {free_commit_gb():.1f} GB",
              flush=True)
        time.sleep(30)   # let it allocate before checking memory for the next one
    time.sleep(60)
