"""Runs a method notebook's code cells in a plain Python process (no Jupyter kernel); `%run ./00_core.ipynb` is
replaced by executing 00_core's code cells. Usage: python run_notebook.py <notebook name or path> (from the project folder); a name is found in code/Round_XX."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rounds import find_notebook   # noqa: E402

# Labels contain φ / μ; a redirected stdout on Windows may default to cp1252
# line_buffering: every printed line reaches the log file at once (redirected stdout is otherwise block-buffered)
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)


def code_cells(path):
    nb = json.load(open(path, encoding='utf-8'))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


namespace = {'__name__': '__main__'}
NOTEBOOK = Path(sys.argv[1]) if Path(sys.argv[1]).exists() else find_notebook(sys.argv[1])
for src in code_cells(NOTEBOOK):
    if src.strip().startswith('%run'):
        first, _, rest = src.strip().partition('\n')
        for core_src in code_cells(NOTEBOOK.parent / first.split()[1]):
            exec(core_src, namespace)
        exec(rest, namespace)   # the rest of the cell (e.g. extra imports after the %run line)
    else:
        exec(src, namespace)
print('\nFinished:', sys.argv[1], flush=True)
