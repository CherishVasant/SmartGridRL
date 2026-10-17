"""Runs a method notebook's code cells in a plain Python process (no Jupyter kernel); `%run ./00_core.ipynb` is
replaced by executing 00_core's code cells. Usage: python run_notebook.py <notebook.ipynb> (from the phase 3 folder)."""
import json
import sys

# Labels contain φ / μ; a redirected stdout on Windows may default to cp1252
# line_buffering: every printed line reaches the log file at once (redirected stdout is otherwise block-buffered)
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)


def code_cells(path):
    nb = json.load(open(path, encoding='utf-8'))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


namespace = {'__name__': '__main__'}
for src in code_cells(sys.argv[1]):
    if src.strip().startswith('%run'):
        first, _, rest = src.strip().partition('\n')
        for core_src in code_cells(first.split()[1]):
            exec(core_src, namespace)
        exec(rest, namespace)   # the rest of the cell (e.g. extra imports after the %run line)
    else:
        exec(src, namespace)
print('\nFinished:', sys.argv[1], flush=True)
