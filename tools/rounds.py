"""Round of every method, read from the `# Round N` comments in tools/core_src.py, and the folder of its notebook.

Layout: code/00_core.ipynb and code/99_compare_results.ipynb, and one folder per round, code/Round_XX/ (Round_00 =
baselines B* and Stage A M1-M6), holding every method notebook of that round (including the __s<seed> copies).
Used by build_notebooks.py, supervisor.py and run_notebook.py.
"""
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_DIR / 'code'
# Methods defined without their own round comment (the full-budget follow-ups of Round 23)
OVERRIDES = {m: 24 for m in ['M109_sac_c2_400k', 'M110_tqc_c2', 'M111_sac_hist_rich_obsnorm', 'M112_sac_hist_200k',
                              'M113_tqc_hist', 'M114_tqc_score_aligned']}


def method_rounds():
    """{method: round number}, from the round comments in tools/core_src.py."""
    rounds, current = {}, 0
    for line in (Path(__file__).parent / 'core_src.py').read_text(encoding='utf-8').splitlines():
        m = re.match(r"\s*# Round (\d+)", line)
        if m:
            current = int(m.group(1))
        for key in re.findall(r"'((?:M|B)\d+_[A-Za-z0-9_]+)'\s*(?::|\]\s*=)", line):
            rounds.setdefault(key, current)
    rounds.update(OVERRIDES)
    return rounds


def round_folder(method):
    """Folder of a method's notebooks (the name may carry a __s<seed> suffix)."""
    return CODE_DIR / f'Round_{method_rounds().get(method.split("__")[0], 0):02d}'


def find_notebook(name):
    """Path of <name>.ipynb: its round folder, code/, or anywhere under code/."""
    name = name[:-6] if name.endswith('.ipynb') else name
    for p in [round_folder(name) / f'{name}.ipynb', CODE_DIR / f'{name}.ipynb']:
        if p.exists():
            return p
    hits = list(CODE_DIR.rglob(f'{name}.ipynb'))
    return hits[0] if hits else CODE_DIR / f'{name}.ipynb'
