"""Writes the Obsidian note '04 - Phase 3 Results Summary.md' (one table per round) from the results on disk.

Every KPI is the official CityLearn Challenge 2022 definition, normalised to the no-battery run (lower is better):
Phase I = mean(C, G) (the published leaderboard on our 5 buildings), Score = mean(C, G, D). Values are means over the
finished seeds. In each table the best value of every column is in *italics*; the best value of each column over all
our deployable methods (every table except the references) is in **bold**.
Usage (from the phase 3 folder): python tools/make_summary.py
"""
import torch   # noqa: F401  (before pandas, or Windows fails to load torch's DLLs: WinError 1114)
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import challenge_kpis as ck   # noqa: E402

PHASE_DIR = Path(__file__).resolve().parent.parent
RESULTS = PHASE_DIR / 'results'
OUT = PHASE_DIR / '04 - Phase 3 Results Summary.md'

# Method labels from the core config (its first two code cells: imports and configuration)
_ns = {'__name__': '__main__'}
for _src in [''.join(c['source']) for c in json.load(open(PHASE_DIR / '00_core.ipynb', encoding='utf-8'))['cells']
             if c['cell_type'] == 'code'][:2]:
    exec(_src, _ns)
LABELS = {m: cfg['label'] for m, cfg in _ns['METHODS'].items()}
LABELS.update({'B1_cost_only': 'Cost-only PPO', 'B3_static_matched': 'Static carbon weight λ = 0.41 (PPO)',
               'B4_dynamic': 'Dynamic carbon weight λ·Î_t (PPO)'})

# Published Phase I leaderboard (AIcrowd, NeurIPS 2022 CityLearn Challenge, 5 training buildings = our dataset), top 15
# real entries (rank-1 "Greener" shows 0.000 / nan and is excluded as broken)
PUBLISHED_PHASE1 = [('tymur_prorochenko', 0.631), ('Together', 0.635), ('ROLEVT', 0.651), ('OneWo', 0.698),
                    ('LA_gang', 0.719), ('DivMARL', 0.728), ('JamesCheng', 0.740), ('ludwigbald', 0.752),
                    ('agnprz', 0.758), ('random_charger', 0.760), ('CUFE', 0.768), ('JoseHS_RL', 0.768),
                    ('TaCo', 0.800), ('Panic', 0.801)]


def published_rank(phase1):
    """Where this Phase I score would place among the published entries ('-' if behind all of them)."""
    beaten = [name for name, s in PUBLISHED_PHASE1 if phase1 < s]
    if not beaten:
        return '–'
    return f'#{len(PUBLISHED_PHASE1) - len(beaten) + 1} (beats {beaten[0]} {dict(PUBLISHED_PHASE1)[beaten[0]]:.3f})'

ROUNDS = [
    ('Baselines (PPO, phase 2 rewards)', ['B1', 'B3', 'B4']),
    ('Stage A (PPO reward designs)', ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']),
    ('Round 1 (normalisation, load², district terms, SAC)', ['M7', 'M8', 'M9', 'M10', 'M11', 'M12']),
    ('Round 2 (SAC combinations)', ['M13', 'M14', 'M15', 'M16']),
    ('Round 3 (SAC + fixed multi-KPI variants)', ['M17', 'M18', 'M19', 'M20']),
    ('Round 4 (counterfactual + full-day peak; collapsed, stopped)', ['M21', 'M24']),
    ('Round 5 (score-aligned rewards)', ['M26', 'M27', 'M28']),
    ('Round 6 (residual SAC around an LP plan; branch closed)', ['M29', 'M30', 'M31', 'M32']),
    ('Round 7 (behaviour cloning of the LP plans)', ['M33', 'M34', 'M35', 'M36']),
    ('Round 8 (bigger clones, DAgger)', ['M37', 'M38', 'M39', 'M40']),
    ('Round 9 (gentle DAgger, clone + SAC fine-tuning)', ['M41', 'M42', 'M43', 'M44', 'M45', 'M46']),
    ('Round 11 (doubling down on M41)', ['M48', 'M49', 'M50']),
    ('Round 12 (doubling down on M51)', ['M51', 'M52', 'M53', 'M54', 'M55', 'M56']),
    ('Round 13 (deep clones of plans with a better Phase I score)', ['M57', 'M58', 'M59', 'M60']),
    ('Round 14 (bigger networks, more DAgger)', ['M61', 'M62', 'M63', 'M64']),
    ('Round 15 (big clones of the Phase I plan O3)', ['M65', 'M66', 'M67']),
]
MEMBER_RUNS = {'M47': 'extra seeds of M33 (ensemble members)', 'M48': None, 'M52': 'extra seeds of M51 (ensemble members)'}
COLS = [('phase1', 'Phase I'), ('phase3', 'Score'), ('C', 'Cost C'), ('G', 'Carbon G'), ('R', 'Ramping'),
        ('one_minus_L', '1 − LF'), ('D', 'Grid D'), ('daily_peak', 'Daily peak'), ('annual_peak', 'Annual peak')]


def peaks(trace):
    e = trace['load'].values
    n = len(e) // 24
    return e[:n * 24].reshape(n, 24).max(1).mean(), e.max()


nb_trace = pd.read_csv(RESULTS / 'B5_no_battery' / 'seed_0' / 'eval_trace.csv')
nb_raw, nb_peaks = ck.raw(nb_trace), peaks(nb_trace)


def kpis_of(trace):
    k = ck.challenge_kpis(trace, nb_raw)
    dp, ap = peaks(trace)
    return {'phase1': k['phase1'], 'phase3': k['phase3'], 'C': k['C'], 'G': k['G'], 'R': k['R'], 'one_minus_L': k['1-L'],
            'D': k['D'], 'daily_peak': dp / nb_peaks[0], 'annual_peak': ap / nb_peaks[1]}


def run_folder(prefix):
    return next((p for p in sorted(RESULTS.iterdir()) if p.is_dir() and re.fullmatch(rf'{prefix}_.+', p.name)), None)


def row_for(folder, label=None):
    seeds = sorted(folder.glob('seed_*/summary.json'))
    if not seeds:
        return None
    vals = [kpis_of(pd.read_csv(s.parent / 'eval_trace.csv')) for s in seeds]
    row = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    row['n'] = len(seeds)
    row['std'] = float(np.std([v['phase1'] for v in vals], ddof=1)) if len(vals) > 1 else float('nan')
    row['label'] = label or LABELS.get(folder.name, folder.name)
    return row


def fmt_table(rows, overall_best, mark_table_best=True):
    """rows: list of (id, row). Italics = best in this table, bold = best over all deployable methods."""
    best_here = {c: min(r[c] for _, r in rows) for c, _ in COLS} if rows else {}
    head = '| ID | Method | Seeds | ' + ' | '.join(name for _, name in COLS) + ' | Published rank (Phase I) |'
    lines = [head, '|' + '---|' * (len(COLS) + 4)]
    for rid, r in rows:
        cells = []
        for c, _ in COLS:
            v, s = r[c], f'{r[c]:.3f}'
            if overall_best is not None and abs(v - overall_best[c]) < 5e-4:
                s = f'**{s}**'
            elif mark_table_best and abs(v - best_here[c]) < 5e-4 and len(rows) > 1:
                s = f'*{s}*'
            cells.append(s)
        seeds = f"{r['n']}" + (f" (±{r['std']:.3f})" if r['n'] > 1 else '')
        lines.append(f"| {rid} | {r['label']} | {seeds} | " + ' | '.join(cells) + f" | {published_rank(r['phase1'])} |")
    return '\n'.join(lines)


# ---- Collect
tables = []
for title, ids in ROUNDS:
    rows = []
    for mid in ids:
        folder = run_folder(mid)
        if folder is not None and (r := row_for(folder)) is not None:
            rows.append((mid, r))
    tables.append((title, rows))
def ensemble_label(p):
    """'Mean action of M41 ×2, M42 ×2, ...' from the ensemble's summary label."""
    label = json.load(open(p / 'seed_0' / 'summary.json'))['label']
    parts = re.findall(r'(M\d+)_[A-Za-z0-9_]+ (?:seeds )?\[([^\]]*)\]', label)
    return 'Mean action of ' + ', '.join(f"{m} ×{len(s.split(','))}" for m, s in parts) + \
           f" ({sum(len(s.split(',')) for _, s in parts)} clones)"


ensembles = [(p.name, row_for(p, ensemble_label(p))) for p in sorted(RESULTS.glob('E_*'))
             if (p / 'seed_0' / 'summary.json').exists()]
ensembles = [(i, r) for i, r in ensembles if r]
plans = [(p.name.split('_')[0], row_for(p, json.load(open(p / 'seed_0' / 'summary.json'))['label']))
         for p in sorted(RESULTS.glob('O*_*')) if (p / 'seed_0' / 'summary.json').exists()]
refs = [('B5', row_for(RESULTS / 'B5_no_battery', 'No battery (the reference = 1.000)'))]
rbc = kpis_of(pd.read_csv(RESULTS / 'B0_RBC' / 'eval_trace.csv'))
refs.append(('B0', dict(rbc, n=1, std=float('nan'), label='Rule-based controller (CityLearn BasicRBC)')))

# Residual SAC around a perfect-foresight plan needs the plan (i.e. the future) at run time: not deployable
PLAN_ASSISTED = {'M29', 'M30', 'M31', 'M32'}
deployable = [r for _, rows in tables for i, r in rows if i not in PLAN_ASSISTED] + [r for _, r in ensembles]
overall = {c: min(r[c] for r in deployable) for c, _ in COLS}
best_single = min(((i, r) for _, rows in tables for i, r in rows), key=lambda x: x[1]['phase1'])
best_ens = min(ensembles, key=lambda x: x[1]['phase1']) if ensembles else None

def beats_callout():
    """Callout listing every method / ensemble / plan that beats published entries."""
    lines = ['', '> [!success] What beats existing published work',
             '> Phase I leaderboard (same 5 buildings, mean of cost and carbon vs no battery). Published top entries:',
             '> ' + ', '.join(f'{n} {s:.3f}' for n, s in PUBLISHED_PHASE1[:7]) + '.', '>']
    groups = [('Deployable single models (our RL / imitation)',
               [(i, r) for _, rows in tables for i, r in rows if i not in PLAN_ASSISTED]),
              ('Ensembles of clones (deployable)', ensembles),
              ('Plan-assisted (residual SAC around the LP plan; needs the plan at run time)',
               [(i, r) for _, rows in tables for i, r in rows if i in PLAN_ASSISTED]),
              ('Perfect-foresight LP plans (planning references, not deployable)', plans)]
    for title, items in groups:
        winners = sorted([(i, r) for i, r in items if published_rank(r['phase1']) != '–'], key=lambda x: x[1]['phase1'])
        if winners:
            lines.append(f'> **{title}:**')
            for i, r in winners:
                lines.append(f"> - {i}: Phase I {r['phase1']:.3f} → {published_rank(r['phase1'])}; Score {r['phase3']:.3f}"
                             f"{' (below the Phase III winner 0.728)' if r['phase3'] < 0.728 else ''}; carbon G {r['G']:.3f}"
                             f" ({r['n']} seed{'s' if r['n'] > 1 else ''})")
    lines.append('> - Caveat: the Phase III public score (Together 0.728) was measured on training + validation buildings, '
                 'so the Score comparison is indicative only.')
    return '\n'.join(lines)


# ---- Write
md = [f"""---
title: Phase 3 Results Summary
tags:
  - smartgrid-rl
  - research-paper
  - results
aliases:
  - Phase 3 Results
---

# 04 · Phase 3 Results Summary

Previous: [[03 - Results and Discussion]] (phase 2) · Generated by `tools/make_summary.py` from `results/` (rerun it to refresh).

> [!abstract] Where things stand
> - **Best single deployable model:** {best_single[0]} ({best_single[1]['label']}): Phase I **{best_single[1]['phase1']:.3f}**,
>   Score {best_single[1]['phase3']:.3f}, carbon G {best_single[1]['G']:.3f} ({best_single[1]['n']} seed(s)).""" +
      (f"""
> - **Best ensemble:** {best_ens[0]}: Phase I **{best_ens[1]['phase1']:.3f}**, Score {best_ens[1]['phase3']:.3f}, carbon G {best_ens[1]['G']:.3f}.""" if best_ens else '') + """
> - **Published state of the art, Phase I (same 5 buildings):** 0.631 (tymur_prorochenko), 0.635 (Together, overall winner),
>   0.651 (ROLEVT), 0.698 (OneWo), 0.719 (LA_gang), 0.728 (DivMARL).
> - **Perfect-foresight LP plan** (planning reference, not deployable): Phase I 0.629 (O3), Score 0.624 (O2).
""" + beats_callout() + """

> [!info] How to read the tables
> - Every number is the **official CityLearn Challenge 2022 KPI ÷ the no-battery run**: below 1.000 is better than doing
>   nothing; 0.700 means 30% less than no battery. **Lower is better in every column.**
> - **Phase I** = mean(cost C, carbon G), the published leaderboard on our 5 buildings. **Score** = mean(C, G, D), the
>   final-phase formula. **Grid D** = mean(ramping, 1 − load factor). Daily / annual peak = mean daily maximum / yearly
>   maximum of the district load.
> - Cost and carbon are clipped at 0 on the **district** net load (exports earn nothing; buildings offset each other).
> - Values are means over the finished seeds; "±" is the seed standard deviation of Phase I.
> - *Italics* = best value in that column **within the table**. **Bold** = best value in that column over **all our
>   deployable methods** (all tables except References, LP plans and the plan-assisted Round 6).
"""]
md.append('## References (not our methods)\n')
md.append(fmt_table(refs, None, mark_table_best=False))
md.append('\n| Published (CityLearn Challenge 2022) | | | Phase I | Score | Cost C | Carbon G | | | Grid D | | |\n'
          '|---|---|---|---|---|---|---|---|---|---|---|---|\n'
          '| #1 Phase I | tymur_prorochenko | – | 0.631 | – | 0.544 | 0.718 | – | – | – | – | – |\n'
          '| #2 Phase I (overall winner) | Together | – | 0.635 | – | 0.578 | 0.691 | – | – | – | – | – |\n'
          '| #3 Phase I | ROLEVT | – | 0.651 | – | 0.554 | 0.749 | – | – | – | – | – |\n'
          '| Phase III public #1 (train + validation buildings) | Together | – | – | 0.728 | 0.639 | 0.808 | – | – | 0.738 | – | – |\n')
md.append('\n## Perfect-foresight LP plans (planning references, `tools/lp_oracle.py`)\n')
md.append(fmt_table(plans, None))
for title, rows in tables:
    if rows:
        md.append(f'\n## {title}\n')
        md.append(fmt_table(rows, overall))
if ensembles:
    md.append('\n## Round 10+ (ensembles of clones, `tools/ensemble_eval.py`)\n')
    md.append(fmt_table(ensembles, overall))
md.append("""
## Notes on specific rows

- **M5, M6, M10, M21, M24:** the battery is never used (collapse to "do nothing"), so every KPI is about 1.000.
- **M21, M24** were stopped after seed 42; **M29–M32** (residual SAC around the plan) after seed 42, as they were clearly
  worse than the plan itself.
- **M47, M48, M52** are extra seeds of M33 / M41 / M51 used only as ensemble members; M48 is listed in Round 11.
- Rounds 11–12 are still running: rows with fewer than 3 seeds will change.
""")
OUT.write_text('\n'.join(md), encoding='utf-8')
print(f'wrote {OUT} ({sum(len(r) for _, r in tables)} methods, {len(ensembles)} ensembles, {len(plans)} LP plans)')
