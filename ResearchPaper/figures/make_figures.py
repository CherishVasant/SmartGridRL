"""Figures for the paper, built only from the genuine RL runs in results/.

Run from anywhere: python ResearchPaper/figures/make_figures.py
Writes fig_methods.pdf, fig_learning.pdf and fig_week.pdf next to this script.
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / 'results'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                     'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7})


def kpi_means():
    rows = defaultdict(list)
    with open(RESULTS / 'challenge_kpis.csv') as f:
        for r in csv.DictReader(f):
            rows[r['method']].append(r)
    out = {}
    for m, rs in rows.items():
        vals = {k: [float(r[k]) for r in rs] for k in ['C', 'G', 'D', 'phase1', 'phase3']}
        out[m] = {k: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) for k, v in vals.items()}
    return out


def fig_methods():
    k = kpi_means()
    picks = [('B0_RBC', 'RBC'), ('B1_cost_only', 'PPO cost-only'), ('M3_multikpi_adaptive', 'PPO multi-KPI adaptive'),
             ('M9_load_squared', 'PPO load$^2$'), ('M12_sac_multikpi_fixed', 'SAC multi-KPI'),
             ('M20_sac_multikpi_counterfactual', 'SAC counterfactual'), ('M26_sac_score_aligned', 'SAC score-aligned'),
             ('M28_sac_phase1_aligned', 'SAC Phase I-aligned'), ('M27_sac_score_aligned_carbon2', 'SAC score-aligned, carbon 2'), ('M110_tqc_c2', 'TQC score-aligned, carbon 2'), ('M113_tqc_hist', 'TQC + history features')]
    y = np.arange(len(picks))[::-1]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for off, key, lab, col in [(-0.2, 'C', 'Cost $C$', '#4472c4'), (0.0, 'G', 'Carbon $G$', '#70ad47'),
                               (0.2, 'phase1', 'Phase I $(C+G)/2$', '#404040')]:
        means = [k[m][key][0] for m, _ in picks]
        errs = [k[m][key][1] for m, _ in picks]
        ax.barh(y + off, means, height=0.2, xerr=errs, color=col, label=lab, error_kw={'lw': 0.6, 'capsize': 1})
    ax.axvline(1.0, color='k', lw=0.6, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in picks])
    ax.set_xlim(0.6, 1.25)
    ax.set_xlabel('Relative to no battery (lower is better)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False, handlelength=1.2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(HERE / 'fig_methods.pdf', bbox_inches='tight', pad_inches=0.02)


def fig_learning():
    """Mean per-step training reward of M27 and M26. With the score-aligned counterfactual reward this is the weighted
    KPI improvement over no battery (0 = as good as idling), so the curve reads directly as progress."""
    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    for method, col, lab in [('M27_sac_score_aligned_carbon2', '#c00000', 'M27 (carbon 2)'),
                             ('M26_sac_score_aligned', '#4472c4', 'M26')]:
        for seed, ls in [(42, '-'), (100, '--'), (2026, ':')]:
            log = pd.read_csv(RESULTS / method / f'seed_{seed}' / 'train_log.csv')
            steps = log['env_step'].diff().fillna(log['env_step'].iloc[0])
            ax.plot(log['episode'], log['reward'] / steps, ls, color=col, lw=0.9,
                    label=f'{lab}, seed {seed}' if seed == 42 else None)
    ax.axhline(0.0, color='k', lw=0.6, ls='--')
    ax.set_xlabel('Training episode (1 simulated year each)')
    ax.set_ylabel('Mean reward per step')
    ax.set_ylim(-1.35, 0.65)
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(HERE / 'fig_learning.pdf', bbox_inches='tight', pad_inches=0.02)


def fig_week():
    tr = pd.read_csv(RESULTS / 'M27_sac_score_aligned_carbon2' / 'seed_42' / 'eval_trace.csv')
    nb = pd.read_csv(RESULTS / 'B5_no_battery' / 'seed_0' / 'eval_trace.csv')
    start = 24 * 182                                     # a week from early July
    w = tr[(tr['step'] >= start) & (tr['step'] < start + 168)]
    wn = nb[(nb['step'] >= start) & (nb['step'] < start + 168)]
    h = (w['step'] - start).values
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 3.2), sharex=True)
    axes[0].plot(h, wn['load'].values, color='#a0a0a0', lw=0.9, label='No battery')
    axes[0].plot(h, w['load'].values, color='#c00000', lw=0.9, label='SAC (M27)')
    axes[0].set_ylabel('District load (kW)')
    axes[0].legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    soc = w[[c for c in w.columns if c.startswith('soc_b')]].mean(axis=1).values
    axes[1].plot(h, soc, color='#404040', lw=0.9)
    axes[1].set_ylabel('Mean SoC')
    axes[1].set_ylim(0, 1)
    axes[2].plot(h, w['pricing'].values, color='#4472c4', lw=0.9)
    axes[2].set_ylabel('Price (\\$/kWh)', color='#4472c4')
    ax2 = axes[2].twinx()
    ax2.plot(h, w['carbon_intensity'].values, color='#70ad47', lw=0.9)
    ax2.set_ylabel('kgCO$_2$/kWh', color='#70ad47')
    axes[2].set_xlabel('Hour of the week')
    fig.tight_layout()
    fig.savefig(HERE / 'fig_week.pdf', bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    fig_methods()
    fig_learning()
    fig_week()
    print('figures written to', HERE)
