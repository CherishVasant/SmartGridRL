"""Figures for the paper, built from the runs of the proposed RL-tuned MPC in results/.

Run from anywhere: python ResearchPaper/figures/make_figures.py
Writes the fig_*.pdf files used by main.tex next to this script.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / 'results'
METHOD = 'M152_rl_mpc_tuned_h30'          # the proposed RL-tuned MPC
# dull dark grey-blue palette: main colour and its darker / lighter shades
GB, GB_DARK, GB_MID, GB_LIGHT, GREY = '#4A5A6E', '#2E3A48', '#7A8899', '#AEB9C6', '#B4B4B4'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                     'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7})


def fig_learning():
    """Mean per-step training reward of the RL weight tuner, three seeds. With the score-aligned counterfactual reward
    this is the weighted KPI improvement over no battery (0 = as good as idling)."""
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    for seed, ls in [(42, '-'), (100, '--'), (2026, ':')]:
        log = pd.read_csv(RESULTS / METHOD / f'seed_{seed}' / 'train_log.csv')
        steps = log['env_step'].diff().fillna(log['env_step'].iloc[0])
        ax.plot(log['episode'], log['reward'] / steps, ls, color=GB, lw=0.9, marker='o', ms=2,
                label=f'seed {seed}')
    ax.set_xlabel('Training episode (1 simulated year each)')
    ax.set_ylabel('Reward per step')
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()
    fig.savefig(HERE / 'fig_rltuned_learning.pdf', bbox_inches='tight', pad_inches=0.02)


def fig_week():
    tr = pd.read_csv(RESULTS / METHOD / 'seed_42' / 'eval_trace.csv')
    nb = pd.read_csv(RESULTS / 'B5_no_battery' / 'seed_0' / 'eval_trace.csv')
    start = 24 * 182                                     # a week from early July
    w = tr[(tr['step'] >= start) & (tr['step'] < start + 168)]
    wn = nb[(nb['step'] >= start) & (nb['step'] < start + 168)]
    h = (w['step'] - start).values
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 3.2), sharex=True)
    axes[0].plot(h, wn['load'].values, color=GREY, lw=0.9, label='No battery')
    axes[0].plot(h, w['load'].values, color=GB, lw=0.9, label='RL-tuned MPC')
    axes[0].axhline(0.0, color='k', lw=0.4)
    axes[0].set_ylabel('Net load (kW)')
    soc = w[[c for c in w.columns if c.startswith('soc_b')]].mean(axis=1).values
    axes[1].plot(h, soc, color=GB_DARK, lw=0.9)
    axes[1].set_ylabel('Mean SoC')
    axes[1].set_ylim(0, 1)
    axes[2].plot(h, w['pricing'].values, color=GB_DARK, lw=0.9, label='Price (left axis)')
    axes[2].set_ylabel('Price (\\$/kWh)')
    ax2 = axes[2].twinx()
    ax2.plot(h, w['carbon_intensity'].values, color=GB_MID, lw=0.9, ls='--',
             label='Carbon intensity (right axis)')
    ax2.set_ylabel('kgCO$_2$/kWh')
    axes[2].set_xlabel('Hour of the week')
    fig.align_ylabels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    # one legend at the top: the two net-load lines on the first row, price and carbon intensity on the second
    h0, l0 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    h3, l3 = ax2.get_legend_handles_labels()
    fig.legend(h0, l0, frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    fig.legend(h2 + h3, l2 + l3, frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 0.955),
               handlelength=2.2)
    fig.savefig(HERE / 'fig_rltuned_week.pdf', bbox_inches='tight', pad_inches=0.02)


def p3_scores():
    d = pd.read_csv(RESULTS / 'phase3' / 'scores.csv').drop_duplicates('name', keep='last').set_index('name')
    return d.loc[['P3_RLTUNED_M152', 'P3_RLTUNED_M152_s100', 'P3_RLTUNED_M152_s2026']]


def fig_kpis():
    """C, G, D and S of the proposed controller in each district (mean and std over three seeds)."""
    r = p3_scores()
    kpis = [('C', 'Cost $C$', GB_DARK), ('G', 'Carbon $G$', GB), ('D', 'Grid $D$', GB_MID),
            ('phase3', 'Score $S$', GB_LIGHT)]
    districts = [('train', 'Train (1–5)'), ('val', 'Validation (6–10)'), ('test', 'Test (11–17)')]
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    for j, (k, lab, col) in enumerate(kpis):
        x = [i + (j - 1.5) * 0.19 for i in range(len(districts))]
        means = [r[f'{d}_{k}'].mean() for d, _ in districts]
        stds = [r[f'{d}_{k}'].std() for d, _ in districts]
        ax.bar(x, means, width=0.18, yerr=stds, color=col, label=lab, error_kw={'lw': 0.6, 'capsize': 1})
    ax.axhline(1.0, color='k', lw=0.6, ls='--')
    ax.text(2.45, 1.005, 'no battery', fontsize=6, ha='right', va='bottom')
    ax.set_xticks(range(len(districts)))
    ax.set_xticklabels([lab for _, lab in districts])
    ax.set_ylim(0.5, 1.08)
    ax.set_ylabel('Relative to no battery')
    ax.legend(frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=1.0,
              columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(HERE / 'fig_rltuned_kpis.pdf', bbox_inches='tight', pad_inches=0.02)


def fig_daily():
    """Average daily profile of the district net load and of the mean state of charge over the year (training
    buildings, seed 42)."""
    tr = pd.read_csv(RESULTS / METHOD / 'seed_42' / 'eval_trace.csv')
    nb = pd.read_csv(RESULTS / 'B5_no_battery' / 'seed_0' / 'eval_trace.csv')
    hod = lambda df: (df['step'] - 1) % 24          # noqa: E731  (hour of the day the step ends in)
    p_nb = nb.groupby(hod(nb))['load'].mean()
    p_rl = tr.groupby(hod(tr))['load'].mean()
    soc = tr.groupby(hod(tr))[[c for c in tr.columns if c.startswith('soc_b')]].mean().mean(axis=1)
    price = tr.groupby(hod(tr))['pricing'].mean()
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    ax.axvspan(16, 21, color=GB_DARK, alpha=0.08, lw=0)
    ax.text(18.5, -4.6, 'peak tariff', fontsize=6, ha='center', color=GB_DARK)
    ax.plot(p_nb.index, p_nb.values, color=GREY, lw=1.0, label='Net load, no battery')
    ax.plot(p_rl.index, p_rl.values, color=GB, lw=1.0, label='Net load, RL-tuned MPC')
    ax.set_xlabel('Hour of the day')
    ax.set_ylabel('Mean net load (kW)')
    ax.set_xticks(range(0, 24, 3))
    ax2 = ax.twinx()
    ax2.plot(soc.index, soc.values, color=GB_DARK, lw=0.9, ls='--', label='Mean SoC (right)')
    ax2.set_ylabel('Mean SoC')
    ax2.set_ylim(0, 1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0),
              handlelength=1.5, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(HERE / 'fig_rltuned_daily.pdf', bbox_inches='tight', pad_inches=0.02)
    return price


# Phase III public leaderboard of the NeurIPS 2022 CityLearn Challenge (all 42 entries; AIcrowd):
# team, score, carbon G, cost C, grid D
LEADERBOARD = [
    ('Together', .728, .808, .639, .738), ('ambitiousengineers', .742, .803, .681, .743), ('CUFE', .753, .804, .665, .789),
    ('EVERGi_team', .756, .826, .643, .800), ('Orca', .760, .836, .659, .784), ('Greener', .764, .821, .663, .809),
    ('DivMARL', .767, .805, .653, .842), ('SCRG_UA', .789, .880, .666, .820), ('LA_gang', .796, .852, .665, .873),
    ('tymur_prorochenko', .797, .861, .693, .838), ('rafael_moran', .799, .869, .680, .847), ('Bingo', .816, .843, .704, .901),
    ('TeamEpoch', .825, .837, .706, .930), ('OneWo', .826, .850, .683, .943), ('TaCo', .831, .843, .731, .920),
    ('agnprz', .834, .855, .692, .956), ('jack-etheredge', .840, .874, .717, .930), ('anton-petrov', .840, .894, .694, .933),
    ('adrien_forbu', .844, .887, .667, .977), ('de_ig', .845, .857, .733, .946), ('Factored', .856, .867, .728, .974),
    ('Saleh.Ragheb', .869, .972, .684, .951), ('sustaingym', .870, .909, .855, .847), ('EECiTeam', .878, .862, .780, .992),
    ('ARC_RL', .878, .896, .731, 1.007), ('OneCharger', .904, .958, .798, .956), ('darlCity', .909, .916, .716, 1.093),
    ('Sneaky_Ninjas', .984, .986, .891, 1.076), ('SpiralMen', .999, 1.002, .995, 1.002), ('sumedh.gups', 1.0, 1.0, 1.0, 1.0),
    ('control-learning', 1.0, 1.0, 1.0, 1.0), ('KENTECH_AI', 1.0, 1.0, 1.0, 1.0), ('heckeRL', 1.0, 1.0, 1.0, 1.0),
    ('Benz2022', 1.0, 1.0, 1.0, 1.0), ('ROLEVT', 1.002, 1.001, 1.001, 1.005), ('vignkri', 1.007, 1.010, 1.009, 1.003),
    ('learners', 1.039, 1.051, 1.038, 1.029), ('R2D2', 1.098, 1.165, 1.065, 1.063), ('noob', 1.098, 1.165, 1.065, 1.063),
    ('HKUST_Smart_Building', 1.098, 1.165, 1.065, 1.063), ('bahae', 1.098, 1.165, 1.065, 1.063), ('CHUF', 1.098, 1.165, 1.065, 1.063),
]


def fig_leaderboard():
    """Left: overall public score of all 42 entries, sorted, with ours inserted. Right: carbon G against grid D."""
    r = p3_scores()
    ours = {k: (0.4 * r[f'train_{k}'] + 0.6 * r[f'val_{k}']).mean() for k in ['C', 'G', 'D']}
    ours['S'] = r['public'].mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.1), gridspec_kw={'width_ratios': [1.6, 1]})
    teams = sorted(LEADERBOARD, key=lambda t: t[1])
    scores = [t[1] for t in teams] + [ours['S']]
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    cols = [GB if i == len(teams) else GREY for i in order]
    a1.bar(range(len(order)), [scores[i] for i in order], color=cols, width=0.8)
    pos = order.index(len(teams))
    a1.annotate('ours (0.774)', (pos, ours['S']), xytext=(pos + 6, 0.70), fontsize=7, color=GB,
                arrowprops={'arrowstyle': '-', 'color': GB, 'lw': 0.6})
    a1.axhline(1.0, color='k', lw=0.6, ls='--')
    a1.set_ylim(0.65, 1.12)
    a1.set_xlim(-1, len(order))
    a1.set_xlabel('Rank among the 42 public entries and ours')
    a1.set_ylabel('Public score')
    g = [t[2] for t in LEADERBOARD]
    dd = [t[4] for t in LEADERBOARD]
    a2.scatter(dd, g, s=10, color=GREY, label='42 entries', zorder=2)
    a2.annotate('winner', (LEADERBOARD[0][4], LEADERBOARD[0][2]), xytext=(0.76, 0.86), fontsize=6, color=GB_DARK,
                arrowprops={'arrowstyle': '-', 'color': GB_MID, 'lw': 0.5})
    a2.scatter([ours['D']], [ours['G']], s=28, marker='*', color=GB, label='ours', zorder=3)
    a2.set_xlabel('Grid $D$')
    a2.set_ylabel('Carbon $G$')
    a2.set_xlim(0.7, 1.12)
    a2.set_ylim(0.74, 1.18)
    a2.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(HERE / 'fig_leaderboard.pdf', bbox_inches='tight', pad_inches=0.02)


def fig_forecast_value():
    """Score of the fixed-weight controller when parts of its forecasts are replaced by the true future."""
    fv = pd.read_csv(RESULTS / 'forecast' / 'forecast_value.csv').drop_duplicates(['mode', 'district'], keep='last')
    base = {'train': 0.784, 'val': 0.779}                # the same controller with its own forecasts
    modes = [('base', 'LightGBM\nforecasts'), ('perfect_solar', 'perfect\nPV'),
             ('perfect_load_h6', 'perfect load,\nnext 6 h'), ('perfect_total', 'perfect\ndistrict total'),
             ('perfect_load', 'perfect\nload')]
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    for j, (d, lab, col) in enumerate([('train', 'Train', GB), ('val', 'Validation', GB_LIGHT)]):
        vals = [base[d] if m == 'base' else
                fv[(fv['mode'] == m) & (fv['district'] == d)]['phase3'].iloc[0] for m, _ in modes]
        ax.bar([i + (j - 0.5) * 0.36 for i in range(len(modes))], vals, width=0.34, color=col, label=lab)
    ax.axhline(0.728, color='k', lw=0.6, ls=':')
    ax.text(4.6, 0.732, 'winner (public)', fontsize=6, ha='right', va='bottom')
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([lab for _, lab in modes], fontsize=6)
    ax.set_ylim(0.6, 0.82)
    ax.set_ylabel('Score $S$')
    ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(HERE / 'fig_forecast_value.pdf', bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    fig_learning()
    fig_week()
    fig_kpis()
    fig_daily()
    fig_leaderboard()
    fig_forecast_value()
    print('figures written to', HERE)
