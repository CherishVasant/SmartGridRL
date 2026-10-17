"""Builds the phase 3 notebooks from core_src.py and the per-method descriptions below."""
import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

SCRATCH = Path(__file__).parent
OUT = Path(sys.argv[1])


def cells_from_percent_script(text):
    cells = []
    for chunk in text.split('# %%')[1:]:
        if chunk.startswith(' [markdown]'):
            body = chunk[len(' [markdown]'):].strip('\n')
            md = '\n'.join(line[2:] if line.startswith('# ') else line.lstrip('#') for line in body.splitlines())
            cells.append(new_markdown_cell(md.strip()))
        else:
            cells.append(new_code_cell(chunk.strip('\n')))
    return cells


def save(nb, name):
    nb.metadata['kernelspec'] = {'name': 'smartgridrl', 'display_name': 'Python (SmartGridRL venv)', 'language': 'python'}
    nbformat.write(nb, str(OUT / name))
    print('wrote', name)


# 00_core
save(new_notebook(cells=cells_from_percent_script((SCRATCH / 'core_src.py').read_text(encoding='utf-8'))), '00_core.ipynb')

# One notebook per method
METHOD_DOCS = {
    'B1_cost_only': ('Baseline B1: Cost-Only PPO',
                     'Central PPO agent with reward `-cost`. Same as phase 2, re-trained at 200k env steps.'),
    'B3_static_matched': ('Baseline B3: Static-Matched PPO (λ = 0.41)',
                          'Central PPO agent with reward `-cost - 0.41·carbon`. λ = 0.41 equals the yearly average carbon '
                          'weight of the Dynamic agent, so it is the fair static comparison. Re-trained at 200k env steps.'),
    'B4_dynamic': ('Baseline B4: Dynamic PPO (linear φ)',
                   'Central PPO agent with reward `-cost - λ₀·Î_t·carbon`, Î_t = carbon intensity scaled to [0, 1]. '
                   'Phase 2 method, re-trained at 200k env steps, with the one-hour carbon-intensity lag fixed.'),
    'M1_dynamic_sharp': ('M1: Dynamic PPO with a sharper φ (top 25% dirtiest hours)',
                         'Central PPO agent with reward `-cost - λ₀·φ(I_t)·carbon`, where φ = 1 in the dirtiest 25% of hours '
                         'and 0 otherwise. Tests whether a sharper carbon signal separates Dynamic from Static.'),
    'M2_multikpi_fixed': ('M2: Multi-KPI reward, fixed weights',
                          'Central PPO agent. Reward = -(cost + carbon + ramping + peak increase + 0.1·battery wear), every '
                          'term divided by its RBC per-step average. All weights fixed at 1: the hand-tuned multi-KPI baseline.'),
    'M3_multikpi_adaptive': ('M3: Multi-KPI reward, adaptive learned weights (main central proposal)',
                             'Central PPO agent. Same terms as M2, but the carbon, ramping and peak weights μ are learned online: '
                             'every simulated week the weights (always summing to 3, like M2) shift toward whichever KPI is '
                             'furthest behind its target vs RBC (Dynamic Weight Average style). The carbon weight is also '
                             'scaled by the normalised carbon intensity.'),
    'M4_per_building_dynamic': ('M4: Per-building agents, Dynamic reward',
                                'One shared PPO policy controls each building separately (5 agents, parameter sharing, building '
                                'ID in the observation). Each building gets its own `-cost_b - λ₀·Î_t·carbon_b`. Comparing with '
                                'B4 isolates the effect of per-building control.'),
    'M5_per_building_adaptive_shared': ('M5: Per-building agents, multi-KPI adaptive weights, one shared set',
                                          'Per-building agents (as M4) with the multi-KPI adaptive reward. All buildings use '
                                          'the same learned μ, updated from district KPIs. Each building also gets 1/5 of the '
                                          'district ramping and peak penalty. Comparing with M6 isolates per-building weights.'),
    'M6_per_building_adaptive_own': ('M6: Per-building agents, multi-KPI adaptive weights, separate weights per building (main per-building proposal)',
                                       'Per-building agents (as M4). Every building has its own learned μ for carbon, ramping and '
                                       'peak, updated from that building\'s KPIs vs the RBC\'s same building: a different, '
                                       'data-learned reward function per building. Each building also gets 1/5 of the district '
                                       'ramping and peak penalty so they coordinate.'),
    'M7_multikpi_adaptive_norm': ('M7: Multi-KPI adaptive (M3) + observation and reward normalisation',
                                  'M3 unchanged, but the agent sees normalised observations and rewards (SB3 VecNormalize: running '
                                  'mean / std). Phase 2 and Stage A fed raw values (month 1-12, loads in kW, prices ~0.2), which '
                                  'PPO handles poorly.'),
    'M8_multikpi_fixed_norm': ('M8: Multi-KPI fixed (M2) + observation and reward normalisation',
                               'M2 unchanged, plus VecNormalize. Together with M7 this tests whether normalisation helps both '
                               'the fixed and the adaptive weights.'),
    'M9_load_squared': ('M9: Cost + carbon + squared district load',
                        'Central PPO agent. Reward = -(cost + carbon + L²), each divided by its RBC per-step mean. The squared '
                        'district load is the classic CityLearn peak / load-factor penalty: it grows quickly with high load, so it '
                        'rewards flattening the demand curve rather than only shaving the single daily peak.'),
    'M10_per_building_dynamic_district': ('M10: Per-building Dynamic (M4) + district ramp / peak penalty',
                                          'M4 was best on cost (-31%) but worst on ramping and peaks: every building charged in '
                                          'the same cheap hours. Here each building also pays 1/5 of the district ramp and peak '
                                          'terms, so the buildings are pushed to spread their charging.'),
    'M11_per_building_adaptive_district': ('M11: Per-building adaptive, grid terms at district level only',
                                           'Fix for the M5/M6 collapse (battery left idle). Each building keeps only its own cost '
                                           'and carbon terms; ramping and peaks are penalised only at district level (1/5 share '
                                           'each), with adaptive weights learned from district KPIs.'),
    'M12_sac_multikpi_fixed': ('M12: SAC with the fixed multi-KPI reward (M2)',
                               'Soft Actor-Critic instead of PPO: off-policy, reuses past experience from a replay buffer '
                               '(capped at 200k steps), one gradient step per env step. Same reward as M2, so M2 vs M12 '
                               'compares the algorithms.'),
    'M13_sac_multikpi_fixed_norm': ('M13: SAC + VecNormalize, fixed multi-KPI reward',
                                    'Combines the two biggest Round 1 wins: SAC (M12) and observation/reward normalisation '
                                    '(M8). M13 vs M12 isolates normalisation under SAC; M13 vs M8 isolates SAC under '
                                    'normalisation.'),
    'M14_sac_multikpi_adaptive_norm': ('M14: SAC + VecNormalize, adaptive learned multi-KPI weights',
                                       'As M13, but with the adaptive weights of M3/M7. M14 vs M13 tests whether learned '
                                       'weights help with the best algorithm. Note: the replay buffer keeps rewards computed '
                                       'with older weights.'),
    'M15_sac_load_squared_norm': ('M15: SAC + VecNormalize, cost + carbon + load² reward',
                                  'The M9 reward (second-best cost in Round 1, poor ramping) with SAC and normalisation. '
                                  'Tests whether SAC keeps the cost savings while fixing the grid KPIs.'),
    'M16_sac_residual_rbc_norm': ('M16: Residual SAC on top of the RBC',
                                  "Battery action = RBC action for that hour + the SAC policy's correction (clipped to the "
                                  'action bounds), with the M2 reward and normalisation. A policy that outputs 0 behaves '
                                  'exactly like the RBC, so learning starts from a sensible controller.'),
    'M17_sac_multikpi_obsnorm': ('M17: SAC + observation-only normalisation, fixed multi-KPI reward',
                                 'As M13, but only observations are normalised; rewards keep their natural (RBC-normalised) '
                                 'scale. SAC is sensitive to reward scale, so this tests whether M13 lost M12\'s cost and peak '
                                 'gains because of reward normalisation alone.'),
    'M18_sac_multikpi_batt05': ('M18: SAC, fixed multi-KPI reward with battery wear weight 0.5',
                                'As M12, with the battery-wear term (|ΔSoC|) weighted 0.5 instead of 0.1. Every battery-using '
                                'agent so far emits more carbon than no battery (round-trip losses, dirty-hour charging); '
                                'this discourages cycling that does not pay off.'),
    'M19_sac_multikpi_carbon2': ('M19: SAC, fixed multi-KPI reward with carbon weight 2',
                                 'As M12, with the carbon weight doubled (ramping and peak stay 1). Pushes carbon down '
                                 'directly, to test whether SAC can beat the no-battery carbon level.'),
    'M20_sac_multikpi_counterfactual': ('M20: SAC, multi-KPI reward measured as the gain over no battery',
                                        'As M12, but each hour\'s reward is the weighted terms of the agent minus the same '
                                        'terms of the cached no-battery run at that hour, i.e. the improvement over doing '
                                        'nothing (the reference CityLearn scores against). The subtracted part does not '
                                        'depend on the action, so the optimal policy is unchanged; it only reduces the '
                                        'variance of the critic\'s targets.'),
    'M21_sac_cf_carbon2': ('M21: SAC, gain over no battery, carbon weight 2 (M19 + M20)',
                           'Combines the two best Round 3 changes on seed 42: the carbon weight doubled (M19) and the reward '
                           'measured as the gain over the no-battery run (M20). Tests whether the two gains stack. Uses the '
                           'full-day peak term (see 00_core section 2), like all Round 4 methods.'),
    'M22_sac_cf_carbon3': ('M22: SAC, gain over no battery, carbon weight 3',
                           'As M21 with the carbon weight raised to 3, to find where a stronger carbon push stops helping.'),
    'M23_sac_cf_carbon2_ramp2': ('M23: SAC, gain over no battery, carbon 2 and ramping 2',
                                 'As M21 with the ramping weight doubled too. Ramping is the one KPI every SAC run still '
                                 'leaves above the RBC.'),
    'M24_sac_cf_peakfix': ('M24: SAC, gain over no battery, full-day peak term (M20 fixed)',
                           'M20 with only the peak term fixed: the day\'s first hour now counts its own load, so a spike '
                           'there is no longer free (in M20 the counterfactual reward even paid for it). M24 vs M20 isolates '
                           'the fix.'),
    'M25_sac_official_cf': ('M25: SAC with the official district-level cost and carbon terms',
                            'As M24, but cost and carbon are the CityLearn Challenge 2022 definitions: clipped at 0 on the '
                            'district net load, so exports earn nothing and one building\'s export offsets another\'s import. '
                            'Earlier methods used CityLearn\'s per-building values, which pay for exports and give no carbon '
                            'credit for offsetting neighbours.'),
    'M26_sac_score_aligned': ('M26: SAC with a reward aligned to the challenge score',
                              'Official cost/carbon terms, every term divided by its no-battery per-step mean, weights shaped '
                              'like the challenge score (C + G + (R + 1-L)/2)/3: cost 1, carbon 1, ramping 0.5, full-day peak '
                              '0.5 (proxy for 1 - load factor), measured as the gain over no battery. The yearly return is '
                              'then (minus) the score, up to the peak proxy.'),
    'M27_sac_score_aligned_carbon2': ('M27: SAC, challenge-score-aligned reward with carbon weight 2',
                                      'As M26 with the carbon weight doubled: optimises the challenge score with extra '
                                      'emphasis on carbon.'),
    'M28_sac_phase1_aligned': ('M28: SAC, reward aligned to the Phase I score (cost + carbon)',
                               'As M26 with ramping and peak weights 0.2: aimed at the Phase I leaderboard score mean(C, G) '
                               'on the 5 training buildings (our dataset), keeping a light grid penalty.'),
    'M29_resid_lp_phase1': ('M29: Residual SAC around the Phase I LP plan',
                            'Fusion of planning and RL. The battery action is the perfect-foresight LP plan\'s action for that '
                            'hour (results/O3_lp_phase1_eff087, from tools/lp_oracle.py) plus a SAC correction scaled to ±25%, '
                            'trained with the Phase I-aligned reward (M28). RL can fix what the LP\'s linear battery model gets '
                            'wrong (efficiency curve, power taper). A correction of 0 reproduces the plan exactly. Requires the '
                            'O3 plan to exist (run tools/lp_oracle.py 0 O3_lp_phase1_eff087 1 0.87).'),
    'M30_resid_lp_grid': ('M30: Residual SAC around the grid-aware LP plan',
                          'As M29, around the LP plan that also minimises ramping and monthly peaks (results/O2_lp_oracle_grid; '
                          'tools/lp_oracle.py 1 O2_lp_oracle_grid), with the challenge-score-aligned reward (M26).'),
    'M31_resid_lp_grid_carbon2': ('M31: Residual SAC around the grid-aware LP plan, carbon weight 2',
                                  'As M30 with the carbon weight doubled in the reward, to push carbon further below the plan.'),
    'M32_resid_lp_grid_full': ('M32: Residual SAC around the grid-aware LP plan, full-range correction',
                               'As M30 but the SAC correction can use the full action range (scale 1.0 instead of 0.25).'),
    'M33_bc_lp_grid': ('M33: Behaviour clone of the grid-aware LP plan',
                       'The SAC actor is fitted by supervised learning to the perfect-foresight LP plan\'s actions '
                       '(results/O2_lp_oracle_grid) on the observations along that plan\'s trajectory. Unlike the residual '
                       'methods it needs only the current observation, not the plan, so it is a real controller. No RL '
                       'update (1,000 env steps, SAC learning starts after 1,000).'),
    'M34_bc_lp_grid_sac': ('M34: Behaviour clone of the grid-aware LP plan + SAC fine-tuning',
                           'M33\'s cloned actor as the starting point of SAC with the challenge-score-aligned reward (M26), '
                           'with a lower initial entropy weight (0.05) so exploration does not wash out the clone.'),
    'M35_bc_lp_phase1': ('M35: Behaviour clone of the Phase I LP plan',
                         'As M33, cloning the Phase I plan (results/O3_lp_phase1_eff087).'),
    'M36_bc_lp_phase1_sac': ('M36: Behaviour clone of the Phase I LP plan + SAC fine-tuning',
                             'As M34, from the Phase I clone, with the Phase I-aligned reward (M28).'),
    'M37_bc_grid_big_dagger': ('M37: Clone of the grid LP plan with a 256x256 network and DAgger',
                               'As M33 with a bigger actor (256x256) and 3 DAgger iterations: the clone is rolled out, every '
                               'state it visits is labelled with the action that moves each battery to the plan\'s state of '
                               'charge for that hour, and the actor is refitted on all data so it learns to recover from its '
                               'own drift.'),
    'M38_bc_grid_big': ('M38: Clone of the grid LP plan with a 256x256 network',
                        'As M33 with a bigger actor only (isolates the network size; compare M37).'),
    'M39_bc_grid_dagger': ('M39: Clone of the grid LP plan with DAgger',
                           'As M33 with 3 DAgger iterations only (isolates DAgger; compare M37).'),
    'M40_bc_grid2_big_dagger': ('M40: Clone of a smoother LP plan, 256x256 + DAgger',
                                'As M37, cloning a plan with twice the grid weight (results/O8_lp_grid2; tools/lp_oracle.py '
                                '2 O8_lp_grid2), since the smoother plan O2 cloned better than the Phase I plan O3.'),
    'M41_bc_grid_big_gentle': ('M41: Big clone of the grid LP plan with gentle DAgger',
                               'As M37, but the DAgger expert is the plan\'s action for that hour plus 30% of the state-of-'
                               'charge error per hour (bc_dagger_gain 0.3) instead of a one-hour catch-up, which made the '
                               'Round 8 clones jumpy (worse ramping and peaks).'),
    'M42_bc_grid2_big_gentle': ('M42: Big clone of the smoother LP plan with gentle DAgger',
                                'As M41, cloning the smoother plan O8 (grid weight 2).'),
    'M43_bc_grid_big_sac_c2': ('M43: Big clone of the grid LP plan + SAC fine-tuning (score reward, carbon 2)',
                               'The 256x256 clone of O2 (as M38) as the start of SAC with the best RL reward so far (M27: '
                               'challenge-score-aligned, carbon weight 2), initial entropy weight 0.05.'),
    'M44_bc_grid2_big_sac_c2': ('M44: Big clone of the smoother LP plan + SAC fine-tuning (score reward, carbon 2)',
                                'As M43, starting from the clone of the smoother plan O8.'),
    'M45_bc_grid_gentle': ('M45: Clone of the grid LP plan (64x64) with gentle DAgger',
                           'As M41 with the default 64x64 actor: the 256x256 clone alone (M38) did worse than the 64x64 one '
                           '(M33) in closed loop despite fitting the plan 10x more closely.'),
    'M46_bc_grid_sac_c2': ('M46: Clone of the grid LP plan (64x64) + SAC fine-tuning (score reward, carbon 2)',
                           'As M43 with the default 64x64 actor (the M33 clone as the starting point).'),
    'M47_bc_grid_members': ('M47: Extra seeds of the M33 clone, as ensemble members',
                            'Identical to M33 (clone of the grid LP plan), trained on seeds 1-7. The 3-seed ensemble of M33 '
                            'beat every single seed, so these are extra members for tools/ensemble_eval.py '
                            '(e.g. `python tools/ensemble_eval.py E10_M33 M33_bc_lp_grid M47_bc_grid_members:1,2,3,4,5,6,7`).'),
    'M48_bc_big_gentle_members': ('M48: Extra seeds of the M41 clone, as ensemble members',
                                  'Identical to M41 (256x256 clone of the grid LP plan + gentle DAgger, the best single model '
                                  'in Round 9), trained on seeds 1-5 as members for tools/ensemble_eval.py.'),
    'M49_bc_big_gentle_dagger6': ('M49: M41 with 6 DAgger iterations',
                                  'As M41 with 6 gentle-DAgger iterations instead of 3: does more recovery data help further?'),
    'M50_bc_big_gentle_gain05': ('M50: M41 with DAgger gain 0.5',
                                 'As M41 with the expert correcting 50% of the state-of-charge error per hour instead of 30%.'),
    'M51_bc_deep_gentle': ('M51: Deep clone (3 x 256) + gentle DAgger',
                           'As M41 with a third hidden layer, now that DAgger counters the overfitting that hurt the plain '
                           'big clone (M38).'),
    'M52_bc_deep_members': ('M52: Extra seeds of the M51 deep clone, as ensemble members',
                            'Identical to M51 (3 x 256 clone of the grid LP plan + gentle DAgger, the best single model in '
                            'Round 11), trained on seeds 1-5 as members for tools/ensemble_eval.py.'),
    'M53_bc_deeper_gentle': ('M53: Deeper clone (4 x 256) + gentle DAgger', 'As M51 with a fourth hidden layer.'),
    'M54_bc_wide_gentle': ('M54: Wide deep clone (3 x 512) + gentle DAgger', 'As M51 with 512 units per layer.'),
    'M55_bc_deep_gentle_o8': ('M55: Deep clone of the smoother LP plan + gentle DAgger',
                              'As M51, cloning the smoother plan O8 (grid weight 2), which cloned well in M42.'),
    'M56_bc_deep_dagger6': ('M56: Deep clone + gentle DAgger, 6 iterations', 'As M51 with 6 DAgger iterations instead of 3.'),
    'M57_bc_deep_o6': ('M57: Deep clone of the LP plan with grid weight 0.5 (O6)',
                       'As M51, cloning O6 (Phase I 0.648, Score 0.627; tools/lp_oracle.py 0.5 O6_lp_grid05 1 0.87) instead '
                       'of O2 (Phase I 0.658): a plan closer to the published top-3 Phase I scores.'),
    'M58_bc_deep_o3': ('M58: Deep clone of the Phase I LP plan (O3)',
                       'As M51, cloning the Phase I-optimal plan O3 (Phase I 0.629), which the small plain clone (M35) could '
                       'not follow (0.797).'),
    'M59_bc_deep_o9': ('M59: Deep clone of the LP plan with grid weight 0.25 (O9)',
                       'As M51, cloning O9 (tools/lp_oracle.py 0.25 O9_lp_grid025), between O3 and O6.'),
    'M60_bc_deep_o6_dagger6': ('M60: Deep clone of O6 with 6 DAgger iterations', 'As M57 with 6 DAgger iterations.'),
    'M61_bc_4x512': ('M61: Clone 4 x 512 + gentle DAgger', 'As M54 (3 x 512, the best single model in Round 12) with a fourth layer.'),
    'M62_bc_3x1024': ('M62: Clone 3 x 1024 + gentle DAgger', 'As M54 with 1024 units per layer.'),
    'M63_bc_3x512_dagger6': ('M63: Clone 3 x 512 + gentle DAgger, 6 iterations',
                             'As M54 with 6 DAgger iterations (6 beat 3 for the 2 x 256 clone, M49 vs M41).'),
    'M64_bc_3x512_o9': ('M64: Clone 3 x 512 of the LP plan O9',
                        'As M54, cloning O9 (grid weight 0.25; Phase I 0.639, Score 0.633) instead of O2 (0.658 / 0.624).'),
    'M65_bc_3x512_o3': ('M65: Clone 3 x 512 of the Phase I LP plan O3',
                        'As M54, cloning the Phase I-optimal plan O3 (Phase I 0.629): the 3 x 256 clone of O3 (M58) reached '
                        'Phase I 0.658, close to the published #3 (0.651).'),
    'M66_bc_4x512_o3': ('M66: Clone 4 x 512 of the Phase I LP plan O3', 'As M65 with a fourth layer.'),
    'M67_bc_3x512_o3_dagger6': ('M67: Clone 3 x 512 of the Phase I LP plan O3, 6 DAgger iterations',
                                'As M65 with 6 DAgger iterations.'),
}
SEEDS_OVERRIDE = {'M47_bc_grid_members': '[1, 2, 3, 4, 5, 6, 7]   # ensemble members (see the description)',
                  'M48_bc_big_gentle_members': '[1, 2, 3, 4, 5]   # ensemble members (see the description)',
                  'M52_bc_deep_members': '[1, 2, 3, 4, 5]   # ensemble members (see the description)'}

for method, (title, desc) in METHOD_DOCS.items():
    cells = [
        new_markdown_cell(
            f"# {title}\n\n{desc}\n\n"
            "**How to run:** Run All. Results go to `02_multi_kpi_adaptive_reward/results/" + method + "/seed_<seed>/`.\n\n"
            "- A seed with `summary.json` is finished and skipped.\n"
            "- A seed that stopped mid-training resumes from its last checkpoint (saved every 20k env steps).\n"
            "- To add seeds later, append them to `SEEDS` below and Run All again."),
        new_code_cell(
            f"METHOD = '{method}'\n"
            + (f"SEEDS = {SEEDS_OVERRIDE[method]}" if method in SEEDS_OVERRIDE
               else "SEEDS = [42, 100, 2026]   # Stage B: add 7, 1234 here and Run All again")),
        new_code_cell("%run ./00_core.ipynb"),
        new_code_cell(
            "print(METHODS[METHOD])\n"
            "results = run_method(METHOD, SEEDS)\n"
            "results"),
        new_markdown_cell("## Summary over seeds (ratios to RBC, lower is better)"),
        new_code_cell(
            "ratio_cols = ['score', 'cost_ratio', 'carbon_ratio', 'ramping_ratio', 'daily_peak_ratio', 'annual_peak_ratio',\n"
            "              'one_minus_load_factor_ratio', 'carbon_per_kwh_imported_ratio']\n"
            "results[ratio_cols].agg(['mean', 'std']).T.round(4)"),
    ]
    save(new_notebook(cells=cells), f'{method}.ipynb')

# Comparison notebook
compare = [
    new_markdown_cell(
        "# 99 Compare results\n\n"
        "Collects every finished `results/<METHOD>/seed_<seed>/summary.json` (whatever has been run so far, on any "
        "machine: copy friends' `results/<METHOD>/` folders into `results/` first).\n\n"
        "1. Per-method table: mean ± 95% CI of every KPI ratio vs RBC (lower is better) and the ranking score\n"
        "2. Stage A ranking of the new methods M1–M6 and the top 3\n"
        "3. Paired differences vs Static-Matched (B3) on shared seeds\n"
        "4. Plots: cost vs carbon ratio, and KPI ratios per method"),
    new_code_cell("%run ./00_core.ipynb\nfrom scipy import stats\nimport matplotlib.pyplot as plt"),
    new_markdown_cell(
        "## 0. No-battery reference (B5)\n\n"
        "Batteries never used (action 0 every hour). CityLearn's own KPIs are normalised against this "
        "\"no control\" case, and in seed-42 results it beat the RBC, so every method is also scored against it below. "
        "Deterministic, about 1 minute, cached in `results/B5_no_battery/`."),
    new_code_cell(
        "LABELS = {m: cfg['label'] for m, cfg in METHODS.items()}\n"
        "LABELS['B5_no_battery'] = 'No battery (do nothing)'\n\n"
        "def run_no_battery():\n"
        "    \"\"\"Batteries never charged or discharged; returns (and caches) its summary.\"\"\"\n"
        "    folder = RESULTS_DIR / 'B5_no_battery' / 'seed_0'\n"
        "    if (folder / 'summary.json').exists():\n"
        "        with open(folder / 'summary.json') as f:\n"
        "            return json.load(f)\n"
        "    folder.mkdir(parents=True, exist_ok=True)\n"
        "    rbc = load_rbc()\n"
        "    env = CityLearnEnv(schema=DATASET_NAME, central_agent=False)\n"
        "    env.reset()\n"
        "    n = len(env.buildings)\n"
        "    tracker = new_tracker(n)\n"
        "    rows = []\n"
        "    while True:\n"
        "        _, _, done, _ = env.step([[0.0]] * n)\n"
        "        rows.append(trace_row(env, env.time_step, step_terms(env, tracker), [0.0] * n))\n"
        "        if done:\n"
        "            break\n"
        "    trace = pd.DataFrame(rows)\n"
        "    kpis = compute_kpis(trace, env.buildings)\n"
        "    with open(RESULTS_DIR / RBC_FOLDER / 'summary.json') as f:\n"
        "        rbc_kpis = json.load(f)['kpis']\n"
        "    summary = {'method': 'B5_no_battery', 'label': LABELS['B5_no_battery'], 'seed': 0, 'kpis': kpis,\n"
        "               'ratios_vs_rbc': kpi_ratios(kpis, rbc['trace'], rbc_kpis)}\n"
        "    trace.to_csv(folder / 'eval_trace.csv', index=False)\n"
        "    with open(folder / 'summary.json', 'w') as f:\n"
        "        json.dump(summary, f, indent=2)\n"
        "    return summary\n\n"
        "no_battery = run_no_battery()\n"
        "{k: round(v, 4) for k, v in no_battery['ratios_vs_rbc'].items()}"),
    new_markdown_cell(
        "## Collect all finished runs\n\n"
        "Adds `score_vs_nb` (the same score, but every KPI divided by the no-battery value instead of RBC) and a "
        "collapse check: `soc_std` is the battery state-of-charge spread over the year, averaged over buildings. "
        "Below 0.05 means the policy hardly uses the battery (`battery_idle`)."),
    new_code_cell(
        "def soc_spread(method, seed):\n"
        "    trace = pd.read_csv(RESULTS_DIR / method / f'seed_{seed}' / 'eval_trace.csv')\n"
        "    return float(trace[[c for c in trace.columns if c.startswith('soc_b')]].std().mean())\n\n"
        "rows = []\n"
        "for method in list(METHODS) + ['B5_no_battery']:\n"
        "    for path in sorted((RESULTS_DIR / method).glob('seed_*/summary.json')):\n"
        "        with open(path) as f:\n"
        "            s = json.load(f)\n"
        "        row = summary_row(s)\n"
        "        row['score_vs_nb'] = float(np.mean([s['kpis'][k] / no_battery['kpis'][k] for k in SCORE_KPIS]))\n"
        "        row['soc_std'] = soc_spread(method, s['seed'])\n"
        "        rows.append(row)\n"
        "all_runs = pd.DataFrame(rows)\n"
        "all_runs['battery_idle'] = (all_runs['soc_std'] < 0.05) & (all_runs['method'] != 'B5_no_battery')\n"
        "print(f\"{len(all_runs)} finished runs\")\n"
        "all_runs[['method', 'seed', 'score', 'score_vs_nb', 'cost_ratio', 'carbon_ratio', 'soc_std', 'battery_idle']].round(4)"),
    new_markdown_cell("## 1. Mean ± 95% CI per method"),
    new_code_cell(
        "KPI_COLS = ['score', 'score_vs_nb'] + [f'{k}_ratio' for k in SCORE_KPIS] + ['carbon_per_kwh_imported_ratio', 'battery_cycles_ratio',\n"
        "                                                          'electricity_consumption_ratio']\n\n"
        "def mean_ci(values):\n"
        "    \"\"\"Mean and 95% CI half-width (t distribution); half-width is NaN with a single seed.\"\"\"\n"
        "    values = np.asarray(values, dtype=float)\n"
        "    if len(values) < 2:\n"
        "        return values.mean(), float('nan')\n"
        "    return values.mean(), stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))\n\n"
        "def method_table(df):\n"
        "    out = []\n"
        "    for method, g in df.groupby('method', sort=False):\n"
        "        row = {'method': method, 'label': LABELS[method], 'n_seeds': len(g), 'idle_seeds': int(g['battery_idle'].sum())}\n"
        "        for c in KPI_COLS:\n"
        "            m, h = mean_ci(g[c])\n"
        "            row[c] = f\"{m:.4f} ± {h:.4f}\" if not np.isnan(h) else f\"{m:.4f}\"\n"
        "        out.append(row)\n"
        "    return pd.DataFrame(out).set_index('method')\n\n"
        "table = method_table(all_runs)\n"
        "table.to_csv(RESULTS_DIR / 'comparison_table.csv')\n"
        "table"),
    new_markdown_cell(
        "## 2. Stage A ranking of the new methods\n\n"
        "Ranked by mean score over the seeds each method has (lower = better on average across cost, carbon, ramping, "
        "daily peak, annual peak and 1 − load factor). Compare only methods with the same seeds. `idle_seeds` > 0 flags seeds where the policy hardly used the battery."),
    new_code_cell(
        "new = all_runs[all_runs['method'].str.startswith('M')]\n"
        "ranking = new.groupby('method').agg(n_seeds=('seed', 'count'), score=('score', 'mean'), score_std=('score', 'std'),\n"
        "                                   score_vs_nb=('score_vs_nb', 'mean'), cost=('cost_ratio', 'mean'),\n"
        "                                   carbon=('carbon_ratio', 'mean'), idle_seeds=('battery_idle', 'sum')).sort_values('score')\n"
        "print('Top 3:', list(ranking.index[:3]))\n"
        "ranking.round(4)"),
    new_markdown_cell(
        "## 3. Paired differences vs Static-Matched (B3)\n\n"
        "For each method, the per-seed difference (method − B3) on seeds both have. Negative = better than B3. "
        "The CI excludes 0 only if the difference is consistent across seeds."),
    new_code_cell(
        "base = all_runs[all_runs['method'] == 'B3_static_matched'].set_index('seed')\n"
        "paired = []\n"
        "for method, g in all_runs.groupby('method', sort=False):\n"
        "    if method == 'B3_static_matched' or base.empty:\n"
        "        continue\n"
        "    g = g.set_index('seed')\n"
        "    common = g.index.intersection(base.index)\n"
        "    if len(common) == 0:\n"
        "        continue\n"
        "    row = {'method': method, 'seeds': list(common)}\n"
        "    for c in ['score', 'cost_ratio', 'carbon_ratio', 'ramping_ratio', 'daily_peak_ratio']:\n"
        "        m, h = mean_ci(g.loc[common, c] - base.loc[common, c])\n"
        "        row[c] = f\"{m:+.4f} ± {h:.4f}\" if not np.isnan(h) else f\"{m:+.4f}\"\n"
        "    paired.append(row)\n"
        "pd.DataFrame(paired).set_index('method') if paired else 'No B3 runs yet'"),
    new_markdown_cell("## 4. Plots"),
    new_code_cell(
        "fig, ax = plt.subplots(figsize=(8, 6))\n"
        "for method, g in all_runs.groupby('method', sort=False):\n"
        "    _, hx = mean_ci(g['cost_ratio']); _, hy = mean_ci(g['carbon_ratio'])\n"
        "    ax.errorbar(g['cost_ratio'].mean(), g['carbon_ratio'].mean(), xerr=0 if np.isnan(hx) else hx,\n"
        "                yerr=0 if np.isnan(hy) else hy, fmt='o', capsize=4, label=LABELS[method])\n"
        "ax.axvline(1.0, color='gray', ls='--', lw=0.8); ax.axhline(1.0, color='gray', ls='--', lw=0.8)\n"
        "ax.set_xlabel('Cost ratio vs RBC (lower is better)'); ax.set_ylabel('Carbon ratio vs RBC (lower is better)')\n"
        "ax.set_title('Cost vs carbon (mean ± 95% CI over seeds)'); ax.legend(fontsize=8)\n"
        "fig.tight_layout(); fig.savefig(PLOTS_DIR / 'cost_vs_carbon.png', dpi=200); plt.show()"),
    new_code_cell(
        "means = all_runs.groupby('method', sort=False)[[f'{k}_ratio' for k in SCORE_KPIS]].mean()\n"
        "means.index = [LABELS[m] for m in means.index]\n"
        "# matplotlib directly: pandas' .plot needs pkg_resources, which this venv does not have\n"
        "fig, ax = plt.subplots(figsize=(12, 5))\n"
        "width = 0.85 / len(means)\n"
        "for i, (label, vals) in enumerate(means.iterrows()):\n"
        "    ax.bar(np.arange(len(SCORE_KPIS)) + i * width, vals.values, width, label=label)\n"
        "ax.axhline(1.0, color='black', lw=0.8); ax.set_ylabel('Ratio vs RBC (lower is better)')\n"
        "ax.set_xticks(np.arange(len(SCORE_KPIS)) + 0.85 / 2 - width / 2)\n"
        "ax.set_xticklabels([k.replace('_', ' ') for k in SCORE_KPIS]); ax.legend(fontsize=7, ncol=3)\n"
        "ax.set_ylim(min(0.8, means.values.min() - 0.02), max(1.2, means.values.max() + 0.02))\n"
        "plt.tight_layout(); plt.savefig(PLOTS_DIR / 'kpi_ratios.png', dpi=200); plt.show()"),
]
save(new_notebook(cells=compare), '99_compare_results.ipynb')
