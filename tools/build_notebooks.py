"""Builds the phase 3 notebooks from core_src.py and the per-method descriptions below."""
import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

SCRATCH = Path(__file__).parent
OUT = Path(sys.argv[1])   # project folder or a scratch copy; notebooks go to OUT/code[/Round_XX]
sys.path.insert(0, str(SCRATCH))
from rounds import round_folder   # noqa: E402


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
    """00_core / 99_compare_results go to OUT/code; method notebooks to OUT/code/Round_XX (see tools/rounds.py)."""
    nb.metadata['kernelspec'] = {'name': 'smartgridrl', 'display_name': 'Python (SmartGridRL venv)', 'language': 'python'}
    folder = OUT / 'code' if name.startswith(('00_', '99_')) else OUT / 'code' / round_folder(name[:-6]).name
    folder.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(folder / name))
    print('wrote', (folder / name).relative_to(OUT))


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
    'M68_bc_4x512_o3_dagger6': ('M68: Clone 4 x 512 of the Phase I LP plan O3, 6 DAgger iterations',
                                'Combines the three levers that each helped: a 4 x 512 actor, 6 gentle-DAgger iterations, and '
                                'the Phase I-optimal plan O3. Aimed at the published Phase I top 3.'),
    'M69_bc_4x512_o9_dagger6': ('M69: Clone 4 x 512 of the LP plan O9, 6 DAgger iterations',
                                'As M68, cloning the balanced plan O9 (grid weight 0.25).'),
    'M70_bc_4x512_o2_dagger6': ('M70: Clone 4 x 512 of the grid LP plan O2, 6 DAgger iterations',
                                'As M68, cloning the grid-aware plan O2 (best for the full challenge score).'),
    'M71_bc_3x512_o3_dagger10': ('M71: Clone 3 x 512 of the Phase I LP plan O3, 10 DAgger iterations',
                                 'As M65 with 10 DAgger iterations: does more DAgger keep helping?'),
    'M72_bc_3x512_o10_dagger6': ('M72: Clone 3 x 512 of the LP plan O10, 6 DAgger iterations',
                                 'Clones O10 (grid weight 0.1; tools/lp_oracle.py 0.1 O10_lp_grid01 1 0.9): almost Phase I-optimal '
                                 '(0.634 vs O3\'s 0.629) but much smoother (D 0.666 vs 1.096). The 3 x 512 clone of O9 (M64) '
                                 'already reached Phase I 0.649.'),
    'M73_bc_3x512_o11_dagger6': ('M73: Clone 3 x 512 of the carbon-weighted LP plan O11, 6 DAgger iterations',
                                 'Clones O11 (grid weight 0.25, carbon weight 2; tools/lp_oracle.py 0.25 O11_lp_grid025_carbon2 '
                                 '2 0.9; carbon G 0.701) for the carbon objective.'),
    'M74_bc_4x512_o10_dagger6': ('M74: Clone 4 x 512 of the LP plan O10, 6 DAgger iterations', 'As M72 with a 4 x 512 actor.'),
    'M75_bc_3x512_o9_dagger6': ('M75: Clone 3 x 512 of the LP plan O9, 6 DAgger iterations',
                                'As M64 (3 x 512 clone of O9, 3 iterations) with 6 DAgger iterations.'),
    'M76_bc_5x512_o3_dagger6': ('M76: Clone 5 x 512 of the Phase I LP plan O3, 6 DAgger iterations',
                                'The 4 x 512 clone of O3 (M66) is a robust published #3 (Phase I 0.642 / 0.639 on two seeds), '
                                '0.005 behind the winner Together (0.635) and about 0.011 above its own plan O3 (0.629). This '
                                'round pushes the precision of the closed-loop imitation; M76 adds a fifth layer.'),
    'M77_bc_4x1024_o3_dagger6': ('M77: Clone 4 x 1024 of the Phase I LP plan O3, 6 DAgger iterations',
                                 'As M76 with 4 layers of 1024 units.'),
    'M78_bc_4x512_o3_dagger10': ('M78: Clone 4 x 512 of the Phase I LP plan O3, 10 DAgger iterations',
                                 'As M66 with 10 DAgger iterations.'),
    'M79_bc_4x512_o3_gain02': ('M79: Clone 4 x 512 of the Phase I LP plan O3, DAgger gain 0.2',
                               'As M66 with 6 DAgger iterations and a gentler expert correction (20% of the SoC error per hour).'),
    'M80_bc_o9_noise01': ('M80: Clone of O9 with input noise 0.1 (generalisation)',
                          'As M75 (3 x 512 clone of O9, 6 DAgger iterations), with Gaussian noise (std 0.1 in normalised '
                          'units) added to the observations during cloning, so the policy relies less on exact input values. '
                          'Aimed at the held-out phase_2 buildings (tools/transfer_eval.py), where the smooth clones keep '
                          'cost and carbon savings but lose the grid benefit.'),
    'M81_bc_o9_noise03': ('M81: Clone of O9 with input noise 0.3 (generalisation)', 'As M80 with noise std 0.3.'),
    'M82_bc_o9_small_noise01': ('M82: Small clone (2 x 256) of O9 with input noise 0.1 (generalisation)',
                                'As M80 with a smaller 2 x 256 actor.'),
    'M83_bc_3x512_o3_dagger15': ('M83: Clone 3 x 512 of the Phase I LP plan O3, 15 DAgger iterations',
                                 'DAgger iterations kept improving the clone of O3 (seed 42 Phase I: 3 iterations 0.650, 6 '
                                 '0.643, 10 0.637; the published winner Together scored 0.635). M83 uses 15.'),
    'M84_bc_3x512_o3_dagger20': ('M84: Clone 3 x 512 of the Phase I LP plan O3, 20 DAgger iterations', 'As M83 with 20 iterations.'),
    'M85_bc_4x512_o3_dagger15': ('M85: Clone 4 x 512 of the Phase I LP plan O3, 15 DAgger iterations', 'As M83 with a 4 x 512 actor.'),
    'M86_bc_3x512_o10_dagger15': ('M86: Clone 3 x 512 of the LP plan O10, 15 DAgger iterations',
                                  'As M83, cloning the balanced plan O10 (grid weight 0.1).'),
    'M87_bc_o9_permute2': ('M87: Clone of O9 with building-order augmentation x2 (generalisation)',
                           'As M75 (3 x 512 clone of O9, 6 DAgger iterations), with 2 extra copies of every cloning sample '
                           'in which the 5 per-building observation blocks and actions are shuffled together. The LP plan '
                           'treats buildings symmetrically, so a shuffled district\'s optimal plan is the shuffled plan; the '
                           'clone can no longer tie a strategy to a building slot, which should transfer better to the '
                           'held-out phase_2 buildings.'),
    'M88_bc_4x512_o9_permute2': ('M88: 4 x 512 clone of O9 with building-order augmentation x2 (generalisation)',
                                 'As M69 (4 x 512 clone of O9, 6 DAgger iterations) with the augmentation of M87.'),
    'M89_bc_o9_permute4': ('M89: Clone of O9 with building-order augmentation x4 (generalisation)',
                           'As M87 with 4 shuffled copies.'),
    'M90_sac_c2_256': ('M90: SAC, 256 x 256 networks',
        "Genuine reinforcement learning, deployable in a real building: the policy learns from reward only and sees just the present, the past and CityLearn's own forecasts (no perfect-foresight plan, no imitation). Base: M27 (SAC, score-aligned reward with carbon weight 2). Explored on seed 42 first; more seeds only if it helps. As M27 with 256 x 256 actor and critics (M27: 64 x 64)."),
    'M91_tqc_c2_256': ('M91: TQC, 256 x 256 networks',
        "Genuine reinforcement learning, deployable in a real building: the policy learns from reward only and sees just the present, the past and CityLearn's own forecasts (no perfect-foresight plan, no imitation). Base: M27 (SAC, score-aligned reward with carbon weight 2). Explored on seed 42 first; more seeds only if it helps. Truncated Quantile Critics (sb3-contrib): 2 distributional critics, the top 2 quantiles per critic dropped to curb Q over-estimation. Reward and settings as M27, 256 x 256."),
    'M92_crossq_c2_256': ('M92: CrossQ, 256 x 256 networks',
        "Genuine reinforcement learning, deployable in a real building: the policy learns from reward only and sees just the present, the past and CityLearn's own forecasts (no perfect-foresight plan, no imitation). Base: M27 (SAC, score-aligned reward with carbon weight 2). Explored on seed 42 first; more seeds only if it helps. CrossQ (sb3-contrib): batch-normalised critics, no target networks, learning rate 1e-3; very sample-efficient in its paper. Reward as M27, 256 x 256."),
    'M93_sac_c2_history': ('M93: SAC + history features',
        "Genuine reinforcement learning, deployable in a real building: the policy learns from reward only and sees just the present, the past and CityLearn's own forecasts (no perfect-foresight plan, no imitation). Base: M27 (SAC, score-aligned reward with carbon weight 2). Explored on seed 42 first; more seeds only if it helps. As M27, with 14 extra observation features computed only from the past: cyclic hour and month, and a persistence forecast from yesterday (carbon intensity and district non-shiftable load minus solar 24 - h hours ago, for the next h = 1, 2, 3, 6, 12 hours). CityLearn gives no carbon-intensity or load forecast, and carbon follows a daily pattern."),
    'M94_sac_c2_nstep3': ('M94: SAC with 3-step returns',
        "Genuine reinforcement learning, deployable in a real building: the policy learns from reward only and sees just the present, the past and CityLearn's own forecasts (no perfect-foresight plan, no imitation). Base: M27 (SAC, score-aligned reward with carbon weight 2). Explored on seed 42 first; more seeds only if it helps. As M27 with 3-step TD targets, so the reward of a discharge reaches the decision to charge hours earlier faster."),
    'M95_sac_c2_gamma095': ('M95: SAC, gamma 0.95',
        "Genuine reinforcement learning, deployable in a real building: the policy learns from reward only and sees just the present, the past and CityLearn's own forecasts (no perfect-foresight plan, no imitation). Base: M27 (SAC, score-aligned reward with carbon weight 2). Explored on seed 42 first; more seeds only if it helps. As M27 with discount 0.95 (horizon about 20 hours, one daily cycle) instead of 0.99 (about 100 hours): lower-variance targets."),
    'M96_sac_c2_256_history': ('M96: SAC, 256 x 256 + history features',
        "Genuine reinforcement learning, deployable in a real building: the policy learns from reward only and sees just the present, the past and CityLearn's own forecasts (no perfect-foresight plan, no imitation). Base: M27 (SAC, score-aligned reward with carbon weight 2). Explored on seed 42 first; more seeds only if it helps. M90 and M93 combined."),
    'M97_tqc_c2_256_history': ('M97: TQC, 256 x 256 + history features',
        "Genuine reinforcement learning, deployable in a real building: the policy learns from reward only and sees just the present, the past and CityLearn's own forecasts (no perfect-foresight plan, no imitation). Base: M27 (SAC, score-aligned reward with carbon weight 2). Explored on seed 42 first; more seeds only if it helps. M91 and M93 combined."),
    'M98_sac_c2_per_building': ('M98: SAC, one shared policy per building',
        "Genuine reinforcement learning, deployable in a real building: the policy learns from reward only and sees just the present, the past and CityLearn's own forecasts (no perfect-foresight plan, no imitation). Base: M27 (SAC, score-aligned reward with carbon weight 2). Explored on seed 42 first; more seeds only if it helps. Decentralised: each building runs the same policy on its own observation plus a one-hot building id (parameter sharing, 5 samples per step, 1M-sample buffer). Can run on any number of buildings."),
    'M99_sac_c2_utd2': ('M99: SAC, 2 gradient updates per env step',
        "Genuine reinforcement learning, deployable in a real building: the policy learns from reward only and sees just the present, the past and CityLearn's own forecasts (no perfect-foresight plan, no imitation). Base: M27 (SAC, score-aligned reward with carbon weight 2). Explored on seed 42 first; more seeds only if it helps. As M27 with update-to-data ratio 2."),
    'M100_crossq_c2_256_history': ('M100: CrossQ, 256 x 256 + history features',
        "Genuine reinforcement learning, deployable in a real building: the policy learns from reward only and sees just the present, the past and CityLearn's own forecasts (no perfect-foresight plan, no imitation). Base: M27 (SAC, score-aligned reward with carbon weight 2). Explored on seed 42 first; more seeds only if it helps. M92 and M93 combined."),
    'M101_sac_c3': ('M101: SAC, carbon weight 3',
        "Genuine reinforcement learning, deployable in a real building: the policy learns from reward only and sees just the present, the past and CityLearn's own forecasts (no perfect-foresight plan, no imitation). Base: M27 (SAC, score-aligned reward with carbon weight 2). Explored on seed 42 first; more seeds only if it helps. As M27 with carbon weight 3 instead of 2 (carbon matters most for the paper)."),
    'M102_sac_c2_60k': ('M102: SAC score-aligned, carbon 2 (as M27) at 60k env steps', 'The fair control for Round 22, which was shortened to 60k env steps for time: same method as M27 with the same budget as M90-M101.'),
    'M103_sac_hist_rich': ('M103: SAC + rich history features',
        "Genuine RL (no plans, no imitation). Round 23: builds on M93 (SAC, score-aligned reward with carbon weight 2, history features; 60k env steps, seed 42 first). As M93, plus each building's own load minus solar at the same lags, the values 24 hours ago, and yesterday's mean / max carbon intensity and district load minus solar."),
    'M104_sac_hist_dense': ('M104: SAC + dense history features',
        'Genuine RL (no plans, no imitation). Round 23: builds on M93 (SAC, score-aligned reward with carbon weight 2, history features; 60k env steps, seed 42 first). As M93 with the persistence forecast for every next hour 1..12 instead of 1, 2, 3, 6, 12.'),
    'M105_sac_hist_obsnorm': ('M105: SAC + history + observation normalisation',
        'Genuine RL (no plans, no imitation). Round 23: builds on M93 (SAC, score-aligned reward with carbon weight 2, history features; 60k env steps, seed 42 first). As M93 with running observation normalisation (rewards unscaled).'),
    'M106_sac_hist_lr1e3': ('M106: SAC + history, learning rate 1e-3',
        'Genuine RL (no plans, no imitation). Round 23: builds on M93 (SAC, score-aligned reward with carbon weight 2, history features; 60k env steps, seed 42 first). As M93 with learning rate 1e-3 instead of 3e-4.'),
    'M107_sac_hist_batch512': ('M107: SAC + history, batch 512',
        'Genuine RL (no plans, no imitation). Round 23: builds on M93 (SAC, score-aligned reward with carbon weight 2, history features; 60k env steps, seed 42 first). As M93 with batch size 512 instead of 256.'),
    'M108_sac_hist_tau002': ('M108: SAC + history, tau 0.02',
        'Genuine RL (no plans, no imitation). Round 23: builds on M93 (SAC, score-aligned reward with carbon weight 2, history features; 60k env steps, seed 42 first). As M93 with target-network update rate 0.02 instead of 0.005.'),
    'M109_sac_c2_400k': ('M109: M27 trained to 400k env steps', 'Genuine RL. M27 (SAC, score-aligned reward, carbon weight 2) continued from its 200k checkpoint and replay buffer to 400k env steps (about 46 simulated years), since it was still improving between 60k and 200k.'),
    'M110_tqc_c2': ('M110: TQC with M27 settings, 200k env steps', 'Genuine RL. As M27 (score-aligned reward, carbon weight 2, 64 x 64 networks, 200k env steps) with TQC instead of SAC (SB3-contrib defaults: 2 distributional critics, 25 quantiles each, top 2 dropped per critic).'),
    'M111_sac_hist_rich_obsnorm': ('M111: SAC + rich history + observation normalisation, 200k env steps', 'Genuine RL. M27 with the rich past-only history features of M103 and observation normalisation (M105), the two best Round 23 changes stacked, at the full 200k budget.'),
    'M112_sac_hist_200k': ('M112: SAC + history features (M93) at 200k env steps', 'Genuine RL. M93 (M27 + history features) at the full 200k budget, to test the history features without the noise of 60k runs.'),
    'M113_tqc_hist': ('M113: TQC + history features, 200k env steps', 'Genuine RL. M110 (TQC with M27 settings) plus the past-only history features of M93, at the full 200k budget.'),
    'M114_tqc_score_aligned': ('M114: TQC with the score-aligned reward (carbon weight 1), 200k env steps', 'Genuine RL. M26 (score-aligned reward, carbon weight 1) with TQC instead of SAC, to test whether the TQC gain of M110 carries over to the other reward weighting.'),
    'M115_tqc_hist_5critics': ('M115: TQC + history, 5 critics',
        'Genuine RL (no plans, no imitation). Round 25: one change to M113 (TQC + score-aligned reward with carbon weight 2 + past-only history features), full budget, 3 seeds. As M113 with 5 distributional critics instead of 2 (the TQC paper default), still dropping the top 2 quantiles per critic.'),
    'M116_tqc_hist_lrdecay300k': ('M116: TQC + history, decaying learning rate, 300k steps',
        'Genuine RL (no plans, no imitation). Round 25: one change to M113 (TQC + score-aligned reward with carbon weight 2 + past-only history features), full budget, 3 seeds. As M113 with the learning rate decayed linearly from 3e-4 to 0, over 300k env steps: constant-rate training past 200k drifted (M109).'),
    'M117_tqc_hist_drop5': ('M117: TQC + history, more conservative critics',
        'Genuine RL (no plans, no imitation). Round 25: one change to M113 (TQC + score-aligned reward with carbon weight 2 + past-only history features), full budget, 3 seeds. As M113 dropping the top 5 of 25 quantiles per critic instead of 2.'),
    'M118_tqc_hist_utd2': ('M118: TQC + history, 2 updates per env step',
        'Genuine RL (no plans, no imitation). Round 25: one change to M113 (TQC + score-aligned reward with carbon weight 2 + past-only history features), full budget, 3 seeds. As M113 with update-to-data ratio 2.'),
    'M119_tqc_hist_ent10': ('M119: TQC + history, target entropy -10',
        'Genuine RL (no plans, no imitation). Round 25: one change to M113 (TQC + score-aligned reward with carbon weight 2 + past-only history features), full budget, 3 seeds. As M113 with target entropy -10 instead of -5 (the action dimension), i.e. less exploration noise late in training.'),
    'M120_tqc_hist_phase1': ('M120: TQC + history, Phase I-focused reward',
        'Genuine RL (no plans, no imitation). Round 25b: one change to M113 (TQC + score-aligned reward with carbon weight 2 + past-only history features), full budget, seed 42. As M113 with ramping and peak weights 0.2 instead of 0.5, aimed at the Phase I score (cost and carbon).'),
    'M121_tqc_hist_c15': ('M121: TQC + history, carbon weight 1.5',
        'Genuine RL (no plans, no imitation). Round 25b: one change to M113 (TQC + score-aligned reward with carbon weight 2 + past-only history features), full budget, seed 42. As M113 with carbon weight 1.5 instead of 2.'),
    'M122_tqc_hist_256': ('M122: TQC + history, 256 x 256 networks',
        'Genuine RL (no plans, no imitation). Round 25b: one change to M113 (TQC + score-aligned reward with carbon weight 2 + past-only history features), full budget, seed 42. As M113 with 256 x 256 actor and critics instead of 64 x 64.'),
    'M123_tqc_hist_gamma098': ('M123: TQC + history, gamma 0.98',
        'Genuine RL (no plans, no imitation). Round 25b: one change to M113 (TQC + score-aligned reward with carbon weight 2 + past-only history features), full budget, seed 42. As M113 with discount 0.98 instead of 0.99.'),
    'M124_tqc_hist_batch512': ('M124: TQC + history, batch 512',
        'Genuine RL (no plans, no imitation). Round 25b: one change to M113 (TQC + score-aligned reward with carbon weight 2 + past-only history features), full budget, seed 42. As M113 with batch size 512.'),
    'M125_tqc_hist_rich': ('M125: TQC + rich history features',
        'Genuine RL (no plans, no imitation). Round 25b: one change to M113 (TQC + score-aligned reward with carbon weight 2 + past-only history features), full budget, seed 42. As M113 with the rich past-only history features (per-building lags, 24 h ago, yesterday statistics).'),
    'M126_tqc_hist_dense': ('M126: TQC + dense history features',
        'Genuine RL (no plans, no imitation). Round 25b: one change to M113 (TQC + score-aligned reward with carbon weight 2 + past-only history features), full budget, seed 42. As M113 with the persistence forecast for every next hour 1..12.'),
    'M127_tqc_hist_nowear': ('M127: TQC + history, no battery-wear penalty',
        'Genuine RL (no plans, no imitation). Round 25b: one change to M113 (TQC + score-aligned reward with carbon weight 2 + past-only history features), full budget, seed 42. As M113 without the battery-wear term (w_batt 0 instead of 0.1).'),
    'M128_tqc_hist_wear03': ('M128: TQC + history, battery-wear weight 0.3',
        'Genuine RL (no plans, no imitation). Round 25b: one change to M113 (TQC + score-aligned reward with carbon weight 2 + past-only history features), full budget, seed 42. As M113 with battery-wear weight 0.3 instead of 0.1.'),
    'M129_rl_tuned_mpc': ('M129: RL-tuned MPC (TQC)',
        'Genuine RL + forecasting + optimisation, all causal (deployable). Round 26. Every hour a TQC policy sets the ramp weight, carbon weight and end-of-horizon value of a receding-horizon LP (MPC) that uses causal forecasts (seasonal persistence with error correction) of load, PV, carbon intensity and price; the MPC turns them into battery actions. The policy sees only district-level features, so it runs on districts of any size. Reward: score-aligned counterfactual, carbon weight 2. 100k env steps.'),
    'M130_rl_tuned_mpc_sac': ('M130: RL-tuned MPC (SAC)',
        'Genuine RL + forecasting + optimisation, all causal (deployable). Round 26. As M129 with SAC instead of TQC.'),
    'M131_rl_tuned_gbm_mpc': ('M131: RL-tuned MPC with LightGBM forecasts (TQC)', 'Genuine RL + forecasting + optimisation, all leakage-free. As M129, but the MPC uses the scale-free LightGBM load / PV forecasters of tools/forecast_lab.py, trained only on buildings outside the controlled district (the recipe of the challenge winners: forecasting, optimisation and RL). 100k env steps.'),
    'M132_rl_tuned_mpc_grid': ('M132: RL-tuned MPC with a grid-weighted reward', 'Genuine RL + forecasting + optimisation, all causal. As M129 (TQC sets the weights of a causal-forecast MPC every hour) with reward weights carbon 1, ramping 1, peak 1, so the grid KPI D counts as much as in the Phase III score.'),
    'M133_rl_gbm_mpc_score_reward': ('M133: RL-tuned LightGBM MPC, carbon weight 1',
        'Genuine RL + forecasting + optimisation, leakage-free. Round 27: one change to M131 (TQC sets the weights of an MPC driven by leave-district-out LightGBM forecasts). As M131 with the score-aligned reward of carbon weight 1 (M26), closer to the Phase III score where C, G and D weigh the same.'),
    'M134_rl_gbm_mpc_200k': ('M134: RL-tuned LightGBM MPC, 200k steps',
        'Genuine RL + forecasting + optimisation, leakage-free. Round 27: one change to M131 (TQC sets the weights of an MPC driven by leave-district-out LightGBM forecasts). As M131 trained for 200k env steps instead of 100k.'),
    'M135_rl_gbm_mpc_sac': ('M135: RL-tuned LightGBM MPC with SAC',
        'Genuine RL + forecasting + optimisation, leakage-free. Round 27: one change to M131 (TQC sets the weights of an MPC driven by leave-district-out LightGBM forecasts). As M131 with SAC instead of TQC.'),
}
SEEDS_OVERRIDE = {'M47_bc_grid_members': '[1, 2, 3, 4, 5, 6, 7]   # ensemble members (see the description)',
                  'M48_bc_big_gentle_members': '[1, 2, 3, 4, 5]   # ensemble members (see the description)',
                  'M52_bc_deep_members': '[1, 2, 3, 4, 5]   # ensemble members (see the description)',
                  'M90_sac_c2_256': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M91_tqc_c2_256': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M92_crossq_c2_256': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M93_sac_c2_history': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M94_sac_c2_nstep3': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M95_sac_c2_gamma095': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M96_sac_c2_256_history': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M97_tqc_c2_256_history': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M98_sac_c2_per_building': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M99_sac_c2_utd2': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M100_crossq_c2_256_history': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M101_sac_c3': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M102_sac_c2_60k': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M103_sac_hist_rich': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M104_sac_hist_dense': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M105_sac_hist_obsnorm': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M106_sac_hist_lr1e3': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M107_sac_hist_batch512': '[42]   # explore on one seed; add 100, 2026 if it works',
                  'M108_sac_hist_tau002': '[42]   # explore on one seed; add 100, 2026 if it works',
                  }

for method, (title, desc) in METHOD_DOCS.items():
    cells = [
        new_markdown_cell(
            f"# {title}\n\n{desc}\n\n"
            "**How to run:** Run All. Results go to `results/" + method + "/seed_<seed>/`.\n\n"
            "- A seed with `summary.json` is finished and skipped.\n"
            "- A seed that stopped mid-training resumes from its last checkpoint (saved every 20k env steps).\n"
            "- To add seeds later, append them to `SEEDS` below and Run All again."),
        new_code_cell(
            f"METHOD = '{method}'\n"
            + (f"SEEDS = {SEEDS_OVERRIDE[method]}" if method in SEEDS_OVERRIDE
               else "SEEDS = [42, 100, 2026]   # Stage B: add 7, 1234 here and Run All again")),
        new_code_cell("%run ../00_core.ipynb"),
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
