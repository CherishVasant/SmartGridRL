# Phase 3 Plan: Multi-KPI Adaptive Reward (one-day version)

- **Deadline:** 1 day.
- **Training length:** 200k env steps for every run. Phase 2 (50k, in `01_notebook_pipeline/`) is the "short training" reference.
- **Status:** Stage A started 2026-10-16 around 10:00. All 9 notebooks run in parallel on this machine, about 40–50 min per seed.

## Methods

| ID | Notebook | Method | Agent |
|---|---|---|---|
| B0 | (in `00_core`) | Rule-based controller, the KPI reference (ratio = 1.0) | — |
| B1 | `B1_cost_only` | Cost-Only PPO: `−cost` | Central |
| B3 | `B3_static_matched` | Static-Matched PPO: `−cost − 0.41·carbon` | Central |
| B4 | `B4_dynamic` | Dynamic PPO: `−cost − λ₀·Î_t·carbon` (phase 2 method) | Central |
| M1 | `M1_dynamic_sharp` | Dynamic, φ = 1 only in the 25% dirtiest hours | Central |
| M2 | `M2_multikpi_fixed` | Multi-KPI reward, all weights fixed at 1 | Central |
| M3 | `M3_multikpi_adaptive` | Multi-KPI reward, **adaptive learned weights** (main central proposal) | Central |
| M4 | `M4_per_building_dynamic` | Dynamic reward, one agent per building | Per-building |
| M5 | `M5_per_building_adaptive_shared` | Multi-KPI adaptive, one shared set of weights | Per-building |
| M6 | `M6_per_building_adaptive_own` | Multi-KPI adaptive, **separate weights per building** (main per-building proposal) | Per-building |

- **Dropped:** B2 (Static λ=1), because phase 2 already covers it.
- **Later, only if nothing else is left to improve:** SAC; longer training (500k).
- **Future work:** 2023 dataset + HVAC.
- The forecasts idea (S2) is already covered: CityLearn 2022 observations include 6/12/24-hour price forecasts.

## Stage A results (3 seeds, 200k steps; score = mean KPI ratio vs RBC, lower is better)

| Method | Score | Cost | Carbon | Ramping | Note |
|---|---|---|---|---|---|
| B5 No battery (never used) | **0.949** | 1.154 | 0.848 | 0.860 | beats RBC and every learned agent |
| M5 / M6 per-building adaptive | 1.075 | 1.155 | 0.849 | 0.861 | collapsed: battery idle on every seed |
| B3 Static-Matched | 1.137 | 0.850 | 1.039 | 1.652 | |
| M2 Multi-KPI fixed | 1.152 | 1.123 | 0.983 | 1.339 | |
| M3 Multi-KPI adaptive | 1.164 | 1.094 | 1.007 | 1.418 | smallest seed spread (std 0.027) |
| M1 Dynamic, top-25% φ | 1.172 | 0.849 | 1.040 | 1.698 | |
| B1 Cost-Only | 1.192 | 0.869 | 1.057 | 1.729 | |
| B4 Dynamic | 1.218 | 0.905 | 1.067 | — | |
| M4 Per-building Dynamic | 1.416 | **0.693** | 0.972 | 2.195 | best cost, worst ramping |

**Takeaways**
- **Cost arbitrage creates ramps and peaks.** Charging in cheap hours is what the agents learn, and it makes all of them worse than doing nothing on grid KPIs.
- **Multi-KPI rewards (M2, M3) help.** They cut ramping by 20–30% vs Static-Matched and bring carbon to about RBC level.

**Decision (user, 13:30):** skip Stage B for now. First try more methods; at night, pick the overall top 3 and run seeds 7 and 1234 for them plus B1, B3 and B4.

## Round 1 (started ~13:45, 3 seeds each)

| ID | Method | Tests |
|---|---|---|
| M7 | M3 + VecNormalize (observation and reward normalisation) | Does PPO just need scaled inputs? |
| M8 | M2 + VecNormalize | Same, for fixed weights |
| M9 | −(cost + carbon + L²), squared district load | Classic peak / load-factor penalty |
| M10 | M4 + 1/5 district ramp/peak penalty per building | Keep M4's cost savings without the ramps |
| M11 | Per-building, local cost + carbon only, district ramp/peak with adaptive weights | Fix for the M5/M6 collapse |
| M12 | SAC with the M2 reward | Algorithm comparison (M2 vs M12) |

**Round 1 results (3 seeds, complete 2026-10-16 22:59):**

| Method | Score (± seed std) | Cost | Carbon | Ramping | Daily peak | Annual peak | Note |
|---|---|---|---|---|---|---|---|
| M12 SAC multi-KPI | **0.962** ± 0.038 | 0.881 | 0.932 | 1.187 | **0.835** | **0.935** | beats RBC on cost **and** carbon on 3/3 seeds |
| M8 PPO + norm | 1.013 ± 0.055 | 1.103 | 0.893 | 1.030 | 0.950 | 1.084 | |
| M7 PPO adaptive + norm | 1.030 ± 0.065 | 1.103 | 0.896 | 1.035 | 0.985 | 1.127 | |
| M10 per-building + district | 1.038 ± 0.068 | 1.150 | 0.850 | 0.887 | 0.878 | 1.441 | battery idle on all seeds |
| M9 cost + carbon + L² | 1.115 ± 0.022 | 0.807 | 0.970 | 1.504 | 0.951 | 1.429 | |
| M11 per-building, district terms | 1.246 ± 0.060 | 0.764 | 0.931 | 1.808 | 1.298 | 1.607 | |

- **SAC is the biggest single win** (M2 → M12: 1.152 → 0.962, same reward). **Normalisation** is second (M2 → M8: 1.152 → 1.013).
- **Per-building agents fail**: 3 of 5 variants (M5, M6, M10) leave the battery idle; the two that use it (M4, M11) wreck ramping and peaks.
- **Learned weights don't beat fixed ones** (M3 ≈ M2, M7 ≈ M8).

## Round 2 (SAC combinations, 3 seeds, complete 2026-10-17 05:39)

| ID | Method | Score (± seed std) | Cost | Carbon | Ramping | Daily peak | Annual peak |
|---|---|---|---|---|---|---|---|
| M13 | SAC + VecNormalize, fixed multi-KPI | 0.980 ± **0.007** | 1.108 | **0.890** | **0.998** | 0.892 | 0.980 |
| M16 | Residual SAC on the RBC (action = RBC + correction) + norm | 0.998 ± 0.045 | 1.076 | 0.907 | 1.047 | 0.925 | 1.018 |
| M14 | SAC + norm, adaptive weights | 1.000 ± 0.010 | 1.132 | 0.921 | 1.049 | 0.898 | 0.988 |
| M15 | SAC + norm, cost + carbon + L² | 1.178 ± 0.022 | **0.695** | 1.040 | 1.740 | 1.192 | 1.368 |

- None beats M12 overall. **M13 is the most consistent method** (std 0.007) and the only learned agent with ramping at or below RBC and both peaks below RBC; it pays ~10% more than RBC.
- Adaptive weights again do not help (M14 vs M13).
- M15 shows the cost-vs-grid conflict most sharply: lowest cost of all, worst grid KPIs among SAC runs.

**Against no battery (CityLearn's own reference):** with the CityLearn 2022 Challenge-style score (mean of cost, carbon and
grid = mean(ramping, 1 − LF), all ÷ no battery), no method beats doing nothing (best: M12 1.016, M13 1.030, RBC 1.041).
**Every battery-using agent emits more carbon than no battery** (round-trip losses, cheap-but-dirty charging). The paper should
say so; claims of beating state of the art are only defensible against the RBC.

## Round 3 (double down on SAC + fixed multi-KPI; started 2026-10-17 10:36)

| ID | Change vs M12 | Tests |
|---|---|---|
| M17 | VecNormalize on observations only (rewards not normalised) | Did M13 lose M12's cost/peak gains because of reward normalisation alone? (SAC is reward-scale sensitive) |
| M18 | Battery-wear weight 0.1 → 0.5 | Less useless cycling → fewer losses → carbon toward the no-battery level |
| M19 | Carbon weight 1 → 2 | Push carbon below no battery directly |
| M20 | Reward = gain over the no-battery run at the same hour (counterfactual baseline) | Lower-variance critic targets; same optimal policy |

**Seed 42 (interim, 11:50):** M19 0.916 (best ever vs RBC; annual peak 0.803), M20 0.941 (cost 0.869, daily peak 0.782).
**M19 and M20 are the first methods below 1.0 on the CityLearn challenge-style score vs no battery** (0.995 and 0.978).
M17 (0.970) and M18 (1.011) did not help. Seeds 100 and 2026 running.

**Reward flaw found (peak term):** the `peak` term is 0 in the first hour of each day and that hour sets the day's running
max, so a load spike at the start of a day is free and hides later peaks. In the counterfactual reward (M20) it is even
paid for (the agent is credited with the no-battery run's evening peak penalty). Fixed with a `full_day_peak` option:
the first hour counts its own load, so a day's terms sum to exactly its peak load (checked: error 0 on the RBC year).
Earlier methods keep the old term so their results stay reproducible.

## Round 4 (doubling down on M19 + M20, all with the full-day peak term; queued 2026-10-17 11:55)

| ID | Config | Tests |
|---|---|---|
| M21 | counterfactual + carbon 2 | Do the two Round 3 gains stack? |
| M22 | counterfactual + carbon 3 | Where does a stronger carbon push stop helping? |
| M23 | counterfactual + carbon 2 + ramp 2 | Ramping is still above RBC for every SAC run |
| M24 | counterfactual only (M20 with the peak fix) | Isolates the effect of the fix (M24 vs M20) |

## Target: the published CityLearn Challenge 2022 scores (user decision, 2026-10-17)

The goal is to beat the published challenge results, not the RBC, and to get carbon below no battery.
Official KPIs (Nweye et al. 2023, PMLR 220, Appendix A): **C** = Σ max(0, E_h·price_h) and **G** = Σ max(0, E_h·carbon_h)
on the **district** net load E_h, each ÷ the no-battery value; **D** = mean(ramping, 1 − monthly load factor) ÷ no battery.
**Phase I** (public leaderboard, the 5 training buildings = our dataset) ranks by mean(C, G); Phase III adds D.
Computed for every run by `tools/challenge_kpis.py` → `results/challenge_kpis.csv`.

| Reference | Phase I mean(C, G) | C | G | D |
|---|---|---|---|---|
| Phase I leaderboard #1 real entry (tymur_prorochenko) | **0.631** | 0.544 | 0.718 | – |
| Phase I #2 (Together, overall winner) | 0.635 | 0.578 | 0.691 | – |
| Phase III public #1 (Together; train + validation buildings, mean(C, G, D) = 0.728) | – | 0.639 | 0.808 | 0.738 |
| Participants' median, Phase III private (mean(C, G, D) = 0.907) | – | 0.792 | 0.940 | 0.996 |
| Ours: M20, seed 42 | 0.826 | 0.777 | 0.874 | 1.102 |
| Ours: M12, 3 seeds | 0.881 | 0.831 | 0.931 | 1.184 |

- **Correction:** with the official district-level clipping, our SAC agents already emit *less* than no battery (G < 1).
  The earlier "every agent emits more than no battery" used CityLearn's per-building values.
- **Why our rewards underperform on this metric:** they used CityLearn's per-building cost (exports are paid) and carbon
  (no credit when one building's export offsets another's import), not the official district-level definitions.

## Round 5 (score-aligned rewards; started 2026-10-17 12:21)

Official district-level cost/carbon terms (`official_terms`); every term ÷ its no-battery per-step mean (`norm_ref`);
gain over no battery (counterfactual); full-day peak. Checked: over a year of RBC actions, −Σreward/T equals
(C−1) + (G−1) + ½(R−1) + ½(P−1) + wear exactly (0.519749 both ways), so M26 optimises the challenge score directly.

| ID | Weights (cost, carbon, ramp, peak) | Tests |
|---|---|---|
| M25 | 1, 1, 1, 1 with RBC scaling | Effect of the official terms alone (vs M24) |
| M26 | 1, 1, 0.5, 0.5 with no-battery scaling | Reward = challenge score |
| M27 | 1, 2, 0.5, 0.5 | Challenge score with extra carbon emphasis |
| M28 | 1, 1, 0.2, 0.2 | Phase I score mean(C, G), light grid penalty |

## Perfect-foresight LP plans (`tools/lp_oracle.py`, `results/O*`)

The Phase I leaderboard scored on the 5 training buildings, so top entries could plan with the year known (Together used
dynamic programming + CMA-ES). An LP over the whole year (official district-level objective; battery 6.4 kWh, 5 kW,
power taper above 80% SoC) is solved with HiGHS in ~30 s, then replayed in CityLearn by tracking its planned SoC.

| Plan | Objective | Phase I mean(C, G) | C | G | D | mean(C, G, D) |
|---|---|---|---|---|---|---|
| O3 | C + G, efficiency 0.87 | **0.629** (published #1: 0.631) | 0.566 | **0.692** | 1.096 | 0.785 |
| O1 | C + G, efficiency 0.90 | 0.631 | 0.569 | 0.692 | 1.113 | 0.792 |
| O2 | C + G + (R + P), efficiency 0.90 | 0.658 | 0.595 | 0.721 | **0.555** | **0.624** |
| O6 | C + G + 0.5 (R + P) | 0.648 | 0.584 | 0.713 | 0.583 | 0.627 |

Bug fixed on the way: after `step()` CityLearn stores the new SoC at `soc[time_step]`; reading `time_step - 1` made the
replay lag one hour (0.792 instead of 0.631).

**Round 3, seed 100 (13:20):** M20 is consistent (Phase I 0.826 / 0.830, G 0.874 / 0.881 on seeds 42 / 100) and the
best RL method; M19's seed-42 lead did not hold (0.871 → 0.898).

**Carbon floor:** a carbon-focused plan (O7, carbon weight 10) reaches G = 0.691, the same as O3 (0.692) and the
published leader Together (0.691). Cost and carbon are almost perfectly aligned at the optimum here, so ~0.69 is the
floor for these batteries; the goal is reaching it without knowing the future.

**Collapse (Round 4):** M21 (counterfactual + full-day peak + carbon 2) and M24 (counterfactual + full-day peak) never
used the battery (training cost/carbon flat at the no-battery 5,636 / 4,640 for 140k–200k steps; M21 eval: all actions
"discharge", SoC 0, every KPI exactly 1.000). With RBC scaling and peak weight 1, any charging that sets a new daily peak
is expensive, so "do nothing" (reward 0) is a trap. M26/M27 (same counterfactual + full-day peak, but no-battery scaling
and peak weight 0.5) learn normally. M21/M24 stopped after seed 42; M22, M23, M25 (same family) dropped.

## Round 6 (fusion: residual SAC around an LP plan; queued 2026-10-17 12:55)

Action = plan's action for that hour + scaled SAC correction (`residual_base`, `residual_scale`); checked that a zero
correction reproduces the plan's year exactly (max load difference 2e-7).

| ID | Base plan | Reward | Correction |
|---|---|---|---|
| M29 | O3 (Phase I) | M28 Phase I-aligned | ±25% |
| M30 | O2 (grid) | M26 score-aligned | ±25% |
| M31 | O2 | M27 score-aligned, carbon 2 | ±25% |
| M32 | O2 | M26 | full range |

**Round 6 result: residual SAC around the plan makes it worse** (seed 42; plan alone in brackets):
M29 Phase I 0.791 (O3: 0.629), M30 0.756 / mean(C, G, D) 0.820 (O2: 0.658 / 0.624), M31 0.759 / 0.818, M32 0.802 / 0.886.
SAC's deterministic correction drifts well away from 0 and adds noise to a near-optimal schedule (D rises from 0.555 to
0.94–1.26). Branch closed after seed 42 (M29–M32 stopped). Imitating the plan (Round 7) keeps far more of its value.

## Round 7 (behaviour cloning of the LP plans; queued 2026-10-17 14:00)

The SAC actor is fitted (supervised, 300 epochs) to the plan's actions on the observations along the plan's trajectory
(`bc_from`). Unlike the residual methods it needs only the current observation, not the plan, so it is a deployable
controller. With VecNormalize (observations only), the normalisation statistics come from the same data.

| ID | Clone of | Then |
|---|---|---|
| M33 | O2 (grid plan) | nothing (1,000 env steps, no SAC update) |
| M35 | O3 (Phase I plan) | nothing |
| M34 | O2 | SAC fine-tuning, score reward (M26), initial entropy weight 0.05 |
| M36 | O3 | SAC fine-tuning, Phase I reward (M28), initial entropy weight 0.05 |

**Results (official KPIs vs no battery):**

| Method | Seeds | Phase I mean(C, G) | mean(C, G, D) | C | G | D |
|---|---|---|---|---|---|---|
| **M33 clone of O2** | 3 | **0.750 ± 0.003** | **0.806** | 0.688 | **0.811** | 0.917 |
| M35 clone of O3 | 3 | ~0.80 | ~0.95 | ~0.74 | ~0.87 | ~1.24 |
| M26 score-aligned SAC (Round 5) | 1 | 0.790 | 0.869 | 0.736 | 0.845 | 1.027 |
| M20 best earlier RL | 2 | 0.828 | 0.922 | 0.778 | 0.878 | 1.108 |

- **M33 is the best controller that does not know the future**, very stable across seeds; ~10th on the published Phase I
  leaderboard (between ludwigbald 0.752 and the leaders' 0.631).
- **Cloning the smooth grid plan beats cloning the Phase I plan, even on the Phase I score**: a smoother target is easier
  to imitate from observations alone.
- The score-aligned reward (M26) is the best pure-RL reward so far.

## Round 8 (better imitation of the smooth plan; queued 2026-10-17 14:45)

DAgger (`bc_dagger_iters`): roll the clone out, label each visited state with the action that moves every battery to the
plan's SoC for that hour, aggregate, refit; lets the clone learn to recover from its own drift. New plan O8 (grid weight 2;
C 0.614, G 0.733, D 0.534, mean(C, G, D) 0.627) is even smoother.

| ID | Clone of | Network | DAgger |
|---|---|---|---|
| M37 | O2 | 256×256 | 3 iterations |
| M38 | O2 | 256×256 | – |
| M39 | O2 | 64×64 | 3 iterations |
| M40 | O8 | 256×256 | 3 iterations |

**Seed 42:** the 256×256 clones fit the plan almost exactly on its own trajectory (MSE 0.002 vs 0.024), but DAgger with a
one-hour catch-up expert hurt the grid KPI: M37 0.762 / 0.878 (Phase I / mean C, G, D), M40 0.758 / 0.858, M39 0.796 /
0.957, all worse than plain M33 (0.753 / 0.806). The catch-up labels are large, spiky corrections. Also M27 (score-aligned,
carbon 2) seed 42: 0.767 / 0.864, G 0.817, the best pure RL so far.

**Bigger clone alone (M38) is worse** (seed 42: 0.775 / 0.838 vs M33 0.753 / 0.806) despite a 10× lower fit error: it
overfits the plan's own trajectory and recovers worse from drift. Small actors generalise better in closed loop.

## Round 10: ensembles of clones (`tools/ensemble_eval.py`, 2026-10-17 15:45)

Each hour's action = mean of the members' deterministic actions (each with its own VecNormalize statistics). No training.

| Ensemble | Phase I mean(C, G) | mean(C, G, D) | C | G | D |
|---|---|---|---|---|---|
| **E_M33: M33 seeds 42, 100, 2026** | **0.734** | **0.788** | 0.673 | **0.796** | 0.895 |
| (M33 single seeds) | 0.747–0.753 | 0.805–0.807 | 0.686–0.691 | 0.807–0.814 | 0.912–0.921 |

| E_M33_M38: M33 + M38 (6 members) | 0.716 | 0.758 | 0.650 | 0.781 | 0.844 |
| **E_clones_mix: M33 ×3, M38 ×3, M37 ×2, M40 ×2** | **0.701** | **0.731** | **0.640** | **0.762** | **0.792** |

Ensembles beat every member on every KPI, and diversity helps even with weaker members (M37/M40 alone are worse).
E_clones_mix is ~6th on the published Phase I leaderboard (OneWo 0.698, LA_gang 0.719); its mean(C, G, D) 0.731 is level
with the Phase III public winner Together's 0.728 (different buildings).

**Best single models (seed 42), Round 9:** gentle DAgger fixes the big clone's overfitting —
M41 (256×256 + gentle DAgger, O2) Phase I **0.720**, mean(C, G, D) 0.751, G 0.775; M42 (same, O8) 0.722 / **0.741**, G 0.775,
D 0.780; M45 (64×64 + gentle DAgger) 0.742 / 0.805. Hard DAgger was worse (M39 0.796).

## Round 11 (double down on M41; queued 2026-10-17 16:09)

| ID | Change vs M41 |
|---|---|
| M48 | none, seeds 1–5 (ensemble members) |
| M49 | 6 DAgger iterations |
| M50 | DAgger gain 0.5 |
| M51 | 3 × 256 network |

**Results (seed 42 unless noted):** M51 (3×256 + gentle DAgger) Phase I **0.685**, mean(C, G, D) **0.687**, C 0.626,
**G 0.745**, D 0.691 — the best single deployable model, past the 5th published Phase I entry (OneWo 0.698).
M41 is stable (seeds 42 / 100: 0.720 / 0.719); M48 seed 1 0.717; M50 (gain 0.5) 0.723 / 0.766 (worse D than gain 0.3).
Ensembles of plain/hard-DAgger clones plateau at ~0.701 (15 members 0.701, 21 members with the weak M35/M39 0.704).

## Round 12 (double down on M51; queued 2026-10-17 16:46)

| ID | Change vs M51 |
|---|---|
| M52 | none, seeds 1–5 (ensemble members) |
| M53 | 4 × 256 |
| M54 | 3 × 512 |
| M55 | clone of O8 (smoother plan) |
| M56 | 6 DAgger iterations |

**Results so far:** M51 is robust (seeds 42 / 100: 0.685 / 0.688; M52 seed 1: 0.688). **M53 (4×256) seed 42: Phase I
0.681, Score 0.672, G 0.738, D 0.654: the best single model.** M55 (deep clone of the smoother O8): 0.695 / 0.687, so a
smoother plan gives a worse Phase I. M49 (2×256, 6 DAgger iterations): 0.703 / 0.713 vs M41's 0.720 / 0.751, so more DAgger
helps.

**Clone + SAC fine-tuning makes the clone worse** (M34 seed 42: 0.789 / 0.878 vs its starting clone M33's 0.753 / 0.806),
the same lesson as Round 6. Branch cut: M34, M36, M43, M44, M46 stopped after seed 42.

## Round 13 (deep clones of plans with a better Phase I score; queued 2026-10-17 17:26)

The deep clone (~0.687) is already close to its plan O2 (0.658), which gives up cost and carbon for smoothness. Target the
published top 3 (≤ 0.651) by cloning less smooth plans.

| ID | Clone of (plan Phase I / Score) | DAgger |
|---|---|---|
| M57 | O6, grid weight 0.5 (0.648 / 0.627) | 3 |
| M58 | O3, Phase I plan (0.629 / 0.785) | 3 |
| M59 | O9, grid weight 0.25 (solving) | 3 |
| M60 | O6 | 6 |

**Seed 42:** M58 (deep clone of O3) Phase I **0.658** (published #3: 0.651), carbon **G 0.717** (published #1: 0.718),
but Score 0.815 (D 1.128: the Phase I plan is not smooth). M57 (O6) 0.682 / 0.698. M54 (3×512 clone of O2) **0.671 /
0.652**, G 0.731, D 0.614: the best on the full score. Two families: clone O3 for Phase I, clone O2/O9 for the full score.

## Round 14 (bigger networks, more DAgger; started 2026-10-17 18:10)

M61 4×512 (O2), M62 3×1024 (O2), M63 3×512 + 6 DAgger iterations (O2), M64 3×512 on O9.

## Round 15 (big clones of the Phase I plan O3, aimed at the published top 3; queued 18:12)

M65 3×512 (O3), M66 4×512 (O3), M67 3×512 + 6 DAgger iterations (O3).

## Round 9 (queued 2026-10-17 15:10)

| ID | Method |
|---|---|
| M41 | big clone of O2 + gentle DAgger (plan action + 30% of the SoC error per hour, `bc_dagger_gain` 0.3) |
| M42 | big clone of O8 + gentle DAgger |
| M43 | big clone of O2 + SAC fine-tuning with the M27 reward (score-aligned, carbon 2), entropy 0.05 |
| M44 | big clone of O8 + SAC fine-tuning with the M27 reward |

## Multi-KPI reward

Every term is divided by its RBC per-step average:

```
r_t = −( ĉost + μ_carbon·φ(Î_t)·ĉarbon + μ_ramp·r̂amp + μ_peak·p̂eak + 0.1·b̂att )
```

- `ramp` = |L_t − L_{t−1}|.
- `peak` = increase above the day's highest load so far.
- `batt` = |ΔSoC|.

**Adaptive weights (M3, M5, M6), Dynamic Weight Average style:**

- Once per simulated week, compute `gap_k = log( (KPI_k / KPI_k^RBC over that week) / target_k )`. The targets are carbon 0.90, ramping 1.0 and peak 1.0.
- New weights = `3 · (0.3/3 + 0.7 · softmax(gap))`, blended 20% into the old weights.
- **The weights always sum to 3**, the same as M2's fixed weights. So M2 vs M3 tests only *how* the weight is distributed.
- **Every KPI keeps a weight of at least 0.3.**
- **Why not plain Lagrangian updates:** a first version used them (μ += η·(ratio − target)). Every weight hit its cap of 10 within two episodes, because early policies are worse than RBC on every KPI. That is worth one sentence in the paper.

**Per-building agents (M4–M6):**

- One PPO policy is shared by the 5 buildings. Each building observes its own data plus a building ID.
- **Each building has its own reward**, plus 1/5 of the district ramp and peak terms.
- **M6:** each building learns its own weights from its own KPIs vs the RBC's same building.
- **M5:** all buildings share district-level weights.

## Fixes vs phase 2

- **Carbon intensity index:** phase 2's dynamic reward read carbon intensity one hour late (`t−1`). CityLearn charges emissions at `t`. Fixed for B4 and all new methods.

## Experiment design

**Stage A (running): every method × seeds 42, 100, 2026**
- **Ranking:** methods are ranked in `99_compare_results` by the mean KPI-ratio score over cost, carbon, ramping, average daily peak, annual peak and 1 − load factor (lower is better).
- **Selection:** the top 3 of M1–M6 go to Stage B.

**Stage B: add seeds 7 and 1234**
- Run them for the top 3, plus B1, B3 and B4.
- To do this, add the seeds to `SEEDS` in those notebooks and Run All. Finished seeds are skipped.
- This can go to friends: each one takes a notebook, then sends back their `results/<METHOD>/seed_*` folders.

**Report**
- Mean ± 95% CI of every KPI ratio.
- Paired difference vs B3 for each seed.
- Cost-vs-carbon and KPI plots (`99_compare_results`).

## Running notebooks

- **In Jupyter:** use the "Python (SmartGridRL venv)" kernel, then Run All.
- **Output layout:** `results/<METHOD>/seed_<s>/` holds `checkpoint.zip/json` (written every 20k steps), `model.zip`, `train_log.csv`, `eval_trace.csv` and `summary.json`.
- **Resume behaviour:** a seed with `summary.json` is finished and skipped. Anything else resumes from its last checkpoint.
- **Memory:** each run reserves about 1 GB of commit memory, mostly from PyTorch. Starting many Jupyter kernels at once crashed with a ZeroMQ error. The Stage A jobs were therefore started as plain Python processes that execute the notebooks' code cells, with live logs in `results/logs/<METHOD>.log`. Opening a notebook afterwards and clicking Run All just loads the finished results.
