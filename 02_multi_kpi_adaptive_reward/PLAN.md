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
