---
title: SmartGridRL – Methods and Results Report
tags:
  - smartgrid
  - reinforcement-learning
  - citylearn
  - battery-control
  - report
created: 2026-10-21
dataset: CityLearn Challenge 2022 (17 buildings)
source: "[[all_methods_results.csv]]"
---

# SmartGridRL – Methods and Results Report

> [!abstract] In one paragraph
> We control the home batteries of a district of buildings (CityLearn Challenge 2022) so that the electricity bill,
> the carbon emissions and the strain on the grid go down. We tried **reinforcement learning** (PPO, SAC, TQC, CrossQ),
> **model predictive control** (an optimiser driven by forecasts) and a **hybrid where RL tunes the optimiser**. Every
> method below except the last section is *deployable*: it only uses information a real building would have at that
> moment. The best deployable controller, the **RL-tuned MPC with LightGBM forecasts**, cuts the bill by about 34% and
> emissions by about 22% on the training buildings, and reaches a Phase III public score of **0.782** (published winner
> 0.728).

All numbers come from [[all_methods_results.csv]] (also `all_methods_results.xlsx`). Results with 3 seeds are the mean
of seeds 42, 100 and 2026; single-seed results use seed 42.

## 0. Target: the published leaderboard

> [!important] The score to beat
> The fair target is the **Phase III public score** of the NeurIPS 2022 CityLearn Challenge (lower is better):
> **0.728** by the winner, *Together*. Matching the top 3 needs **≤ 0.753**, the top 10 **≤ 0.797**.
> **Our best deployable controller: 0.782** (RL-tuned MPC with LightGBM forecasts) → between 7th and 8th place.

**Phase III public leaderboard** (0.4 × training buildings + 0.6 × unseen validation buildings; the official ranking)

| Rank | Team | Score | Cost C | Carbon G | Grid D |
|---|---|---|---|---|---|
| 1 | Together | **0.728** | 0.639 | 0.808 | 0.738 |
| 2 | ambitiousengineers | 0.742 | 0.681 | 0.803 | 0.743 |
| 3 | CUFE | 0.753 | 0.665 | 0.804 | 0.789 |
| 4 | EVERGi_team | 0.756 | 0.643 | 0.826 | 0.800 |
| 5 | Orca | 0.760 | 0.659 | 0.836 | 0.784 |
| 6 | Greener | 0.764 | 0.663 | 0.821 | 0.809 |
| 7 | DivMARL | 0.767 | 0.653 | 0.805 | 0.842 |
| – | ***Ours: RL-tuned MPC (M131)*** | ***0.782*** | ***0.661*** | ***0.769*** | ***0.916*** |
| 8 | SCRG_UA | 0.789 | 0.666 | 0.880 | 0.820 |
| 9 | LA_gang | 0.796 | 0.665 | 0.852 | 0.873 |
| 10 | tymur_prorochenko | 0.797 | 0.693 | 0.861 | 0.838 |

Our C, G and D in this row are the same 0.4 / 0.6 blend of the training and validation districts (M131). We already beat
every listed team on **carbon** (0.769 vs 0.803 at best); the gap to the top is mainly the **grid KPI D**.

**Phase I leaderboard** (cost and carbon only, on the 5 public training buildings): tymur 0.631 · Together 0.635 ·
ROLEVT 0.651 · OneWo 0.698 · LA_gang 0.719 · DivMARL 0.728. Our best deployable Phase I is **0.723** (≈ 5th).

> [!warning] Why Phase I is not the fair target
> Phase I was scored on the same public year the teams were given, and at least one top team optimised on the actual
> future data there. Phase III adds buildings that no team had seen, so it measures a real-world controller. Our
> forecasters are stricter still: they are never trained on the buildings they forecast, while the published teams'
> forecasters were trained on the training buildings that make up 40% of the public score.

---

## 1. Metrics

> [!info] Every metric is a ratio to the same buildings **without a battery**: 1.00 = no change, **lower is better**.

| Metric | Meaning (one line) |
|---|---|
| **Cost (C)** | Electricity bill of the whole district; electricity exported to the grid earns nothing. |
| **Carbon (G)** | Carbon emissions of the electricity the district imports (kWh × carbon intensity of that hour). |
| **Ramping (R)** | How abruptly the district's grid load jumps from one hour to the next (sum of the hourly changes). |
| **Load factor (1 − L)** | How "peaky" the load is each month: 1 − average load ÷ peak load (lower = flatter). |
| **Grid (D)** | The grid KPI: the average of ramping and (1 − load factor). |
| **Phase I score** | mean(C, G): cost and carbon only; the challenge's Phase I leaderboard. |
| **Phase III score (S)** | mean(C, G, D): cost, carbon and grid; the challenge's final metric for one district. |
| **Public score** | 0.4 × S(5 training buildings) + 0.6 × S(5 unseen validation buildings); the published Phase III leaderboard. |
| **Private score** | 0.2 × S(train) + 0.3 × S(validation) + 0.5 × S(7 unseen test buildings); used to pick the winners. |
| **Mean ± std** | Average and spread over the random seeds (training repeated with different random starts). |
| **vs RBC** | The same KPIs relative to CityLearn's rule-based controller instead of no battery (an older reference). |
| **Deployable** | Uses only present / past data and the forecasts the environment provides – usable in a real building. |

> [!note] Buildings
> *Training district* = buildings 1–5 (`citylearn_challenge_2022_phase_1`), *validation* = 6–10 (`phase_2`),
> *test* = 11–17 (`phase_3`). Controllers are trained on the training district only; validation and test buildings
> are never seen during training.

---

## 2. References (no learning)

| Controller | What it is | Phase I | Phase III (S) | Cost C | Carbon G | Grid D |
|---|---|---|---|---|---|---|
| No battery | Batteries never used | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Rule-based controller (RBC) | CityLearn's fixed charge / discharge schedule by hour | 1.132 | 1.113 | 1.070 | 1.193 | 1.075 |

The fixed schedule ignores prices, carbon and solar, so it is **worse than no battery** on every KPI.

---

## 3. The methods

### 3.1 PPO – Proximal Policy Optimization

> [!info] What it is
> An **on-policy** RL algorithm: the agent plays a batch of hours with its current policy, then improves the policy
> with gradient steps that are *clipped* so it never changes too much at once. Stable and simple, but it can only learn
> from data collected by the current policy, so it needs a lot of experience.

We used PPO with several reward designs: cost only, cost + a static or carbon-intensity-weighted carbon term, multi-KPI
rewards (cost, carbon, ramping, peaks, battery wear) with fixed or **adaptive weights** (weights shift towards the KPI
that is furthest behind its target), and a squared-load penalty; both one central agent and one shared agent per building.

| Run | Reward / setup | Seeds | Phase I | S | C | G | D |
|---|---|---|---|---|---|---|---|
| B1 | cost only | 3 | 1.004 | 1.174 | 0.938 | 1.070 | 1.515 |
| B3 | cost + static carbon weight | 3 | 0.984 | 1.144 | 0.918 | 1.050 | 1.464 |
| B4 | cost + carbon weighted by carbon intensity | 3 | 1.007 | 1.181 | 0.942 | 1.072 | 1.530 |
| M2 | multi-KPI, fixed weights | 3 | 0.959 | 1.068 | 0.937 | 0.982 | 1.285 |
| M3 | multi-KPI, adaptive weights | 3 | 0.950 | 1.077 | 0.923 | 0.978 | 1.330 |
| M9 | cost + carbon + load² | 3 | **0.900** | **1.061** | **0.833** | **0.967** | 1.382 |
| M5 | one shared agent per building | 3 | 1.001 | 1.003 | 1.001 | 1.001 | 1.006 |

> [!warning] Result
> PPO cuts the bill a little (up to 17%) but **raises ramping a lot** (D ≈ 1.3–1.5), so its overall score is no better
> than leaving the batteries idle. The per-building version learns to do nothing at all (all KPIs ≈ 1.00).

### 3.2 SAC – Soft Actor-Critic

> [!info] What it is
> An **off-policy** actor-critic algorithm: every hour of experience goes into a replay buffer and is reused many
> times. A *critic* network learns how good each action is, an *actor* learns to pick good actions, and an **entropy
> bonus** keeps the actor exploring. Much more sample-efficient than PPO for continuous actions such as battery rates.

The biggest gains came from **reward design** with SAC, step by step:
- **Counterfactual reward** – reward = improvement over doing nothing in the same hour (lower-variance learning signal).
- **Score-aligned reward** – per-hour terms computed exactly like the official KPIs (district level, exports earn
  nothing, scaled by the no-battery case), weighted like the challenge score.
- **Carbon weight 2** – extra emphasis on emissions.

| Run | Change | Seeds | Phase I | S | C | G | D |
|---|---|---|---|---|---|---|---|
| M12 | multi-KPI reward (same as PPO M2) | 3 | 0.881 ± 0.020 | 0.982 | 0.831 | 0.931 | 1.184 |
| M13 | + reward normalisation | 3 | 0.966 ± 0.053 | 1.004 | 0.952 | 0.980 | 1.080 |
| M16 | residual on the RBC | 3 | 0.956 | 1.007 | 0.933 | 0.978 | 1.111 |
| M20 | counterfactual reward | 3 | 0.829 ± 0.003 | 0.916 | 0.781 | 0.877 | 1.091 |
| M26 | score-aligned reward | 3 | 0.786 ± 0.008 | 0.870 | 0.732 | 0.840 | **1.036** |
| M28 | score-aligned, Phase I weights | 3 | 0.779 ± 0.003 | 0.876 | 0.727 | 0.831 | 1.071 |
| **M27** | **score-aligned, carbon weight 2** | 3 | **0.768 ± 0.005** | **0.867 ± 0.005** | **0.720** | **0.817** | 1.064 |

Other SAC experiments (training length, network size, features, ensembles):

| Run | Change from M27 | Seeds | Phase I | S |
|---|---|---|---|---|
| M102 | trained 60k steps instead of 200k | 3 | 0.794 ± 0.016 | 0.882 |
| M109 | trained 400k steps | 3 | 0.786 ± 0.013 | 0.885 |
| M112 | + history features (yesterday's carbon / load as a forecast) | 3 | 0.771 ± 0.007 | 0.863 |
| M90 / M99 / M94 / M95 | 256×256 nets / 2 updates per step / 3-step returns / γ = 0.95 (60k steps) | 1 | 0.809 / 0.810 / 0.864 / 0.808 | 0.899 / 0.898 / 0.928 / 0.887 |
| M98 | one shared SAC agent per building | 1 | 1.000 | 1.000 |
| ER_M27 | average of the 3 trained M27 policies | – | 0.841 | 0.907 |

> [!success] Result
> SAC with the score-aligned reward and carbon weight 2 (**M27**) cuts the bill by **28.0%** and emissions by **18.3%**
> (3 seeds, very consistent). On the unseen validation buildings it keeps cost 0.722 and carbon 0.793. 200k training
> steps is the sweet spot; reward normalisation, bigger networks, per-building agents and policy averaging do not help.

### 3.3 TQC – Truncated Quantile Critics

> [!info] What it is
> An extension of SAC whose critics predict a whole **distribution** of future returns (25 quantiles per critic)
> instead of one number, and which **drops the most optimistic quantiles** before learning. This reduces the
> over-estimation of action values that makes SAC over-confident.

| Run | Setup | Seeds | Phase I | S | C | G | D | Unseen public |
|---|---|---|---|---|---|---|---|---|
| M110 | TQC, M27 settings | 3 | 0.768 ± 0.004 | **0.861 ± 0.003** | 0.720 | 0.816 | **1.046** | **0.853** |
| **M113** | **TQC + history features** | 3 | **0.763 ± 0.005** | 0.862 ± 0.006 | **0.709** | 0.816 | 1.060 | 0.854 |
| M114 | TQC, carbon weight 1 | 3 | 0.774 ± 0.006 | 0.866 | 0.721 | 0.826 | 1.051 | – |
| M123 | TQC + history, γ = 0.98 | 1 | 0.757 | 0.855 | 0.706 | **0.808** | 1.051 | – |

Tried on top of M113 without improvement (seed 42): more conservative critics, 2 updates per step, less exploration, Phase I-only reward, carbon weight 1.5, 256×256 networks, batch 512, richer / denser history features,
learning-rate decay to 300k steps, no or stronger battery-wear penalty (Phase I 0.762–0.783).

> [!success] Result
> TQC is the **best pure RL algorithm**: same cost and carbon as SAC, smoother grid load and the most consistent seeds.
> With history features (M113) the bill falls by **29.1%**; on unseen buildings the public score is **0.853–0.854**.

### 3.4 CrossQ

> [!info] What it is
> A recent off-policy algorithm (2024) that removes the *target networks* of SAC and instead uses **batch
> normalisation** in the critics, which makes learning faster and simpler.

| Run | Setup (60k steps, seed 42) | Phase I | S |
|---|---|---|---|
| M92 | CrossQ, 256×256 | 0.805 | 0.892 |
| M100 | CrossQ + history features | 0.812 | 0.890 |
| *control* | SAC M27 at 60k steps | 0.803 | 0.883 |

> [!note] Result
> No better than SAC at the same budget, so it was not pursued further.

### 3.5 Forecasting

> [!info] What it is
> Predicting each building's **load** and **solar generation** (and the grid's carbon intensity) for the next 24 hours,
> using only past data and the weather forecasts the environment provides. The forecasts feed the MPC below.

| Forecaster | How it works | Net-load error (train / val / test districts) |
|---|---|---|
| Seasonal (persistence) | 0.5 × same hour yesterday + 0.5 × same hour over the last week, plus the current error fading over the horizon | 2.65 / 3.28 / 3.82 kWh |
| **LightGBM** | Gradient-boosted trees on lags, calendar and weather forecasts; trained only on buildings *outside* the district it forecasts; scale-free features | **2.30 / 2.83 / 3.71** kWh |
| Deep (MLP / LSTM / Transformer) | Prepared for GPU training in Colab (`colab/deep_forecaster_colab.ipynb`) | *not yet trained* |

Solar is easy to forecast (R² 0.95–0.98); household **load is not** (R² 0.4–0.6, about 40% average error), and it is
the main limit: with *perfect* load forecasts the same MPC would score S = 0.633 instead of 0.81.

### 3.6 MPC – Model Predictive Control

> [!info] What it is
> An **optimiser, not a learner**. Every hour it forecasts the next 24 hours, solves a small linear program for the best
> battery schedule over that horizon (cost + carbon + a penalty on load ramps, with the real battery limits), applies
> only the first hour, and repeats one hour later with fresh data.

| Run | Setup | Phase I | S | C | G | D | Public | Private |
|---|---|---|---|---|---|---|---|---|
| MPC1 | seasonal forecasts, no ramp penalty | 0.774 | 0.974 | 0.706 | 0.842 | 1.374 | 0.958 | 1.083 |
| MPC9 | seasonal forecasts, ramp weight 2 | 0.745 | 0.811 | 0.691 | 0.798 | 0.944 | 0.808 | 0.856 |
| MPC15 | seasonal forecasts, ramp weight 3 | 0.744 | 0.808 | 0.697 | 0.791 | 0.934 | 0.807 | 0.856 |
| GBM-MPC | **LightGBM forecasts**, ramp weight 3 | 0.745 | 0.792 | 0.706 | 0.785 | **0.886** | **0.789** | **0.848** |

> [!success] Result
> A forecast-driven optimiser beats every pure RL controller on the official Phase III score (public 0.789 vs 0.853).
> The ramp penalty is essential (without it D = 1.37), and better forecasts help directly.

### 3.7 RL-tuned MPC (hybrid: forecasting + optimisation + RL)

> [!info] What it is
> The MPC above does the scheduling, and a **TQC or SAC agent sets the MPC's weights every hour** (how strongly to
> smooth the load, how much to weigh carbon, how much to value energy left in the battery) from district-level
> features: time, the 24-hour forecasts, the battery charge and recent forecast errors. Because the agent only sees
> district-level features, the same policy runs on districts of any size (5 or 7 buildings).

| Run | Forecasts | RL | Phase I | S | C | G | D | Public | Private |
|---|---|---|---|---|---|---|---|---|---|
| M129 | seasonal | TQC | 0.728 | 0.808 | **0.663** | 0.793 | 0.970 | 0.807 | 0.864 |
| M130 | seasonal | SAC | 0.735 | 0.810 | 0.679 | 0.791 | 0.960 | 0.809 | 0.866 |
| M132 | seasonal, grid-weighted reward | TQC | 0.747 | 0.810 | 0.703 | 0.791 | 0.934 | 0.808 | 0.866 |
| **M131** | **LightGBM** | **TQC** | **0.723** | 0.787 | 0.664 | **0.783** | 0.914 | **0.782** | 0.845 |
| **M135** | **LightGBM** | **SAC** | 0.724 | **0.786** | 0.665 | **0.783** | **0.909** | **0.782** | **0.844** |

> [!success] Result
> RL on top of the LightGBM-driven MPC gives the **best deployable results of the project** on every score: bill
> −33.6%, emissions −21.7%, smoother grid load (D 0.91), and on the unseen buildings public **0.782** / private
> **0.845**. TQC and SAC give the same result here; what matters is the forecasts and the reward.

---

## 4. Best results

> [!tip] Best deployable controller: **RL-tuned MPC with LightGBM forecasts (M131 TQC / M135 SAC)**
> | Score | Value | Meaning |
> |---|---|---|
> | Phase I (training buildings) | **0.723** | bill and emissions on average 27.7% lower than without a battery |
> | Phase III S (training buildings) | **0.786–0.787** | cost −33.6%, carbon −21.7%, grid 0.91 |
> | Phase III **public** | **0.782** | ≈ 8th on the published leaderboard (Together 0.728, ambitiousengineers 0.742, CUFE 0.753) |
> | Phase III **private** | **0.844–0.845** | including the 7 unseen test buildings |
> (single seed, 42)

| Category | Best method | Seeds | Phase I | S | Public | Private |
|---|---|---|---|---|---|---|
| RL-tuned MPC (hybrid) | M131 / M135 | 1 | **0.723** | **0.786** | **0.782** | **0.844** |
| MPC (no RL) | GBM-MPC, ramp weight 3 | – | 0.745 | 0.792 | 0.789 | 0.848 |
| Pure RL (most robust) | M113 TQC + history | 3 | 0.763 ± 0.005 | 0.862 | 0.854 * | – |
| Pure RL (best single run) | M123 TQC + history, γ 0.98 | 1 | 0.757 | 0.855 | – | – |
| PPO | M9 load² | 3 | 0.900 | 1.061 | – | – |
| Rule-based | RBC | – | 1.132 | 1.113 | – | – |

\* public score computed on training + unseen validation buildings (same formula).

> [!example] Published Phase III public leaderboard (top 10) for comparison
> Together 0.728 · ambitiousengineers 0.742 · CUFE 0.753 · EVERGi 0.756 · Orca 0.760 · Greener 0.764 · DivMARL 0.767 ·
> SCRG_UA 0.789 · LA_gang 0.796 · tymur 0.797. Our 0.782 sits between DivMARL and SCRG_UA. The published teams
> trained their forecasters on the public buildings that are also 40% of the public score; ours never see the
> buildings they forecast.

---

## 5. Not deployable (for reference only)

> [!failure] These use the future and could not run in a real building
> - **Perfect-foresight LP plan** (O1–O15): the same optimiser as the MPC, but given the *actual* whole year in
>   advance. An upper bound: best Phase I **0.629** (O3), best S **0.624** (O2).
> - **Residual RL on that plan** (M29–M32): RL correcting the perfect-foresight plan. Phase I 0.79.
> - **Imitation / cloning** (M33–M89): a network trained to copy the perfect-foresight plan (behaviour cloning, DAgger).
>   Phase I down to **0.639** (M71) and S **0.641** (M69) on the training buildings, but only 0.774 on the unseen
>   buildings' public score, and it is only possible because the plan was computed with future data.
>
> They show how much headroom exists, but they are excluded from every claim above.
