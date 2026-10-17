---
title: Results and Discussion
tags:
  - smartgrid-rl
  - research-paper
  - results
aliases:
  - Results
  - Findings
---

# 03 · Results and Discussion

Previous: [[01 - Problem and Background]] · [[02 - Methodology]]

> [!abstract] Headline
> - **All PPO agents beat the rule-based controller.** They lower cost by 3–4.5 %, lower carbon by 2–5.6 %, and raise solar self-consumption from 44 % to about 61 %.
> - **Dynamic beats Static λ = 1 on carbon on every seed** (≈ −150 kg).
> - **The matched baseline shows why.** A *static* agent given the **same average carbon weight** as Dynamic (λ = 0.41, "Static-Matched") emits **even less** than Dynamic, on **all 3 seeds**, at about the same cost.
> - **So in this setup, Dynamic's advantage comes from a smaller average carbon weight, not from adapting the weight to grid carbon intensity.** The core hypothesis is **not supported yet**; see [[#5. What this means for the paper]].

Run date: 2026-10-06/08. The 4 original methods reproduce the July numbers exactly; only the self-consumption metric changed, because its bug was fixed. Source files: `01_notebook_pipeline/results/tables/`.

---

## 1. Main comparison (3 seeds: 42, 100, 2026)

| Method | Weight on carbon | Cost ($) | Emissions (kg CO₂) | CO₂ avoided vs RBC | Cost vs RBC | Emissions vs RBC | Self-consumption |
|---|---|---|---|---|---|---|---|
| RBC | — | 4,886.44 ± 0.00 | 5,457.33 ± 0.00 | 0 | 0 % | 0 % | 43.9 % |
| Cost-Only PPO | 0 | 4,736.21 ± 188.38 | 5,259.02 ± 278.93 | 198.3 kg | −3.07 % | −3.63 % | 61.6 % |
| Static PPO | 1.0 (fixed) | **4,669.47 ± 125.42** | 5,351.14 ± 202.44 | 106.2 kg | **−4.44 %** | −1.95 % | 61.1 % |
| **Static-Matched PPO** | 0.41 (fixed) | 4,706.48 ± 206.71 | **5,152.17 ± 243.46** | **305.2 kg** | −3.68 % | **−5.59 %** | **62.7 %** |
| Dynamic PPO (proposed) | 0 → 1.0, avg 0.41 | 4,712.11 ± 206.35 | 5,194.28 ± 210.61 | 263.0 kg | −3.57 % | −4.82 % | 61.2 % |

*(mean ± std over 3 seeds; RBC is deterministic so its std is 0)*

**Static-Matched** is the fair baseline for Dynamic. With λ₀ = 1, Dynamic's weight is $\lambda_0 \hat I_t$, and the mean of $\hat I_t$ over the year is **0.4075**, computed in notebook 03, Part E. A static agent with λ = 0.41 therefore applies the **same carbon pressure on average**. The only difference is that Dynamic concentrates that pressure in dirty hours.

### Per seed: carbon (kg CO₂)

| Seed | RBC | Cost-Only | Static (λ = 1) | Static-Matched (λ = 0.41) | Dynamic | Dynamic − Static | **Dynamic − Static-Matched** |
|---|---|---|---|---|---|---|---|
| 42 | 5,457 | 5,094 | 5,258 | **5,042** | 5,097 | −161 | **+55** |
| 100 | 5,457 | 5,581 | 5,583 | **5,431** | 5,436 | −147 | **+5** |
| 2026 | 5,457 | 5,102 | 5,212 | **4,983** | 5,050 | −162 | **+67** |

### Per seed: cost ($)

| Seed | RBC | Cost-Only | Static | Static-Matched | Dynamic |
|---|---|---|---|---|---|
| 42 | 4,886 | 4,797 | 4,726 | 4,636 | 4,898 |
| 100 | 4,886 | 4,886 | 4,756 | 4,939 | 4,749 |
| 2026 | 4,886 | 4,525 | 4,526 | 4,544 | 4,490 |

### Cost–carbon Pareto plot
![[pareto_frontier.png]]
*Large markers = mean ± std; small faded markers = individual seeds. Lower-left is better. Static-Matched (purple) is left of Dynamic (green) at almost the same cost.*

---

## 2. Static λ sweep

Static reward $R = -\text{cost} - \lambda\,\text{carbon}$, seed 42 only:

| λ | Cost ($) | Carbon (kg) |
|---|---|---|
| 0.2 | 4,827.8 | **4,978.8** |
| 0.4 | 4,791.2 | 5,046.9 |
| 0.41 (Static-Matched) | **4,636.4** | 5,042.1 |
| 0.6 | 4,838.5 | 5,111.2 |
| 0.8 | 4,865.2 | 5,053.0 |
| 1.0 | 4,726.2 | 5,258.1 |
| 5.0 | 5,315.2 | 5,335.4 |
| *Dynamic (λ₀ = 1, avg 0.41)* | *4,897.8* | *5,096.9* |

![[static_lambda_sweep.png]]

What the sweep shows:
- **λ = 5 collapses:** cost is worse than RBC and carbon gets worse too. A dominant carbon term breaks learning.
- **λ = 1, used in the main comparison, is the worst carbon result of all λ ≤ 1.** It was never a "well-tuned" baseline.
- **Every static λ from 0.2 to 0.8 emits less than Dynamic on seed 42.**
- λ = 0.4 and λ = 0.41 give similar carbon (5,047 vs 5,042) but very different cost (4,791 vs 4,636). A single seed is noisy; differences under about 100 kg or $150 shouldn't be over-interpreted.

> [!warning] This changes an earlier conclusion
> `RESULTS.md` (July) called λ = 1 "well-tuned" and reported a sweep that can't be reproduced by the current code. Use the sweep above.

---

## 3. Training behaviour

![[training_curves_cost_carbon.png]]

- All PPO agents learn steadily and similarly. Training-episode cost and carbon fall over the ~5 training episodes. The carbon term doesn't destabilise training at λ ≤ 1.
- **The curves are still falling at 50k steps, so the agents are not converged.**
- Training episodes use the stochastic, exploring policy (cost ≈ $6,300–6,400, carbon ≈ 8,400 kg). That is much worse than the deterministic policy used for evaluation (≈ $4,700, ≈ 5,200 kg).

![[training_curves.png]]
*Reward scales differ between methods; compare shapes, not levels.*

---

## 4. Explainability: how the agents use the battery

Mean battery action for building 0 on seed 42 (positive = charge, negative = discharge; high = ≥ 80th percentile, low = ≤ 20th):

| Grid state | Hours | Cost-Only | Static (λ = 1) | Static-Matched | Dynamic |
|---|---|---|---|---|---|
| Overall | 8,759 | −0.139 | −0.066 | −0.151 | −0.083 |
| High carbon | 1,752 | −0.150 | −0.026 | −0.135 | −0.042 |
| Low carbon | 1,752 | −0.136 | −0.122 | −0.195 | −0.135 |
| High price | 1,825 | −0.385 | −0.237 | −0.329 | −0.242 |
| Low price | 4,617 | −0.085 | −0.023 | −0.108 | −0.059 |
| High carbon & high price | 669 | −0.361 | −0.196 | −0.315 | −0.210 |
| **High carbon & low price** (conflict) | 289 | −0.035 | **+0.063** | −0.008 | +0.014 |
| Low carbon & high price | 71 | −0.349 | −0.199 | −0.303 | −0.201 |
| Low carbon & low price | 1,323 | −0.143 | −0.122 | −0.196 | −0.143 |

![[explainability_actions_by_state.png]]

What this shows:
- **Price drives behaviour more than carbon does** for every agent. All of them discharge hardest when the price is high.
- **Dynamic closely resembles Static λ = 1** in most rows (e.g. high price −0.242 vs −0.237; low carbon −0.135 vs −0.122). It does **not** look like a policy that switches behaviour between clean and dirty hours. Its policy is closer to a "cautious" static agent.
- **Static-Matched uses the battery much more** (overall −0.151, close to Cost-Only). That fits its lower cost and carbon: active discharging displaces grid imports.
- In the conflict state (cheap but dirty), Static λ = 1 is the only agent charging noticeably (+0.063). Dynamic is near neutral (+0.014), and Static-Matched discharges slightly.

Caveats: these are **requested** actions (CityLearn limits them by battery capacity and state of charge), for one building and one seed. The low-carbon & high-price state has only 71 hours.

> [!warning] The July explainability claims don't hold
> The old traces logged the observation one hour after each action. With that fixed, the old story ("Cost-Only charges aggressively in dirty-cheap hours, Dynamic suppresses it") no longer appears in the data.

---

## 5. What this means for the paper

> [!danger] The core hypothesis is not supported in the current setup
> Dynamic carbon-adaptive shaping (λ₀ = 1, linear $\phi$) does **not** beat a static reward with the same average weight. Static-Matched has lower carbon on 3/3 seeds (by 5–67 kg) at similar cost. Dynamic's apparent win over Static λ = 1 is explained by the **average weight** (0.41 vs 1.0), not by the **adaptivity**.

Likely reasons the adaptivity doesn't help (yet):
1. **Too little training.** At 50k steps (≈ 5.7 years) the curves are still falling. The dynamic reward's signal changes with time of day, which is harder to learn than a fixed one, so it may need more training to pay off.
2. **Price and carbon move together.** Their correlation is about 0.31 in this dataset, and true conflict hours (cheap & dirty) are only 289 of 8,760, about 3 %. There's little room for a dynamic weight to make different decisions than a static one.
3. **The policy can't see the weight.** The agent observes carbon intensity, but the reward's weight changes from hour to hour, so value estimates get noisier. A static weight gives a cleaner learning signal.
4. **Linear $\phi$ is mild.** $\lambda(\hat I) = \lambda_0 \hat I$ hardly separates dirty hours from average ones. A sharper (convex or step) $\phi$ would make the adaptivity matter more.
5. **Seed variance.** Std across seeds (≈ 200–280 kg) is far larger than the Dynamic vs Static-Matched gap (5–67 kg). Neither is shown to be *significantly* better with 3 seeds. "Static-Matched ≥ Dynamic" is consistent across seeds, but small.

### Honest framings for a paper
- **Option A, negative/ablation result:** "Carbon-aware reward shaping reduces emissions vs cost-only and RBC, but its benefit comes from the *magnitude* of the average carbon weight, not from time-varying it. A properly matched static baseline is essential, and omitting it overstates dynamic shaping." This is publishable as a careful empirical study, and the matched-baseline lesson is useful to the field.
- **Option B, make the method work first:** run the experiments in §7 (longer training, sharper $\phi$, a λ₀ sweep), then compare the **Pareto curves** of dynamic vs static across weights. Claim superiority only if Dynamic's curve dominates.

Either way, **don't claim** "dynamic beats static" based on the Static λ = 1 comparison. A reviewer would ask for the matched baseline, and that baseline wins.

---

## 6. Findings

1. ✅ **RL beats the rule-based controller:** 3–4.5 % lower cost, 2–5.6 % lower carbon, self-consumption 44 % → about 61 %.
2. ✅ **Adding carbon to the reward helps when the weight is moderate.** Static-Matched (λ = 0.41) has the best carbon of all methods (−5.6 % vs RBC, −2.0 % vs Cost-Only).
3. ✅ **Static carbon shaping is sensitive to λ.** λ = 5 is worse than RBC, and λ = 1 is worse on carbon than every λ from 0.2 to 0.8.
4. ⚠️ **Dynamic beats Static λ = 1 on all seeds, but that comparison is confounded** (average weight 0.41 vs 1.0).
5. ❌ **Dynamic does not beat Static-Matched:** it emits 5–67 kg more on every seed, at similar cost.
6. ❌ **The old "100 % self-consumption" finding was a bug.** The real values are 44 % (RBC) and 55–69 % (PPO).

## 7. Next steps (to make or break the dynamic method)

- [ ] **Longer training** (200k–500k steps, or until the curves flatten) for Static-Matched and Dynamic. This is the cheapest test of reason 1.
- [ ] **λ₀ sweep for Dynamic** (e.g. 0.5, 1, 2, 3) and a plot of the dynamic Pareto curve vs the static λ curve on all seeds. This is the fair, weight-independent comparison.
- [ ] **Sharper scaling functions:** $\phi(\hat I) = \hat I^2$, or a step function (penalise only the top 20–30 % dirtiest hours).
- [ ] **More seeds (≥ 5)** with paired differences and confidence intervals.
- [ ] **A dataset with more price/carbon conflict**, where dynamic weighting has more to exploit.
