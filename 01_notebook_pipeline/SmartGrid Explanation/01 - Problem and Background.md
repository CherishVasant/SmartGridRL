---
title: Problem and Background
tags:
  - smartgrid-rl
  - research-paper
  - background
aliases:
  - Problem Statement
  - Background
---

# 01 · Problem and Background

> [!abstract] One-line summary
> We train a reinforcement-learning agent to charge and discharge building batteries so that electricity **cost** and **carbon emissions** both go down. Our contribution is to make the carbon penalty in the reward **change with how dirty the grid is right now**, instead of keeping it fixed.

Next: [[02 - Methodology]] · [[03 - Results and Discussion]]

---

## 1. The problem in plain words

Buildings now have **rooftop solar panels** and **batteries**. Every hour, something has to decide what the battery does:

- **charge** it, using solar or electricity bought from the grid,
- **discharge** it to cover the building's own demand, or
- **do nothing**.

That decision affects two things:

| Objective | What drives it |
|---|---|
| **Operational cost** ($) | The electricity price at that hour (time-of-use tariff) |
| **Carbon emissions** (kg CO₂) | The grid's **marginal carbon intensity** at that hour: how much CO₂ each kWh pulled from the grid causes |

Price and carbon intensity **don't always move together**. At night, electricity can be cheap while the grid is running on fossil plants, so cheap and dirty. On a sunny or windy afternoon, the grid can be clean while the tariff is at its peak, so clean and expensive. A controller that only looks at price makes the wrong carbon decision in exactly those hours.

This is a **sequential decision problem under uncertainty**. Solar output, building demand, prices and carbon intensity all change hour to hour, and charging now changes what you can do later (state of charge). That makes it a natural fit for **reinforcement learning (RL)**: a model-free agent learns a policy directly from simulated interaction, without needing an exact model of the system.

### Exact problem we solve

> [!question] Research question
> Does **dynamic carbon-adaptive reward shaping** give a better cost-vs-emissions trade-off (Pareto frontier) than
> (a) a **cost-only** RL agent,
> (b) a **static-weighted** carbon-aware RL agent, and
> (c) a standard **rule-based controller**,
> while keeping training stable and sample-efficient?

---

## 2. What is normally done

### 2.1 Classical control
- **Rule-based controllers (RBC):** fixed schedules such as "charge at night, discharge in the evening". Simple, predictable and the industry default, but they ignore the actual price and carbon signals.
- **Model Predictive Control (MPC) / optimisation:** solve an optimisation over a forecast horizon every hour. Strong when forecasts are good, but expensive to run in real time and fragile when forecasts are wrong.

### 2.2 Reinforcement learning for energy
RL is widely used in power systems:

| Area | Typical method |
|---|---|
| Optimal power flow | Q-learning, policy gradients |
| **Building-level battery dispatch** | **PPO, SAC in CityLearn**, which is our setting |
| Microgrid coordination | Multi-agent RL |
| Voltage / frequency limits | Safe RL |
| Grid topology control | Graph RL |

### 2.3 Recent papers and what they miss
| Paper | What they do | Gap relevant to us |
|---|---|---|
| **Yu et al. (2026)**, *Energy Strategy Reviews* | LSTM forecaster + PPO dispatch; financial risk management | Reward is **purely financial**, with no carbon term. Forecast errors propagate into the policy. |
| **Alymani et al. (2026)**, *Sustainable Computing* | GNN state encoder + DRL + blockchain coordination, on an IEEE 33-bus system | **No explicit carbon term**, despite the sustainability framing. Non-standard metrics; small test system. |
| **Jeyaraj et al. (2026)**, *Applied Soft Computing* | Multi-agent DRL microgrid reconfiguration under cyberattacks, on an IEEE 300-bus system | Optimises voltage, load shedding and cost, with **no carbon objective**. Black box with no explainability. |

> [!important] Common pattern in the literature
> Carbon is treated as a **side effect**: "if we cut cost, carbon will probably fall too". It is rarely an explicit, first-class part of the reward.

### 2.4 When carbon *is* included: the static weighted reward
The few works that put carbon in the reward use a **fixed weight**:

$$
R_t = -\text{cost}_t - \lambda \cdot \text{carbon}_t, \qquad \lambda \text{ fixed before training}
$$

---

## 3. The gap we target

A fixed $\lambda$ is a blunt instrument:

- When the grid is **dirty** (fossil peaker plants online), avoiding grid electricity is worth a lot of CO₂.
- When the grid is **clean** (lots of wind or solar), grid electricity costs almost no CO₂, so the agent should focus on cost.

A static $\lambda$ applies **the same carbon penalty in both situations**, so the agent cannot learn to be *selectively* careful. It also has to be tuned by hand: too small does nothing, and too large dominates the reward and can break learning. We test this directly in the λ sweep in [[03 - Results and Discussion#2. Static λ sweep]].

---

## 4. What we do differently

> [!tip] Our idea: dynamic carbon-adaptive reward
> Make the carbon weight a **function of the grid's real-time carbon intensity** $I_t$:
>
> $$
> R_t = -\text{cost}_t - \lambda(I_t)\cdot \text{carbon}_t, \qquad \lambda(I_t) = \lambda_0 \cdot \hat I_t, \qquad \hat I_t = \frac{I_t - I_{\min}}{I_{\max}-I_{\min}} \in [0,1]
> $$

What this formula does:

- **Cleanest hour** ($\hat I_t = 0$): the reward becomes the **cost-only** reward. Grid power is clean, so only the price matters.
- **Dirtiest hour** ($\hat I_t = 1$): the reward becomes the **static** reward with weight $\lambda_0$, the maximum carbon penalty.
- **In between:** the penalty scales linearly with how dirty the grid is.

The same formula behaves as either baseline depending on the hour, so it adds **no new hyper-parameter** beyond $\lambda_0$. The unit tests in notebook 02 check that it reduces to cost-only and to static at the two extremes.

### Contributions for the paper
1. **Dynamic carbon-adaptive reward shaping** for battery dispatch, where the carbon weight follows the grid's real-time carbon intensity.
2. A **controlled comparison** of four controllers (RBC, Cost-Only PPO, Static PPO, Dynamic PPO). All use the same environment, PPO hyper-parameters, training budget and **3 random seeds**, so the reward is the only thing that changes.
3. A **λ-sensitivity study** of the static formulation, showing why a fixed weight is hard to tune.
4. A **post-hoc explainability analysis**: how each agent uses the battery in conflict states such as "cheap but dirty".

### Scope / non-goals
- ✅ Simulation study in **CityLearn**, focused on the reward formulation.
- ❌ Not trying to beat state-of-the-art controllers with a new network architecture.
- ❌ No hardware deployment, and no detailed electrical network constraints (voltages, line limits).

Continue to [[02 - Methodology]].
