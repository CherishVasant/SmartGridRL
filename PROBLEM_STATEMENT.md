# Carbon-Adaptive Reward Shaping for Smart Grid Energy Dispatch

## 1. Problem Statement

Modern energy grids are experiencing a paradigm shift. The integration of distributed energy resources (DERs) such as rooftop solar photovoltaics (PV), local wind generation, and battery energy storage systems (BESS) transforms traditional consumption points into active grid participants (prosumers). Managing these assets requires making high-frequency, sequential routing decisions under significant uncertainty: local renewable generation is highly intermittent, customer load demand is stochastic, and grid electricity prices and marginal carbon intensities vary continuously. 

This challenge is a classic sequential decision-making problem under uncertainty. Traditional optimization techniques (e.g., Model Predictive Control) often struggle with non-linear dynamics, stochastic forecasting errors, and high-dimensional states without computationally expensive real-time optimization. Reinforcement Learning (RL) presents a natural alternative, as model-free RL agents can learn optimal policy mappings directly from interaction data, adapting to complex environmental dynamics.

The primary objective of this project is to develop and evaluate an RL agent that manages smart grid energy dispatch (specifically coordinating local generation, BESS state-of-charge, and grid interaction) to simultaneously minimize both **operational cost** (electricity bills) and **carbon emissions** (from grid electricity consumption).

---

## 2. Existing Research & Critique

Reinforcement learning has been applied extensively to energy management systems. However, existing methodologies exhibit key limitations in addressing carbon intensity dynamically.

### Yu et al. (2026) — *Energy Strategy Reviews*
*   **Summary:** This work proposes a joint framework combining an LSTM-based load and price forecasting model with a Proximal Policy Optimization (PPO) dispatch agent. It frames the objective around financial risk-management in volatile electricity markets.
*   **Critique:** 
    1.  *Cost-Centric Objective:* The reward function focus is exclusively financial (maximizing cost efficiency and reducing risk), completely omitting any carbon-intensity or emissions metrics.
    2.  *Modular Architecture Vulnerability:* The modular design (forecasting feeding into dispatch) means that errors in the LSTM forecast propagate directly into the PPO dispatch policy, potentially degrading performance compared to end-to-end model-free RL architectures.

### Alymani et al. (2026) — *Sustainable Computing*
*   **Summary:** The authors utilize Graph Neural Networks (GNNs) to encode complex electrical grid topology into low-dimensional state representations. A Deep Reinforcement Learning (DRL) agent manages node dispatch, coordinated across the grid via a blockchain consensus layer (Hyperledger Fabric) to ensure trust.
*   **Critique:**
    1.  *Evaluation Comparability:* The paper introduces highly specific, non-standard evaluation metrics tied to blockchain consensus and GNN stability, making direct comparisons to existing literature difficult.
    2.  *Scalability and Validation:* Validation is restricted to a small IEEE 33-bus system, leaving questions about scalability unanswered.
    3.  *Absence of Emissions Term:* Despite being published in a sustainable computing venue, the reward function lacks any explicit carbon objective, relying on general efficiency as a proxy for sustainability.

### Jeyaraj et al. (2026) — *Applied Soft Computing*
*   **Summary:** This paper details a Multi-Agent Deep Reinforcement Learning (MADRL) framework designed for dynamic microgrid reconfiguration under cyberattacks. It is validated using New York ISO data on a large-scale IEEE 300-bus system.
*   **Critique:**
    1.  *Static Security Scenarios:* While the resilience focus is valuable, the simulated cyberattacks (e.g., denial of service, false data injection) are static and pre-defined, rather than adaptive or dynamic.
    2.  *No Carbon Objective:* The multi-agent formulation optimizes for voltage stability, load shedding minimization, and operational cost, but does not formulate carbon reduction as a target.
    3.  *Black-Box Limitations:* The system lacks any explainability or interpretability layers, which is a major barrier to real-world grid adoption where operators must trust dispatch actions.

### General Survey Findings
Across the literature, RL is applied to power systems in several paradigms:
*   *Optimal Power Flow:* Handled via Q-learning and policy gradient algorithms.
*   *Building-Level Dispatch:* Frequently validated in environments like CityLearn using PPO and Soft Actor-Critic (SAC).
*   *Microgrid Coordination:* Solved using Multi-Agent RL (MARL).
*   *Safe RL:* Employed to handle voltage and frequency constraints.
*   *Topology Control:* Resolved using Graph RL.

Despite this breadth of applications, **carbon emissions are consistently treated as a secondary or implicit metric** (e.g., assuming cost minimization correlates with carbon reduction) rather than a first-class, dynamically-weighted optimization term.

---

## 3. Identified Gap

In the limited literature where carbon emissions *are* explicitly optimized, the standard approach relies on a static weighted reward formulation:

$$R_t = -\text{cost}_t - \lambda \cdot \text{carbon}_t$$

where $\lambda \ge 0$ is a fixed scalar hyperparameter chosen prior to training. 

This static formulation is a blunt instrument. In actual electricity grids, marginal carbon intensity (the carbon emissions generated per unit of electricity by the marginal generator currently online) fluctuates dramatically. 
*   **High-Intensity Periods:** When the grid is stressed and fossil-fuel peaker plants (coal, natural gas) are activated, the marginal carbon intensity is extremely high.
*   **Low-Intensity Periods:** During high renewable penetration (high wind/solar output), grid power is clean, and marginal intensity drops toward zero.

A static $\lambda$ fails to account for these dynamics. It penalizes grid draw equally at all times, preventing the agent from learning to be **selectively aggressive**—meaning the agent should be highly averse to grid draw when the grid is dirty, but flexible and cost-oriented when the grid is clean.

---

## 4. Proposed Solution: Dynamic Carbon-Adaptive Reward Shaping

We propose a dynamic carbon-adaptive reward shaping mechanism where the trade-off coefficient $\lambda$ is formulated as a function of the real-time marginal carbon intensity of the grid ($I_t$):

$$R_t = -\text{cost}_t - \lambda(I_t) \cdot \text{carbon}_t$$

where $\lambda(I_t)$ dynamically adjusts based on the grid's carbon intensity at step $t$.

### Function Formulation
Let $\hat{I}_t$ be the normalized marginal carbon intensity at timestep $t$, scaled to $[0, 1]$ relative to historical grid bounds $[I_{\min}, I_{\max}]$:

$$\hat{I}_t = \frac{I_t - I_{\min}}{I_{\max} - I_{\min}}$$

1.  **Linear / Piecewise Linear Control (Base Model):**
    We define $\lambda(\hat{I}_t)$ as:
    
    $$\lambda(\hat{I}_t) = \lambda_0 \cdot \phi(\hat{I}_t)$$
    
    where $\lambda_0$ is a base carbon penalty weight, and $\phi(\hat{I}_t)$ is an scaling function (e.g., a simple linear scale $\phi(\hat{I}_t) = \hat{I}_t$ or a step-wise function). This ensures that when grid carbon intensity is maximum ($\hat{I}_t = 1$), the penalty on carbon is maximized, forcing the agent to discharge local BESS or curtail load. Conversely, when the grid is clean ($\hat{I}_t = 0$), the carbon penalty drops, allowing the agent to focus purely on cost-optimal grid-charging.
    
2.  **Adaptive / Learning-Based (Stretch Goal):**
    If time permits, we will explore modeling $\lambda(\cdot)$ via a secondary Contextual Bandit or a lightweight meta-policy network that adjusts the trade-off coefficient based on the current state and performance feedback.

---

## 5. Research Question

> **Does dynamic carbon-adaptive reward shaping produce a superior cost-vs-emissions Pareto frontier compared to (a) a cost-only agent, (b) a static-weighted carbon-aware agent, and (c) a standard rule-based controller—while maintaining comparable training stability and sample efficiency?**

---

## 6. Evaluation Plan

To rigorously evaluate the proposed method for both the Reinforcement Learning and Climate/Global Warming courses, we will measure:

### Reinforcement Learning Metrics
*   **Learning Curves:** Episode reward plotted against training steps across multiple random seeds to inspect convergence speed and policy stability.
*   **Sample Efficiency:** The number of steps/episodes required to reach asymptotic performance.
*   **Variance / Stability:** Standard deviation of episode rewards across seeds.

### Climate and Environmental Metrics
*   **CO2-equivalent Avoided ($kg$):** Cumulative emissions reduction compared to the Rule-Based Controller (RBC) baseline.
*   **Renewable Self-Consumption Rate:** The fraction of local solar/wind generation consumed locally rather than exported to the grid.
*   **Pareto Frontier:** A scatter plot mapping the operational cost ($y$-axis) against carbon emissions ($x$-axis) for each model variant. A superior model will push the Pareto frontier downward and to the left.

### Post-Hoc Explainability (Secondary Contribution)
To address the "black-box" critique, we will build a post-hoc analysis script. After training, trace data (states, actions, grid carbon intensity, and battery actions) will be logged. A subset of critical decision points (e.g., highest carbon avoidance periods) will be fed to a Large Language Model (LLM) using detailed templates to generate natural-language justifications, such as:
> *"The agent chose to discharge the BESS at 6:00 PM because the grid's marginal carbon intensity peaked at 720g CO2/kWh due to low solar output, successfully avoiding 4.2kg of CO2 emissions."*

This explainability layer will remain strictly detached from training to ensure RL remains the primary computational contribution.

---

## 7. Scope and Non-Goals

### In-Scope
*   A clean, reproducible, and controlled ablation study comparing the four agents: RBC, Cost-Only PPO, Static-Weighted Carbon PPO, and Dynamic Carbon-Adaptive PPO.
*   Running all experiments across at least 3 random seeds using identical environments and PPO hyperparameters.
*   Generating a standardized results suite containing Pareto plots, training curves, and performance tables.

### Non-Goals
*   **Beating SOTA Dispatch:** The goal is not to engineer a novel neural network architecture that outperforms state-of-the-art power grid controllers in absolute terms.
*   **Real-time Control Deployment:** The agents are evaluated in simulation (CityLearn) and are not intended for direct hardware-in-the-loop power systems dispatch.
*   **Dynamic Grid Security Constraints:** We do not model high-fidelity electrical network constraints (e.g., line impedance, reactive power flow, bus voltage limits).
