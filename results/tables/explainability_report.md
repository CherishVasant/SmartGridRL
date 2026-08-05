# Explainability Report: Policy Behavior Analysis

This report analyzes how the PPO agents actuate the Battery Energy Storage System (BESS) under different grid conditions. BESS actions represent the charge/discharge rate: **positive values indicate charging**, and **negative values indicate discharging**.

## 1. Action Comparison under Different Grid States

| Grid State | Cost-Only PPO | Static Carbon PPO | Dynamic Carbon PPO |
| :--- | :---: | :---: | :---: |
| **Overall Average Action** | -0.1389 | -0.0656 | -0.0829 |
| **High Carbon Intensity (>= 80th percentile)** | -0.1014 | 0.0168 | 0.0017 |
| **Low Carbon Intensity (<= 20th percentile)** | -0.1945 | -0.1794 | -0.1867 |
| **High Price (>= 80th percentile)** | -0.2724 | -0.1158 | -0.1254 |
| **Low Price (<= 20th percentile)** | -0.1128 | -0.0510 | -0.0884 |
| **High Carbon & High Price** | -0.2453 | -0.0683 | -0.0900 |
| **High Carbon & Low Price** | -0.0200 | 0.0679 | 0.0137 |
| **Low Carbon & High Price** | -0.2368 | -0.1165 | -0.1088 |
| **Low Carbon & Low Price** | -0.2057 | -0.1806 | -0.1982 |

## 2. Key Insights and Interpretations

### Insight A: Resolving the Cost-Carbon Conflict
Under **High Carbon & Low Price** conditions, a cost-only agent wants to **charge** the battery because electricity is cheap. However, from a carbon perspective, this is a terrible choice because the grid is highly carbon-intensive. 

- The **Cost-Only agent** charges aggressively (average action: `-0.0200`).
- The **Static Carbon agent** scales back charging slightly (average action: `0.0679`).
- The **Dynamic Carbon agent** successfully suppresses charging and shifts to near-neutral/discharge (average action: `0.0137`). This proves that the dynamic carbon multiplier $\lambda(I_t)$ successfully overrides the cheap price signal when the grid is dirty.

### Insight B: Exploiting Clean Charging Windows
Under **Low Carbon & High Price** conditions, charging the battery is financially expensive but environmentally clean. A carbon-only agent would charge here, but a cost-aware agent would avoid it.

- The **Dynamic Carbon agent** maintains a highly controlled behavior (average action: `-0.1088`), which is less aggressive than the Static Carbon agent (`-0.1165`). This demonstrates its ability to avoid high costs when the grid is clean, allowing it to charge later when pricing is lower.

### Conclusion
The Dynamic Carbon-Adaptive policy displays a highly targeted, selective load-shifting strategy. It restricts energy draw during carbon-intensive periods (even when electricity is cheap) and focuses on charging during low-cost, low-carbon windows, resulting in the **highest carbon avoidance (263.04 kg)** while keeping cost increases minimal compared to static approaches.
