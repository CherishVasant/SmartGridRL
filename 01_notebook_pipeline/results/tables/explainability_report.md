# Explainability Report: Policy Behavior Analysis

Battery actions of the PPO agents (building 0, evaluation seed 42) under different grid conditions.
**Positive = charging**, **negative = discharging**. "High"/"low" = at or above the 80th / at or below the 20th percentile.

## 1. Mean action per grid state

| Grid state               |   hours |   Cost-Only |   Static |   Static-Matched |   Dynamic |
|:-------------------------|--------:|------------:|---------:|-----------------:|----------:|
| Overall                  |    8759 |     -0.1389 |  -0.0656 |          -0.1512 |   -0.0829 |
| High carbon              |    1752 |     -0.1501 |  -0.0261 |          -0.135  |   -0.042  |
| Low carbon               |    1752 |     -0.1362 |  -0.1217 |          -0.1945 |   -0.1348 |
| High price               |    1825 |     -0.3852 |  -0.2373 |          -0.3287 |   -0.2416 |
| Low price                |    4617 |     -0.0854 |  -0.0228 |          -0.108  |   -0.0585 |
| High carbon & high price |     669 |     -0.3609 |  -0.1961 |          -0.315  |   -0.2099 |
| High carbon & low price  |     289 |     -0.0348 |   0.0634 |          -0.0082 |    0.0141 |
| Low carbon & high price  |      71 |     -0.349  |  -0.1988 |          -0.303  |   -0.2012 |
| Low carbon & low price   |    1323 |     -0.1428 |  -0.1219 |          -0.1963 |   -0.143  |

## 2. Conflict state: high carbon & low price (289 hours)
Electricity is cheap but the grid is dirty, so cost and carbon pull in opposite directions.

- Cost-Only: `-0.0348`
- Static: `0.0634`
- Static-Matched: `-0.0082`
- Dynamic: `0.0141`

Dynamic's mean action is higher than Cost-Only, lower than Static, higher than Static-Matched
in this state (lower = less charging / more discharging).

## 3. Clean but expensive: low carbon & high price (71 hours)

- Cost-Only: `-0.3490`
- Static: `-0.1988`
- Static-Matched: `-0.3030`
- Dynamic: `-0.2012`

## 4. Emissions avoided vs RBC (mean over seeds)

- Cost-Only: 198.30 kg
- Static: 106.18 kg
- Static-Matched: 305.15 kg
- Dynamic: 263.04 kg

Largest carbon avoidance: **Static-Matched** (305.15 kg).
