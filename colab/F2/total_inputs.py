"""Inputs and targets of the deep DISTRICT-TOTAL load forecasters of colab/F2/F2_district_total.ipynb (pure numpy).

Single source of truth: the F2 notebook embeds this file verbatim and the controller imports it, so training and
control build exactly the same vectors.

At decision time t, for each total-load series (a district's summed non-shiftable load; in training, the sum of a
random subset of the other districts' buildings, a "pseudo-district"):
  input  = past two weeks (336 h) of the total, divided by its mean |x| over the past week (scale-free), the 28-day
           same-hour mean for each of the next 24 hours (scaled the same way), the current weather, the environment's
           weather forecasts for each of the next 24 hours (issued at t+h-6 / t+h-12 / t+h-24, never later than t),
           the hour of each of the next 24 hours (sin / cos), the month (sin / cos), today's day type and the day type
           of the target day one week earlier (one-hot each).
  target = the next 24 hours of the total, divided by the same scale.
Only data up to t is used (and forecasts already issued by t), so this is deployable.
"""
import numpy as np

W, H, D = 336, 24, 28
WEATHER_SCALE = np.array([30.0, 100.0, 1000.0, 1000.0], dtype=np.float32)   # temperature, humidity, diffuse, direct
N_INPUTS = W + H + 4 + 4 * H + 2 * H + 2 + 7 + 7


def scale(x, t):
    """Scale of series x (T, n) at time t: mean |x| over the past week, plus a floor."""
    return np.abs(x[max(t - 167, 0):t + 1]).mean(axis=0) + 0.05


def inputs(x, weather, weather_pred, cal, t):
    """(n, N_INPUTS) inputs at time t and the scales (n,). x: total-load series (T, n), weather (T, 4),
    weather_pred (T, 4, 3) = forecasts issued at t for t+6 / t+12 / t+24, cal (T, 3) = month, day type (1..7),
    hour (1..24). Needs t >= 0; history before hour 0 is padded with the first hour."""
    n = x.shape[1]
    s = scale(x, t)
    past = (x[np.clip(np.arange(t - W + 1, t + 1), 0, None)] / s).T                  # (n, W)
    hs = np.arange(1, H + 1)
    same = np.stack([x[np.clip(t + hs - 24 * d, 0, t)] for d in range(1, D + 1)]).mean(0) / s   # (H, n)
    lead = np.where(hs <= 6, 6, np.where(hs <= 12, 12, 24))
    slot = np.where(hs <= 6, 0, np.where(hs <= 12, 1, 2))
    issued = t + hs - lead
    w_fut = np.where((issued >= 0)[:, None], weather_pred[np.maximum(issued, 0), :, slot], weather[t][None])
    hour_next = (cal[t, 2] - 1 + hs) % 24
    day_target = int(cal[max(t + 24 - 168, 0), 1])
    shared = np.concatenate([weather[t] / WEATHER_SCALE, (w_fut / WEATHER_SCALE).ravel(),
                             np.sin(2 * np.pi * hour_next / 24), np.cos(2 * np.pi * hour_next / 24),
                             [np.sin(2 * np.pi * cal[t, 0] / 12), np.cos(2 * np.pi * cal[t, 0] / 12)],
                             np.eye(7)[int(cal[t, 1]) - 1], np.eye(7)[day_target - 1]])
    X = np.concatenate([past, same.T, np.broadcast_to(shared, (n, len(shared)))], axis=1).astype(np.float32)
    return X, s.astype(np.float32)


def targets(x, t, s):
    """(n, H) targets: next 24 h of the total / scale."""
    return (x[t + 1:t + H + 1] / s).T.astype(np.float32)

