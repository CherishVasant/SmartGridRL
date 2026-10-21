"""Inputs and targets of the deep load / PV forecasters (pure numpy).

Single source of truth: the Colab notebook embeds this file verbatim, and the controller (DeepForecaster in the core)
imports it, so training and control build exactly the same vectors.

At decision time t, for every building of a district:
  input  = past week (168 h) of non-shiftable load and of PV, each divided by the building's own mean |x| over that week
           (scale-free, so a model trained on some buildings works on others), the current weather, the environment's
           weather forecasts for each of the next 24 hours (forecast issued at t+h-6 / t+h-12 / t+h-24, never later than
           t), the hour of each of the next 24 hours (sin / cos), the month (sin / cos) and the day type (one-hot).
  target = the next 24 hours of load and of PV, divided by the same scales.
Only data up to t is used (and forecasts already issued by t), so this is deployable.
"""
import numpy as np

W, H = 168, 24
WEATHER_SCALE = np.array([30.0, 100.0, 1000.0, 1000.0], dtype=np.float32)   # temperature, humidity, diffuse, direct
N_INPUTS = 2 * W + 4 + 4 * H + 2 * H + 2 + 7


def scale(x, t):
    """Per-building scale of series x (T, B) at time t: mean |x| over the past week, plus a floor."""
    return np.abs(x[max(t - W + 1, 0):t + 1]).mean(axis=0) + 0.05


def inputs(load, solar, weather, weather_pred, cal, t):
    """(B, N_INPUTS) inputs at time t and the per-building scales (load, PV). Arrays: load / solar (T, B),
    weather (T, 4), weather_pred (T, 4, 3) = forecasts issued at t for t+6 / t+12 / t+24, cal (T, 3) = month,
    day type (1..7), hour (1..24)."""
    B = load.shape[1]
    idx = np.clip(np.arange(t - W + 1, t + 1), 0, None)
    sl, ss = scale(load, t), scale(solar, t)
    past_l = (load[idx] / sl).T
    past_s = (solar[idx] / ss).T
    hs = np.arange(1, H + 1)
    lead = np.where(hs <= 6, 6, np.where(hs <= 12, 12, 24))
    slot = np.where(hs <= 6, 0, np.where(hs <= 12, 1, 2))
    issued = t + hs - lead
    w_fut = np.where((issued >= 0)[:, None], weather_pred[np.maximum(issued, 0), :, slot], weather[t][None])
    hour_next = (cal[t, 2] - 1 + hs) % 24
    shared = np.concatenate([weather[t] / WEATHER_SCALE, (w_fut / WEATHER_SCALE).ravel(),
                             np.sin(2 * np.pi * hour_next / 24), np.cos(2 * np.pi * hour_next / 24),
                             [np.sin(2 * np.pi * cal[t, 0] / 12), np.cos(2 * np.pi * cal[t, 0] / 12)],
                             np.eye(7)[int(cal[t, 1]) - 1]])
    X = np.concatenate([past_l, past_s, np.broadcast_to(shared, (B, len(shared)))], axis=1).astype(np.float32)
    return X, sl.astype(np.float32), ss.astype(np.float32)


def targets(load, solar, t, sl, ss):
    """(B, 2H) targets: next 24 h of load / sl, then of PV / ss."""
    return np.concatenate([(load[t + 1:t + H + 1] / sl).T, (solar[t + 1:t + H + 1] / ss).T], axis=1).astype(np.float32)
