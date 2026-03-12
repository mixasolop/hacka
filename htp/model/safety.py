"""Centralized numerical safety helpers."""

from __future__ import annotations

import math

import numpy as np

from .bounds import CO2_max_hard, CO2_min, temperature_clamp_max_c, temperature_clamp_min_c
from .constants import REFERENCE_TEMP_C


def clamp(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    return min(max(float(x), float(lo)), float(hi))


def clamp01(x: float) -> float:
    return clamp(x, 0.0, 1.0)


def clamp_percent(x: float) -> float:
    return clamp(x, 0.0, 100.0)


def safe_float(x: object, default: float) -> float:
    try:
        value = float(x)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    try:
        return float(default)
    except (TypeError, ValueError):
        return 0.0


def ensure_finite(x: float, fallback: float) -> float:
    return safe_float(x, fallback)


def safe_log(x: float, min_value: float = 1e-30) -> float:
    return math.log(max(safe_float(x, min_value), min_value))


def safe_exp(x: float, clip_lo: float = -60.0, clip_hi: float = 60.0) -> float:
    return math.exp(clamp(safe_float(x, 0.0), clip_lo, clip_hi))


def sanitize_array(arr: np.ndarray | list[float], fallback: float = 0.0) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.size == 0:
        return out
    return np.where(np.isfinite(out), out, safe_float(fallback, 0.0))


def sanitize_state(T_c: float, CO2_ppm: float, albedo: float, H_pct: float) -> tuple[float, float, float, float]:
    temp = clamp(ensure_finite(T_c, REFERENCE_TEMP_C), temperature_clamp_min_c, temperature_clamp_max_c)
    co2 = clamp(ensure_finite(CO2_ppm, CO2_min), CO2_min, CO2_max_hard)
    alpha = clamp01(ensure_finite(albedo, 0.3))
    hab = clamp_percent(ensure_finite(H_pct, 0.0))
    return temp, co2, alpha, hab
