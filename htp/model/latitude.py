"""Latitudinal profile helpers and habitability computation."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .constants import (
    DEFAULT_SEASONALITY_AMPLITUDE,
    HABITABILITY_LAT_BANDS,
    HABITABILITY_SOFT_MARGIN_C,
    REFERENCE_TEMP_C,
)
from .earth import (
    SURFACE_CLASS_DESERT,
    SURFACE_CLASS_SNOW_ICE,
    SURFACE_CLASS_VEGETATED,
)
from .physics import soft_step
from .safety import clamp, clamp_percent, sanitize_array


def diffuse_lat_temperature(T_lat: np.ndarray, D: float, dphi: float, dt: float) -> np.ndarray:
    arr = sanitize_array(T_lat, fallback=0.0).astype(float)
    if arr.size < 3:
        return arr
    D_eff = max(0.0, float(D))
    dphi_eff = max(1e-6, float(dphi))
    dt_eff = max(0.0, float(dt))
    if D_eff == 0.0 or dt_eff == 0.0:
        return arr

    max_dt = 0.45 * dphi_eff * dphi_eff / D_eff
    step_dt = max(1e-6, min(dt_eff, max_dt))
    n_steps = int(np.ceil(dt_eff / step_dt))
    step_dt = dt_eff / max(1, n_steps)
    coeff = D_eff * step_dt / (dphi_eff * dphi_eff)

    for _ in range(max(1, n_steps)):
        lap = np.zeros_like(arr)
        lap[1:-1] = arr[:-2] - 2.0 * arr[1:-1] + arr[2:]
        lap[0] = arr[1] - arr[0]
        lap[-1] = arr[-2] - arr[-1]
        arr = sanitize_array(arr + coeff * lap, fallback=0.0)
    return arr


def thermal_shape(
    global_T_c: float,
    warm_albedo: float,
    ice_albedo: float,
    enable_seasonality: bool,
) -> tuple[float, float, float]:
    hot_anomaly = max(0.0, float(global_T_c) - 22.0)
    cold_anomaly = max(0.0, 8.0 - float(global_T_c))
    albedo_contrast = max(0.0, float(ice_albedo) - float(warm_albedo))
    seasonality = DEFAULT_SEASONALITY_AMPLITUDE if enable_seasonality else 0.0

    equatorial_heat_excess = clamp(4.8 + 1.05 * hot_anomaly - 0.08 * cold_anomaly + 6.0 * seasonality, 2.5, 18.0)
    polar_cooling_strength = clamp(
        20.0
        + 0.55 * cold_anomaly
        + 0.65 * hot_anomaly
        + 6.0 * max(0.0, albedo_contrast - 0.20)
        + 12.0 * seasonality,
        12.0,
        42.0,
    )
    thermal_spread = equatorial_heat_excess + polar_cooling_strength
    return equatorial_heat_excess, polar_cooling_strength, thermal_spread


def lat_temperature_profile(global_T_c: float, planet_params: Mapping[str, float | bool], year: int | None = None) -> dict[str, np.ndarray | float]:
    lat_deg = np.linspace(-89.5, 89.5, HABITABILITY_LAT_BANDS)
    lat_rad = np.deg2rad(lat_deg)
    enable_seasonality = bool(planet_params.get("enable_seasonality", True))
    equatorial_heat_excess, polar_cooling_strength, thermal_spread = thermal_shape(
        global_T_c=global_T_c,
        warm_albedo=float(planet_params.get("warm_albedo", 0.30)),
        ice_albedo=float(planet_params.get("ice_albedo", 0.62)),
        enable_seasonality=enable_seasonality,
    )
    lat_temp = (
        float(global_T_c)
        + equatorial_heat_excess * (np.cos(lat_rad) ** 2)
        - polar_cooling_strength * (np.sin(lat_rad) ** 2)
    )
    if year is not None and enable_seasonality:
        # Keep annual variability subtle so behavior remains close to previous screens.
        phase = float(year) - np.floor(float(year))
        lat_temp = lat_temp + 0.8 * np.sin(2.0 * np.pi * phase) * np.sin(lat_rad)

    soft_margin_c = (
        HABITABILITY_SOFT_MARGIN_C
        + 0.05 * clamp(thermal_spread - 30.0, 0.0, 24.0)
        + (0.45 if enable_seasonality else 0.0)
    )
    return {
        "lat_deg": lat_deg,
        "lat_rad": lat_rad,
        "lat_temp_c": sanitize_array(lat_temp, fallback=global_T_c),
        "thermal_spread": float(thermal_spread),
        "soft_margin_c": float(soft_margin_c),
    }


def habitability_percent_from_lat_profile(
    T_lat: np.ndarray,
    hab_min: float,
    hab_max: float,
    *,
    soft_margin_c: float = HABITABILITY_SOFT_MARGIN_C,
) -> float:
    temps = sanitize_array(T_lat, fallback=REFERENCE_TEMP_C)
    n = max(temps.size, 1)
    lat_rad = np.deg2rad(np.linspace(-89.5, 89.5, n))
    margin = max(1e-6, float(soft_margin_c))
    lower_ok = np.vectorize(soft_step)((temps - float(hab_min)) / margin)
    upper_ok = np.vectorize(soft_step)((float(hab_max) - temps) / margin)
    local_habitability = np.clip(lower_ok * upper_ok, 0.0, 1.0)
    weights = np.cos(lat_rad)
    denom = float(np.sum(weights)) if np.sum(weights) != 0 else 1.0
    habitable_frac = 100.0 * float(np.sum(weights * local_habitability) / denom)
    return clamp_percent(habitable_frac)


def habitability_stress(global_T_c: float, global_CO2_ppm: float, thermal_spread: float) -> float:
    overheat = clamp((float(global_T_c) - 30.0) / 14.0, 0.0, 1.0)
    cold_stress = clamp((2.0 - float(global_T_c)) / 16.0, 0.0, 1.0)
    greenhouse_stress = clamp((float(global_CO2_ppm) - 650.0) / 1000.0, 0.0, 1.0)
    spread_stress = clamp((float(thermal_spread) - 48.0) / 26.0, 0.0, 1.0)
    comfort_stress = clamp((abs(float(global_T_c) - REFERENCE_TEMP_C) - 2.0) / 30.0, 0.0, 1.0)
    return clamp(
        0.23 * overheat + 0.14 * cold_stress + 0.18 * greenhouse_stress + 0.10 * spread_stress + 0.18 * comfort_stress,
        0.0,
        0.50,
    )


def local_temperature_field_c(
    global_T_c: float,
    lat_deg: np.ndarray,
    elevation_km: np.ndarray,
    microclimate_c: np.ndarray,
    planet_params: Mapping[str, float | bool],
    surface_class: np.ndarray | None = None,
    year: int | None = None,
) -> dict[str, np.ndarray | float]:
    profile = lat_temperature_profile(global_T_c=float(global_T_c), planet_params=planet_params, year=year)
    lat_map = np.asarray(lat_deg, dtype=float)
    profile_lat = np.asarray(profile["lat_deg"], dtype=float)
    profile_temp = np.asarray(profile["lat_temp_c"], dtype=float)
    lat_profile = np.interp(lat_map, profile_lat, profile_temp)
    elev = np.asarray(elevation_km, dtype=float)
    micro = np.asarray(microclimate_c, dtype=float)

    # Keep local modifiers realistic for an Earth visual map; avoid extreme collapse from noisy pseudo-topography.
    elev_pos = np.clip(elev, 0.0, 4.5)
    elev_neg = np.clip(elev, -1.0, 0.0)
    lapse_cooling = 4.3 * elev_pos + 0.7 * elev_neg
    micro_limited = np.clip(micro, -3.2, 3.2)
    local_temp = lat_profile[:, None] + micro_limited - lapse_cooling

    class_adjustment = np.zeros_like(local_temp, dtype=float)
    if surface_class is not None:
        cls = np.asarray(surface_class, dtype=int)
        # Explicitly separate hot arid deserts from cold high-albedo snow/ice.
        class_adjustment = np.where(cls == SURFACE_CLASS_DESERT, class_adjustment + 3.8, class_adjustment)
        class_adjustment = np.where(cls == SURFACE_CLASS_SNOW_ICE, class_adjustment - 8.5, class_adjustment)
        class_adjustment = np.where(cls == SURFACE_CLASS_VEGETATED, class_adjustment + 0.2, class_adjustment)
        local_temp = local_temp + class_adjustment

    return {
        "local_temp_c": sanitize_array(local_temp, fallback=float(global_T_c)),
        "lat_profile_temp_c": lat_profile,
        "surface_class_adjustment_c": class_adjustment,
        "thermal_spread": float(profile["thermal_spread"]),
        "soft_margin_c": float(profile["soft_margin_c"]),
    }


def local_habitability_from_temperature(
    local_temp_c: np.ndarray,
    global_T_c: float,
    global_CO2_ppm: float,
    hab_min_c: float,
    hab_max_c: float,
    thermal_spread: float,
    soft_margin_c: float,
) -> dict[str, np.ndarray | float]:
    local_temp = sanitize_array(local_temp_c, fallback=float(global_T_c))
    margin = max(1e-6, float(soft_margin_c))
    lower_ok = np.vectorize(soft_step)((local_temp - float(hab_min_c)) / margin)
    upper_ok = np.vectorize(soft_step)((float(hab_max_c) - local_temp) / margin)
    temp_score = np.clip(lower_ok * upper_ok, 0.0, 1.0)
    stress = habitability_stress(
        global_T_c=float(global_T_c),
        global_CO2_ppm=float(global_CO2_ppm),
        thermal_spread=float(thermal_spread),
    )
    co2_score = np.full(local_temp.shape, 1.0 - float(stress), dtype=float)
    score = np.clip(temp_score * co2_score, 0.0, 1.0)
    temp_ok = (local_temp >= float(hab_min_c)) & (local_temp <= float(hab_max_c))
    return {
        "score": score,
        "temp_score": temp_score,
        "co2_score": co2_score,
        "temp_ok": temp_ok,
        "stress": float(stress),
    }


def evaluate_local_human_habitability(
    global_T_c: float,
    global_CO2_ppm: float,
    lat_deg: np.ndarray,
    elevation_km: np.ndarray,
    microclimate_c: np.ndarray,
    planet_params: Mapping[str, float | bool],
    surface_class: np.ndarray | None = None,
    year: int | None = None,
) -> dict[str, np.ndarray | float]:
    temp_fields = local_temperature_field_c(
        global_T_c=float(global_T_c),
        lat_deg=np.asarray(lat_deg, dtype=float),
        elevation_km=np.asarray(elevation_km, dtype=float),
        microclimate_c=np.asarray(microclimate_c, dtype=float),
        surface_class=None if surface_class is None else np.asarray(surface_class, dtype=int),
        planet_params=planet_params,
        year=year,
    )
    hab = local_habitability_from_temperature(
        local_temp_c=np.asarray(temp_fields["local_temp_c"], dtype=float),
        global_T_c=float(global_T_c),
        global_CO2_ppm=float(global_CO2_ppm),
        hab_min_c=float(planet_params["habitable_temp_min_c"]),
        hab_max_c=float(planet_params["habitable_temp_max_c"]),
        thermal_spread=float(temp_fields["thermal_spread"]),
        soft_margin_c=float(temp_fields["soft_margin_c"]),
    )
    return {
        "local_temp_c": np.asarray(temp_fields["local_temp_c"], dtype=float),
        "lat_profile_temp_c": np.asarray(temp_fields["lat_profile_temp_c"], dtype=float),
        "surface_class_adjustment_c": np.asarray(temp_fields["surface_class_adjustment_c"], dtype=float),
        "thermal_spread": float(temp_fields["thermal_spread"]),
        "soft_margin_c": float(temp_fields["soft_margin_c"]),
        "score": np.asarray(hab["score"], dtype=float),
        "temp_score": np.asarray(hab["temp_score"], dtype=float),
        "co2_score": np.asarray(hab["co2_score"], dtype=float),
        "temp_ok": np.asarray(hab["temp_ok"], dtype=bool),
        "stress": float(hab["stress"]),
    }
