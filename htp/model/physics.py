"""Shared physical parameterizations."""

from __future__ import annotations

import math
from typing import Mapping

from .bounds import CO2_min
from .constants import (
    ALBEDO_REF,
    BIOSPHERE_COEFF,
    BIOSPHERE_OPT_TEMP_C,
    BIOSPHERE_TEMP_WIDTH_C,
    CO2_BASELINE_PPM,
    DEFAULT_ICE_TRANSITION_TEMP_C,
    DEFAULT_K_CO2,
    ICE_TRANSITION_WIDTH_C,
    K_ALBEDO,
    K_FLUX,
    NATURAL_OUTGASSING_PPM_PER_YEAR,
    REFERENCE_TEMP_C,
    WEATHERING_COEFF,
    WEATHERING_TEMP_SENS,
)
from .safety import clamp, clamp01, clamp_percent, safe_exp, safe_log


def soft_step(x: float) -> float:
    return 1.0 / (1.0 + safe_exp(-x))


def albedo_from_temperature(
    T_c: float,
    warm_albedo: float,
    ice_albedo: float,
    T_ice: float = DEFAULT_ICE_TRANSITION_TEMP_C,
    width: float = ICE_TRANSITION_WIDTH_C,
) -> float:
    width_safe = max(1e-6, abs(width))
    exponent = (float(T_c) - float(T_ice)) / width_safe
    w_ice = 1.0 / (1.0 + safe_exp(exponent))
    alpha = float(warm_albedo) * (1.0 - w_ice) + float(ice_albedo) * w_ice
    return clamp01(alpha)


def co2_forcing_wm2(co2_ppm: float, K_CO2: float = DEFAULT_K_CO2, co2_base: float = CO2_BASELINE_PPM) -> float:
    CO2_safe = max(float(co2_ppm), CO2_min)
    return float(K_CO2) * safe_log(CO2_safe / float(co2_base))


def cold_trap_cooling(
    temp_c: float,
    co2_ppm: float,
    stellar_flux_multiplier: float,
    warm_albedo: float,
    ice_albedo: float,
) -> float:
    if temp_c >= 9.0:
        return 0.0
    cold_severity = clamp((9.0 - temp_c) / 30.0, 0.0, 1.0)
    albedo_contrast = clamp((ice_albedo - warm_albedo - 0.18) / 0.45, 0.0, 1.0)
    low_flux = clamp((1.0 - stellar_flux_multiplier) / 0.22, 0.0, 1.0)
    if albedo_contrast <= 0.0 and low_flux <= 0.0:
        return 0.0
    co2_escape = clamp((co2_ppm - CO2_BASELINE_PPM) / 1500.0, 0.0, 1.0)
    persistence = 1.0 - 0.40 * co2_escape
    return 0.40 * cold_severity * (0.65 * albedo_contrast + 0.35 * low_flux) * persistence


def equilibrium_temperature_c(
    stellar_flux_multiplier: float,
    warm_albedo: float,
    ice_albedo: float,
    co2_ppm: float,
    K_CO2: float = DEFAULT_K_CO2,
    iterations: int = 3,
) -> tuple[float, float]:
    temp_c = float(REFERENCE_TEMP_C)
    co2_safe = max(float(co2_ppm), CO2_min)
    albedo_contrast = clamp((float(ice_albedo) - float(warm_albedo) - 0.18) / 0.45, 0.0, 1.0)
    low_flux = clamp((1.0 - float(stellar_flux_multiplier)) / 0.22, 0.0, 1.0)
    for _ in range(max(1, int(iterations))):
        alpha_eff = albedo_from_temperature(temp_c, warm_albedo, ice_albedo)
        cold_index = soft_step((6.0 - temp_c) / 4.0)
        cold_locking_feedback = 6.5 * cold_index * (0.55 * albedo_contrast + 0.45 * low_flux)
        temp_c = (
            REFERENCE_TEMP_C
            + K_FLUX * (float(stellar_flux_multiplier) - 1.0)
            - K_ALBEDO * (alpha_eff - ALBEDO_REF)
            + co2_forcing_wm2(co2_safe, K_CO2=K_CO2, co2_base=CO2_BASELINE_PPM)
            - cold_locking_feedback
        )
    return float(temp_c), albedo_from_temperature(temp_c, warm_albedo, ice_albedo)


def temperature_relaxation_update(
    temp_c: float,
    temp_eq_c: float,
    co2_ppm: float,
    dt_years: float,
    response_years: float,
    stellar_flux_multiplier: float,
    warm_albedo: float,
    ice_albedo: float,
) -> float:
    cold_severity = clamp((8.0 - temp_c) / 20.0, 0.0, 1.0)
    contrast = clamp((ice_albedo - warm_albedo - 0.18) / 0.45, 0.0, 1.0)
    response = max(1e-6, response_years * (1.0 + 1.40 * cold_severity * contrast))
    temp_next = float(temp_c) + (float(temp_eq_c) - float(temp_c)) * (float(dt_years) / response)
    temp_next -= cold_trap_cooling(
        temp_next,
        co2_ppm=co2_ppm,
        stellar_flux_multiplier=stellar_flux_multiplier,
        warm_albedo=warm_albedo,
        ice_albedo=ice_albedo,
    )
    return temp_next


def weathering_sink_ppm_per_year(
    temp_c: float,
    co2_ppm: float,
    *,
    coeff: float = WEATHERING_COEFF,
    temp_sensitivity: float = WEATHERING_TEMP_SENS,
    co2_base_ppm: float = CO2_BASELINE_PPM,
    reference_temp_c: float = REFERENCE_TEMP_C,
) -> float:
    exponent = clamp(temp_sensitivity * (temp_c - reference_temp_c), -20.0, 20.0)
    temp_factor = safe_exp(exponent)
    co2_ratio = max(float(co2_ppm), CO2_min) / float(co2_base_ppm)
    return float(coeff) * co2_ratio * temp_factor


def biosphere_sink_ppm_per_year(
    T_global_c: float,
    CO2_ppm: float,
    H_pct: float,
    params: Mapping[str, float] | None = None,
) -> float:
    params = params or {}
    T_opt = float(params.get("bio_temp_opt_c", BIOSPHERE_OPT_TEMP_C))
    sigma_T = max(1e-6, float(params.get("bio_temp_sigma_c", BIOSPHERE_TEMP_WIDTH_C)))
    coeff = float(params.get("biosphere_coeff", BIOSPHERE_COEFF))
    co2_base = max(1e-9, float(params.get("co2_base_ppm", CO2_BASELINE_PPM)))
    CO2_safe = max(float(CO2_ppm), CO2_min)

    exponent = -((float(T_global_c) - T_opt) ** 2) / (2.0 * sigma_T * sigma_T)
    exponent = clamp(exponent, -60.0, 0.0)
    bio_T = safe_exp(exponent)
    bio_area_factor = clamp((clamp_percent(H_pct) - 30.0) / 30.0, 0.0, 1.0)
    return coeff * (CO2_safe / co2_base) * bio_T * bio_area_factor


def natural_outgassing_ppm_per_year() -> float:
    return NATURAL_OUTGASSING_PPM_PER_YEAR

