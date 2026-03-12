"""Civilization dynamics and emissions coupling."""

from __future__ import annotations

from typing import Mapping

from .constants import EMISSIONS_TO_PPM_PER_YEAR
from .safety import clamp, clamp01, safe_exp


def _smoothstep01(x: float) -> float:
    t = clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def survival_factor_temperature(T_global_c: float) -> float:
    T = float(T_global_c)
    if T <= -10.0:
        return 0.0
    if T < 0.0:
        return _smoothstep01((T + 10.0) / 10.0)
    if T <= 30.0:
        return 1.0
    if T < 40.0:
        return 1.0 - _smoothstep01((T - 30.0) / 10.0)
    return 0.0


def survival_factor_habitability(H_pct: float) -> float:
    H = float(H_pct)
    if H > 60.0:
        return 1.0
    if H > 30.0:
        return (H - 30.0) / 30.0
    return 0.0


def civilization_survival_factor(T_global_c: float, H_pct: float) -> float:
    return clamp01(survival_factor_temperature(T_global_c) * survival_factor_habitability(H_pct))


def human_emissions_rate_base(
    mode: str,
    emissions_rate: float,
    year: int,
    mitigation_start_year: int,
    mitigation_strength: float,
) -> float:
    year_i = max(0, int(year))
    emissions = max(0.0, float(emissions_rate))
    mitigation_strength = clamp01(float(mitigation_strength))
    mitigation_start_year = max(0, int(mitigation_start_year))

    if mode == "Growing":
        gross = emissions * (1.0 + 0.010 * year_i)
    elif mode == "Carefree":
        gross = emissions * (1.0 + 0.018 * year_i) / (1.0 + 0.0012 * year_i)
    elif mode == "Stabilization":
        pre_growth = emissions * (1.0 + 0.004 * min(year_i, mitigation_start_year))
        if year_i < mitigation_start_year:
            gross = pre_growth
        else:
            decay = 0.035 * max(0.1, mitigation_strength) * (year_i - mitigation_start_year)
            gross = pre_growth * safe_exp(-decay)
    elif mode == "Aggressive Mitigation":
        if year_i < mitigation_start_year:
            gross = emissions
        else:
            decay = 0.065 * max(0.2, mitigation_strength) * (year_i - mitigation_start_year)
            gross = emissions * safe_exp(-decay)
    else:
        gross = emissions

    if year_i >= mitigation_start_year and mode in {"Constant", "Growing", "Carefree"}:
        gross *= max(0.05, 1.0 - 0.85 * mitigation_strength)

    return float(EMISSIONS_TO_PPM_PER_YEAR * max(0.0, gross))


def mitigation_base(
    year: int,
    mitigation_start_year: int,
    mitigation_strength: float,
) -> float:
    if int(year) < int(mitigation_start_year):
        return 0.0
    return clamp01(float(mitigation_strength))


def emissions_and_mitigation(
    civ: Mapping[str, float | int | str],
    *,
    year: int,
    T_global_c: float,
    H_pct: float,
    natural_planet_mode: bool,
) -> dict[str, float]:
    mode = str(civ.get("emissions_growth_mode", "Constant"))
    emissions_rate = float(civ.get("emissions_rate", 0.0))
    mitigation_start_year = int(civ.get("mitigation_start_year", 0))
    mitigation_strength = float(civ.get("mitigation_strength", 0.0))

    E_human_base = human_emissions_rate_base(
        mode=mode,
        emissions_rate=emissions_rate,
        year=int(year),
        mitigation_start_year=mitigation_start_year,
        mitigation_strength=mitigation_strength,
    )
    M_base = mitigation_base(
        year=int(year),
        mitigation_start_year=mitigation_start_year,
        mitigation_strength=mitigation_strength,
    )
    C_civ = civilization_survival_factor(T_global_c=T_global_c, H_pct=H_pct)
    if natural_planet_mode:
        C_civ = 0.0
    E_human_eff = E_human_base * C_civ
    M_eff = M_base * C_civ
    return {
        "C_civ": clamp01(C_civ),
        "E_human_base": max(0.0, E_human_base),
        "E_human_eff": max(0.0, E_human_eff),
        "M_base": clamp01(M_base),
        "M_eff": clamp01(M_eff),
    }

