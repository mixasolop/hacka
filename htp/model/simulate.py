"""Authoritative simulation engine used by all pages."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .bounds import CO2_max_hard, CO2_min, temperature_clamp_max_c, temperature_clamp_min_c
from .civilization import emissions_and_mitigation
from .constants import (
    CO2_BASELINE_PPM,
    CO2_DANGER_THRESHOLD_PPM,
    HABITABILITY_THRESHOLDS,
    NATURAL_OUTGASSING_PPM_PER_YEAR,
    PREVIEW_HORIZON_YEARS,
    SIM_DT_YEARS_DEFAULT,
    SIM_YEARS_DEFAULT,
    SOLAR_CONSTANT,
    TEMP_COLD_THRESHOLD_C,
    TEMP_HOT_THRESHOLD_C,
    TEMPERATURE_RESPONSE_YEARS,
)
from .latitude import habitability_percent_from_lat_profile, habitability_stress, lat_temperature_profile
from .physics import (
    albedo_from_temperature,
    biosphere_sink_ppm_per_year,
    equilibrium_temperature_c,
    temperature_relaxation_update,
    weathering_sink_ppm_per_year,
)
from .safety import clamp, clamp_percent, ensure_finite, safe_log, sanitize_array, sanitize_state
from .scenario_io import scenario_from_any
from .schema import ScenarioModel


def _natural_imported_temperature_anchor(
    scenario: ScenarioModel,
    initial_co2_ppm: float,
) -> tuple[float, float] | None:
    """Return (imported_eq_c, model_eq_at_initial_forcing_c) for natural imported planets.

    The model equilibrium reference is intentionally kept *unclamped* so the
    anchor delta stays self-consistent even for extreme forcing values.
    """
    if not bool(scenario.planet.natural_planet_mode):
        return None
    imported_teq_k = scenario.planet.imported_equilibrium_temperature_k
    if imported_teq_k is None:
        return None
    imported_eq_c = clamp(
        ensure_finite(float(imported_teq_k) - 273.15, 0.0),
        temperature_clamp_min_c,
        temperature_clamp_max_c,
    )
    model_eq_c, _ = equilibrium_temperature_c(
        stellar_flux_multiplier=scenario.planet.stellar_flux_multiplier,
        warm_albedo=scenario.planet.warm_albedo,
        ice_albedo=scenario.planet.ice_albedo,
        co2_ppm=initial_co2_ppm,
        K_CO2=scenario.planet.K_CO2,
    )
    model_eq_c = ensure_finite(model_eq_c, imported_eq_c)
    return imported_eq_c, model_eq_c


def initial_temperature_c(scenario: ScenarioModel | Mapping[str, Any]) -> float:
    scenario_obj = scenario_from_any(scenario)
    imported_teq_k = scenario_obj.planet.imported_equilibrium_temperature_k
    if imported_teq_k is not None:
        T0 = float(imported_teq_k) - 273.15
        return clamp(ensure_finite(T0, 0.0), temperature_clamp_min_c, temperature_clamp_max_c)

    T_eq, _ = equilibrium_temperature_c(
        stellar_flux_multiplier=scenario_obj.planet.stellar_flux_multiplier,
        warm_albedo=scenario_obj.planet.warm_albedo,
        ice_albedo=scenario_obj.planet.ice_albedo,
        co2_ppm=scenario_obj.planet.initial_co2_ppm,
        K_CO2=scenario_obj.planet.K_CO2,
    )
    return clamp(ensure_finite(T_eq, 0.0), temperature_clamp_min_c, temperature_clamp_max_c)


def _regime_label(
    temp_c: float,
    co2_ppm: float,
    habitable_pct: float,
    mode: str,
    mitigation_start_year: int,
    mitigation_strength: float,
    dtemp: float,
    dco2: float,
    dhabitable: float,
    year: int,
    warm_albedo: float,
    ice_albedo: float,
) -> str:
    co2_ratio = max(float(co2_ppm), CO2_min) / CO2_BASELINE_PPM
    co2_pressure = max(0.0, safe_log(co2_ratio))
    hot_stress = max(0.0, (float(temp_c) - 18.0) / 9.0)
    cold_stress = max(0.0, (8.0 - float(temp_c)) / 10.0)
    high_co2_penalty = max(0.0, (float(co2_ppm) - 700.0) / 500.0)
    habitability_stress = max(0.0, (75.0 - float(habitable_pct)) / 28.0)

    base_score = 0.85 * co2_pressure + 1.00 * hot_stress + 0.45 * cold_stress + 0.48 * habitability_stress + 0.40 * high_co2_penalty
    emissions_pressure = {
        "Aggressive Mitigation": -0.35,
        "Stabilization": -0.18,
        "Constant": 0.08,
        "Growing": 0.28,
        "Carefree": 0.38,
    }.get(mode, 0.10)

    years_until_mitigation = max(0, int(mitigation_start_year) - int(year))
    delay_pressure = 0.22 * clamp(years_until_mitigation / 120.0, 0.0, 1.0)
    weakness_pressure = 0.14 * (1.0 - float(mitigation_strength)) * (1.0 if years_until_mitigation > 0 else 0.4)
    mitigation_pressure = delay_pressure + weakness_pressure

    trend_pressure = 3.2 * max(0.0, dtemp) + 0.040 * max(0.0, dco2) + 0.045 * max(0.0, -dhabitable)
    cold_lock_in = 0.12 * clamp((5.0 - temp_c) / 15.0, 0.0, 1.0) * clamp((ice_albedo - warm_albedo - 0.20) / 0.40, 0.0, 1.0)
    score = base_score + emissions_pressure + mitigation_pressure + trend_pressure + cold_lock_in

    if score < 1.05:
        return "Stable"
    if score < 2.25:
        return "Marginal"
    return "Unstable"


def _event_specifications() -> list[dict[str, str]]:
    specs = [
        {
            "id": "temp_hot",
            "kind": "danger",
            "event_label": f"Global temperature exceeds {TEMP_HOT_THRESHOLD_C:.0f}°C",
            "initial_label": f"Initial condition: temperature already above {TEMP_HOT_THRESHOLD_C:.0f}°C",
        },
        {
            "id": "temp_cold",
            "kind": "danger",
            "event_label": f"Global temperature falls below {TEMP_COLD_THRESHOLD_C:.0f}°C",
            "initial_label": f"Initial condition: temperature already below {TEMP_COLD_THRESHOLD_C:.0f}°C",
        },
        {
            "id": "co2_danger",
            "kind": "danger",
            "event_label": f"CO₂ exceeds danger threshold ({CO2_DANGER_THRESHOLD_PPM:.0f} ppm)",
            "initial_label": f"Initial condition: CO₂ already above {CO2_DANGER_THRESHOLD_PPM:.0f} ppm",
        },
    ]
    for threshold in HABITABILITY_THRESHOLDS:
        specs.append(
            {
                "id": f"hab_{threshold}",
                "kind": "habitability",
                "event_label": f"Habitable surface drops below {threshold}%",
                "initial_label": f"Initial condition: habitability already below {threshold}%",
            }
        )
    return specs


def _threshold_state(temp_c: float, co2_ppm: float, habitable_pct: float) -> dict[str, bool]:
    flags: dict[str, bool] = {
        "temp_hot": temp_c >= TEMP_HOT_THRESHOLD_C,
        "temp_cold": temp_c <= TEMP_COLD_THRESHOLD_C,
        "co2_danger": co2_ppm >= CO2_DANGER_THRESHOLD_PPM,
    }
    for threshold in HABITABILITY_THRESHOLDS:
        flags[f"hab_{threshold}"] = habitable_pct < float(threshold)
    return flags


def simulate_time_series(
    scenario: ScenarioModel | Mapping[str, Any],
    years: int = SIM_YEARS_DEFAULT,
    dt_years: float = SIM_DT_YEARS_DEFAULT,
) -> dict[str, Any]:
    s = scenario_from_any(scenario)
    horizon_years = max(1, int(years))
    dt = max(1e-6, float(dt_years))
    n_steps = int(np.floor(horizon_years / dt))
    time_years = np.arange(n_steps + 1, dtype=float) * dt

    temp_c_series = np.zeros_like(time_years, dtype=float)
    temp_k_series = np.zeros_like(time_years, dtype=float)
    co2_series = np.zeros_like(time_years, dtype=float)
    albedo_series = np.zeros_like(time_years, dtype=float)
    habitable_series = np.zeros_like(time_years, dtype=float)
    civ_series = np.zeros_like(time_years, dtype=float)
    e_human_series = np.zeros_like(time_years, dtype=float)
    m_eff_series = np.zeros_like(time_years, dtype=float)
    regime_series: list[str] = []
    events: list[dict[str, Any]] = []
    initial_conditions: list[dict[str, Any]] = []

    co2_ppm = clamp(float(s.planet.initial_co2_ppm), CO2_min, CO2_max_hard)
    temp_c = initial_temperature_c(s)
    imported_anchor = _natural_imported_temperature_anchor(s, co2_ppm)
    event_specs = _event_specifications()
    mitigation_event_added = False
    prev_flags: dict[str, bool] | None = None

    for i, year_float in enumerate(time_years):
        year = int(round(year_float))
        albedo = albedo_from_temperature(temp_c, s.planet.warm_albedo, s.planet.ice_albedo)
        lat_profile = lat_temperature_profile(
            global_T_c=temp_c,
            planet_params={
                "warm_albedo": s.planet.warm_albedo,
                "ice_albedo": s.planet.ice_albedo,
                "enable_seasonality": s.planet.enable_seasonality,
            },
            year=year,
        )
        habitable_pct = habitability_percent_from_lat_profile(
            np.asarray(lat_profile["lat_temp_c"], dtype=float),
            s.planet.habitable_temp_min_c,
            s.planet.habitable_temp_max_c,
            soft_margin_c=float(lat_profile["soft_margin_c"]),
        )
        stress = habitability_stress(
            global_T_c=temp_c,
            global_CO2_ppm=co2_ppm,
            thermal_spread=float(lat_profile["thermal_spread"]),
        )
        habitable_pct = clamp_percent(habitable_pct * (1.0 - stress))
        civ_terms = emissions_and_mitigation(
            s.civilization.model_dump(mode="python"),
            year=year,
            T_global_c=temp_c,
            H_pct=habitable_pct,
            natural_planet_mode=bool(s.planet.natural_planet_mode),
        )

        temp_c, co2_ppm, albedo, habitable_pct = sanitize_state(temp_c, co2_ppm, albedo, habitable_pct)
        if i > 0:
            dtemp = temp_c - float(temp_c_series[i - 1])
            dco2 = co2_ppm - float(co2_series[i - 1])
            dhabitable = habitable_pct - float(habitable_series[i - 1])
        else:
            dtemp = 0.0
            dco2 = 0.0
            dhabitable = 0.0

        regime = _regime_label(
            temp_c=temp_c,
            co2_ppm=co2_ppm,
            habitable_pct=habitable_pct,
            mode=s.civilization.emissions_growth_mode,
            mitigation_start_year=s.civilization.mitigation_start_year,
            mitigation_strength=s.civilization.mitigation_strength,
            dtemp=dtemp,
            dco2=dco2,
            dhabitable=dhabitable,
            year=year,
            warm_albedo=s.planet.warm_albedo,
            ice_albedo=s.planet.ice_albedo,
        )
        flags = _threshold_state(temp_c, co2_ppm, habitable_pct)

        temp_c_series[i] = temp_c
        temp_k_series[i] = temp_c + 273.15
        co2_series[i] = co2_ppm
        albedo_series[i] = albedo
        habitable_series[i] = habitable_pct
        civ_series[i] = float(civ_terms["C_civ"])
        e_human_series[i] = float(civ_terms["E_human_eff"])
        m_eff_series[i] = float(civ_terms["M_eff"])
        regime_series.append(regime)

        if i == 0:
            for spec in event_specs:
                if flags[spec["id"]]:
                    initial_conditions.append({"label": spec["initial_label"], "kind": spec["kind"]})
            prev_flags = flags.copy()
        else:
            for spec in event_specs:
                now_violated = bool(flags[spec["id"]])
                was_violated = bool(prev_flags[spec["id"]]) if prev_flags is not None else False
                if now_violated and not was_violated:
                    events.append({"year": int(round(year_float)), "label": spec["event_label"], "kind": spec["kind"]})
            prev_flags = flags.copy()

        if (
            (not mitigation_event_added)
            and year == s.civilization.mitigation_start_year
            and year <= horizon_years
            and year > 0
        ):
            events.append({"year": int(year), "label": "Mitigation begins", "kind": "mitigation"})
            mitigation_event_added = True

        if i == len(time_years) - 1:
            continue

        temp_eq_c_model_raw, _ = equilibrium_temperature_c(
            stellar_flux_multiplier=s.planet.stellar_flux_multiplier,
            warm_albedo=s.planet.warm_albedo,
            ice_albedo=s.planet.ice_albedo,
            co2_ppm=co2_ppm,
            K_CO2=s.planet.K_CO2,
        )
        temp_eq_c_model_raw = ensure_finite(temp_eq_c_model_raw, temp_c)
        if imported_anchor is not None:
            imported_eq_c, model_eq_ref_c = imported_anchor
            # Preserve forcing sensitivity while anchoring the natural-mode baseline to imported Teq.
            temp_eq_c = imported_eq_c + (float(temp_eq_c_model_raw) - float(model_eq_ref_c))
        else:
            temp_eq_c = float(temp_eq_c_model_raw)
        temp_eq_c = clamp(ensure_finite(temp_eq_c, temp_c), temperature_clamp_min_c, temperature_clamp_max_c)
        temp_next = temperature_relaxation_update(
            temp_c=temp_c,
            temp_eq_c=temp_eq_c,
            co2_ppm=co2_ppm,
            dt_years=dt,
            response_years=TEMPERATURE_RESPONSE_YEARS,
            stellar_flux_multiplier=s.planet.stellar_flux_multiplier,
            warm_albedo=s.planet.warm_albedo,
            ice_albedo=s.planet.ice_albedo,
        )

        sink_weathering = weathering_sink_ppm_per_year(temp_c=temp_c, co2_ppm=co2_ppm)
        sink_biosphere = biosphere_sink_ppm_per_year(T_global_c=temp_c, CO2_ppm=co2_ppm, H_pct=habitable_pct)
        dco2_dt = float(civ_terms["E_human_eff"]) + NATURAL_OUTGASSING_PPM_PER_YEAR - sink_weathering - sink_biosphere

        CO2_safe = max(co2_ppm, CO2_min)
        co2_next = max(CO2_min, CO2_safe + dco2_dt * dt)
        co2_next = min(co2_next, CO2_max_hard)
        if not np.isfinite(co2_next):
            co2_next = co2_ppm
        temp_c = clamp(ensure_finite(temp_next, temp_c), temperature_clamp_min_c, temperature_clamp_max_c)
        co2_ppm = clamp(ensure_finite(co2_next, co2_ppm), CO2_min, CO2_max_hard)

    events = sorted(events, key=lambda item: item["year"])
    result = {
        "time_years": sanitize_array(time_years, fallback=0.0),
        "global_temperature_c": sanitize_array(temp_c_series, fallback=0.0),
        "global_temperature_k": sanitize_array(temp_k_series, fallback=273.15),
        "co2_ppm": sanitize_array(co2_series, fallback=CO2_min),
        "effective_albedo": sanitize_array(albedo_series, fallback=0.3),
        "habitable_surface_percent": sanitize_array(np.clip(habitable_series, 0.0, 100.0), fallback=0.0),
        "c_civ": sanitize_array(civ_series, fallback=0.0),
        "human_emissions_eff": sanitize_array(e_human_series, fallback=0.0),
        "mitigation_eff": sanitize_array(m_eff_series, fallback=0.0),
        "regime": regime_series,
        "initial_conditions": initial_conditions,
        "events": events,
        "meta": {
            "seed": int(s.planet.seed),
            "T0_c": float(temp_c_series[0]) if len(temp_c_series) else 0.0,
            "natural_planet_mode": bool(s.planet.natural_planet_mode),
            "import_classification": s.planet.import_classification,
        },
    }
    return result


def preview_state(scenario: ScenarioModel | Mapping[str, Any], horizon_years: int = PREVIEW_HORIZON_YEARS) -> dict[str, float | str]:
    s = scenario_from_any(scenario)
    horizon = max(1, int(horizon_years))
    series = simulate_time_series(s, years=horizon, dt_years=1.0)
    idx0 = 0
    idx_h = min(horizon, len(series["time_years"]) - 1)
    temp0 = float(series["global_temperature_c"][idx0])
    co20 = float(series["co2_ppm"][idx0])
    hab0 = float(series["habitable_surface_percent"][idx0])
    alb0 = float(series["effective_albedo"][idx0])
    co2_h = float(series["co2_ppm"][idx_h])
    regime_h = str(series["regime"][idx_h])

    ice_fraction = clamp((alb0 - s.planet.warm_albedo) / max(1e-6, s.planet.ice_albedo - s.planet.warm_albedo), 0.0, 1.0)
    flux_cold = clamp((1.0 - s.planet.stellar_flux_multiplier) / 0.18, 0.0, 1.0)
    snowball_score = clamp(
        0.55 * ice_fraction
        + 0.25 * clamp((s.planet.ice_albedo - s.planet.warm_albedo - 0.15) / 0.45, 0.0, 1.0)
        + 0.20 * flux_cold,
        0.0,
        1.0,
    )
    runaway_score = clamp(
        0.40 * clamp((temp0 - 28.0) / 12.0, 0.0, 1.0)
        + 0.25 * clamp((co2_h - 700.0) / 900.0, 0.0, 1.0)
        + 0.20 * clamp((s.planet.stellar_flux_multiplier - 1.02) / 0.18, 0.0, 1.0)
        + 0.15 * clamp(float(series["human_emissions_eff"][idx0]) / 6.0, 0.0, 1.0),
        0.0,
        1.0,
    )
    tipping_label = {"Stable": "Low", "Marginal": "Elevated", "Unstable": "High"}.get(regime_h, "Elevated")
    if max(snowball_score, runaway_score) > 0.72:
        tipping_label = "High"
    elif max(snowball_score, runaway_score) > 0.45 and tipping_label == "Low":
        tipping_label = "Elevated"

    return {
        "stellar_flux_w_m2": float(SOLAR_CONSTANT * s.planet.stellar_flux_multiplier),
        "effective_albedo": alb0,
        "temperature_c": temp0,
        "temperature_k": temp0 + 273.15,
        "co2_ppm": co20,
        "projected_co2_ppm": co2_h,
        "habitable_surface_pct": hab0,
        "ice_fraction": ice_fraction,
        "snowball_score": snowball_score,
        "runaway_score": runaway_score,
        "stability_index": float(max(0.0, snowball_score + runaway_score)),
        "stability_outlook": regime_h,
        "system_state": regime_h,
        "tipping_label": tipping_label,
        "net_emissions": float(series["human_emissions_eff"][idx0]) + NATURAL_OUTGASSING_PPM_PER_YEAR,
        "T0_c": float(series["meta"]["T0_c"]),
    }
