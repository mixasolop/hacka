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
    co2_forcing_wm2,
    equilibrium_temperature_c,
    temperature_relaxation_update,
    weathering_sink_ppm_per_year,
)
from .safety import clamp, clamp_percent, ensure_finite, sanitize_array, sanitize_state
from .scenario_io import scenario_from_any
from .schema import ScenarioModel

NATURAL_ATMOSPHERE_BASELINE_CO2_PPM = {
    "Minimal": 1.0,
    "Earth-like": CO2_BASELINE_PPM,
    "Dense": 5000.0,
}
NATURAL_ATMOSPHERE_RELAX_YEARS = 160.0
NATURAL_IMPORTED_OUTGASSING_PPM_PER_YEAR = {
    "Minimal": 0.06,
    "Earth-like": 0.20,
    "Dense": 0.40,
}
NATURAL_IMPORTED_MAX_CO2_RISE_PPM_PER_YEAR = {
    "Minimal": 0.08,
    "Earth-like": 0.28,
    "Dense": 0.55,
}
NATURAL_IMPORTED_GREENHOUSE_BASELINE_CO2_PPM = 1.0
NATURAL_IMPORTED_GREENHOUSE_TEMP_PER_WM2 = 0.25
NATURAL_IMPORTED_GREENHOUSE_MAX_OFFSET_C = 25.0
EXTREME_HOT_IMPORTED_THRESHOLD_C = 350.0
EXTREME_HOT_RESPONSE_YEARS = 120.0
EXTREME_HOT_MAX_STEP_C_PER_YEAR = 2.0
EXTREME_HOT_ANCHOR_DELTA_CAP_C = 120.0
EXTREME_HOT_DIAGNOSTIC_LABEL = "Stabilized diagnostic trajectory"
REGIME_STABLE_HABITABLE_MIN_PCT = 60.0
REGIME_UNSTABLE_HABITABLE_MAX_PCT = 20.0
REGIME_STABLE_TEMP_MIN_C = -5.0
REGIME_STABLE_TEMP_MAX_C = 35.0
REGIME_UNSTABLE_TEMP_MIN_C = -35.0
REGIME_UNSTABLE_TEMP_MAX_C = 55.0


def _natural_imported_temperature_anchor(
    scenario: ScenarioModel,
    initial_co2_ppm: float,
) -> tuple[float, float] | None:
    """Return (imported_eq_c_with_atmosphere, model_eq_at_initial_forcing_c_unclamped) for natural imported planets."""
    if not bool(scenario.planet.natural_planet_mode):
        return None
    imported_teq_k = scenario.planet.imported_equilibrium_temperature_k
    if imported_teq_k is None:
        return None
    imported_eq_base_c = clamp(
        ensure_finite(float(imported_teq_k) - 273.15, 0.0),
        temperature_clamp_min_c,
        temperature_clamp_max_c,
    )
    imported_greenhouse_offset_c = _natural_imported_greenhouse_offset_c(scenario, initial_co2_ppm)
    imported_eq_c = clamp(
        imported_eq_base_c + imported_greenhouse_offset_c,
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


def _is_extreme_hot_imported_world(
    scenario: ScenarioModel,
    imported_anchor: tuple[float, float] | None,
) -> bool:
    if not bool(scenario.planet.natural_planet_mode):
        return False
    if imported_anchor is None:
        return False
    imported_eq_c, _ = imported_anchor
    return float(imported_eq_c) >= EXTREME_HOT_IMPORTED_THRESHOLD_C


def initial_temperature_c(scenario: ScenarioModel | Mapping[str, Any]) -> float:
    scenario_obj = scenario_from_any(scenario)
    imported_teq_k = scenario_obj.planet.imported_equilibrium_temperature_k
    if imported_teq_k is not None:
        imported_teq_c = float(imported_teq_k) - 273.15
        imported_greenhouse_offset_c = _natural_imported_greenhouse_offset_c(
            scenario_obj,
            scenario_obj.planet.initial_co2_ppm,
        )
        T0 = imported_teq_c + imported_greenhouse_offset_c
        return clamp(ensure_finite(T0, 0.0), temperature_clamp_min_c, temperature_clamp_max_c)

    T_eq, _ = equilibrium_temperature_c(
        stellar_flux_multiplier=scenario_obj.planet.stellar_flux_multiplier,
        warm_albedo=scenario_obj.planet.warm_albedo,
        ice_albedo=scenario_obj.planet.ice_albedo,
        co2_ppm=scenario_obj.planet.initial_co2_ppm,
        K_CO2=scenario_obj.planet.K_CO2,
    )
    return clamp(ensure_finite(T_eq, 0.0), temperature_clamp_min_c, temperature_clamp_max_c)


def _natural_atmosphere_baseline_co2_ppm(scenario: ScenarioModel) -> float | None:
    if not bool(scenario.planet.natural_planet_mode):
        return None
    assumption = scenario.planet.atmosphere_assumption
    if assumption == "Custom":
        return clamp(float(scenario.planet.initial_co2_ppm), CO2_min, CO2_max_hard)
    if assumption in NATURAL_ATMOSPHERE_BASELINE_CO2_PPM:
        return float(NATURAL_ATMOSPHERE_BASELINE_CO2_PPM[assumption])
    return None


def _natural_imported_assumption_value(
    scenario: ScenarioModel,
    mapping: dict[str, float],
    *,
    low: float,
    high: float,
) -> float:
    assumption = scenario.planet.atmosphere_assumption
    if assumption in mapping:
        return float(mapping[assumption])
    if assumption == "Custom":
        co2_span = np.log10(CO2_max_hard / CO2_min)
        co2_rel = np.log10(max(CO2_min, float(scenario.planet.initial_co2_ppm)) / CO2_min) / max(1e-9, co2_span)
        return float(low + clamp(co2_rel, 0.0, 1.0) * (high - low))
    return float(mapping["Earth-like"])


def _natural_planet_outgassing_ppm_per_year(scenario: ScenarioModel) -> float:
    if not bool(scenario.planet.natural_planet_mode):
        return NATURAL_OUTGASSING_PPM_PER_YEAR
    return _natural_imported_assumption_value(
        scenario,
        NATURAL_IMPORTED_OUTGASSING_PPM_PER_YEAR,
        low=NATURAL_IMPORTED_OUTGASSING_PPM_PER_YEAR["Minimal"],
        high=NATURAL_IMPORTED_OUTGASSING_PPM_PER_YEAR["Dense"],
    )


def _natural_planet_max_co2_rise_ppm_per_year(scenario: ScenarioModel) -> float | None:
    if not bool(scenario.planet.natural_planet_mode):
        return None
    return _natural_imported_assumption_value(
        scenario,
        NATURAL_IMPORTED_MAX_CO2_RISE_PPM_PER_YEAR,
        low=NATURAL_IMPORTED_MAX_CO2_RISE_PPM_PER_YEAR["Minimal"],
        high=NATURAL_IMPORTED_MAX_CO2_RISE_PPM_PER_YEAR["Dense"],
    )


def _natural_imported_greenhouse_offset_c(
    scenario: ScenarioModel,
    co2_ppm: float,
) -> float:
    if not bool(scenario.planet.natural_planet_mode):
        return 0.0
    if scenario.planet.imported_equilibrium_temperature_k is None:
        return 0.0
    forcing_delta = co2_forcing_wm2(
        co2_ppm=max(float(co2_ppm), CO2_min),
        K_CO2=scenario.planet.K_CO2,
        co2_base=NATURAL_IMPORTED_GREENHOUSE_BASELINE_CO2_PPM,
    )
    offset_c = max(0.0, float(forcing_delta) * NATURAL_IMPORTED_GREENHOUSE_TEMP_PER_WM2)
    return clamp(offset_c, 0.0, NATURAL_IMPORTED_GREENHOUSE_MAX_OFFSET_C)


def _natural_atmosphere_relaxation_ppm_per_year(
    co2_ppm: float,
    baseline_co2_ppm: float | None,
) -> float:
    if baseline_co2_ppm is None:
        return 0.0
    return (float(baseline_co2_ppm) - float(co2_ppm)) / max(1.0, NATURAL_ATMOSPHERE_RELAX_YEARS)


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
    _ = (co2_ppm, mode, mitigation_start_year, mitigation_strength, dtemp, dco2, dhabitable, year, warm_albedo, ice_albedo)
    t_c = float(temp_c)
    h_pct = float(habitable_pct)
    if h_pct >= REGIME_STABLE_HABITABLE_MIN_PCT and REGIME_STABLE_TEMP_MIN_C <= t_c <= REGIME_STABLE_TEMP_MAX_C:
        return "Stable"
    if (
        h_pct <= REGIME_UNSTABLE_HABITABLE_MAX_PCT
        or t_c <= REGIME_UNSTABLE_TEMP_MIN_C
        or t_c >= REGIME_UNSTABLE_TEMP_MAX_C
    ):
        return "Unstable"
    return "Marginal"


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
    extreme_hot_world = _is_extreme_hot_imported_world(s, imported_anchor)
    natural_atmosphere_baseline = _natural_atmosphere_baseline_co2_ppm(s)
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

        temp_eq_c_model, _ = equilibrium_temperature_c(
            stellar_flux_multiplier=s.planet.stellar_flux_multiplier,
            warm_albedo=s.planet.warm_albedo,
            ice_albedo=s.planet.ice_albedo,
            co2_ppm=co2_ppm,
            K_CO2=s.planet.K_CO2,
        )
        if imported_anchor is not None:
            imported_eq_c, model_eq_ref_c = imported_anchor
            # Preserve forcing sensitivity while anchoring the natural-mode baseline to imported Teq.
            forcing_delta_c = float(temp_eq_c_model) - float(model_eq_ref_c)
            if extreme_hot_world:
                forcing_delta_c = clamp(
                    forcing_delta_c,
                    -EXTREME_HOT_ANCHOR_DELTA_CAP_C,
                    EXTREME_HOT_ANCHOR_DELTA_CAP_C,
                )
            temp_eq_c = imported_eq_c + forcing_delta_c
        else:
            temp_eq_c = float(temp_eq_c_model)
        temp_eq_c = clamp(ensure_finite(temp_eq_c, temp_c), temperature_clamp_min_c, temperature_clamp_max_c)
        response_years = TEMPERATURE_RESPONSE_YEARS
        if extreme_hot_world:
            response_years = max(TEMPERATURE_RESPONSE_YEARS, EXTREME_HOT_RESPONSE_YEARS)
        temp_next = temperature_relaxation_update(
            temp_c=temp_c,
            temp_eq_c=temp_eq_c,
            co2_ppm=co2_ppm,
            dt_years=dt,
            response_years=response_years,
            stellar_flux_multiplier=s.planet.stellar_flux_multiplier,
            warm_albedo=s.planet.warm_albedo,
            ice_albedo=s.planet.ice_albedo,
        )
        if extreme_hot_world:
            max_step_c = EXTREME_HOT_MAX_STEP_C_PER_YEAR * dt
            temp_next = clamp(temp_next, temp_c - max_step_c, temp_c + max_step_c)

        sink_weathering = weathering_sink_ppm_per_year(temp_c=temp_c, co2_ppm=co2_ppm)
        sink_biosphere = biosphere_sink_ppm_per_year(T_global_c=temp_c, CO2_ppm=co2_ppm, H_pct=habitable_pct)
        outgassing_ppm_per_year = _natural_planet_outgassing_ppm_per_year(s)
        dco2_dt = float(civ_terms["E_human_eff"]) + outgassing_ppm_per_year - sink_weathering - sink_biosphere
        dco2_dt += _natural_atmosphere_relaxation_ppm_per_year(
            co2_ppm=co2_ppm,
            baseline_co2_ppm=natural_atmosphere_baseline,
        )
        max_rise_ppm_per_year = _natural_planet_max_co2_rise_ppm_per_year(s)
        if max_rise_ppm_per_year is not None:
            dco2_dt = min(dco2_dt, float(max_rise_ppm_per_year))

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
            "outside_calibrated_habitable_regime": bool(extreme_hot_world),
            "extreme_hot_diagnostic_mode": bool(extreme_hot_world),
            "diagnostic_label": EXTREME_HOT_DIAGNOSTIC_LABEL if extreme_hot_world else None,
            "trajectory_mode": "stabilized_diagnostic" if extreme_hot_world else "nominal",
        },
    }
    return result


def preview_state(
    scenario: ScenarioModel | Mapping[str, Any],
    horizon_years: int = PREVIEW_HORIZON_YEARS,
) -> dict[str, float | str | bool | None]:
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
    regime_0 = str(series["regime"][idx0])
    regime_h = str(series["regime"][idx_h])
    meta = dict(series.get("meta", {}))

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
        "system_state": regime_0,
        "tipping_label": tipping_label,
        "net_emissions": float(series["human_emissions_eff"][idx0]) + NATURAL_OUTGASSING_PPM_PER_YEAR,
        "T0_c": float(series["meta"]["T0_c"]),
        "outside_calibrated_habitable_regime": bool(meta.get("outside_calibrated_habitable_regime", False)),
        "extreme_hot_diagnostic_mode": bool(meta.get("extreme_hot_diagnostic_mode", False)),
        "diagnostic_label": meta.get("diagnostic_label"),
        "trajectory_mode": str(meta.get("trajectory_mode", "nominal")),
    }
