"""Normalization for imported NASA exoplanet rows."""

from __future__ import annotations

from typing import Any

from .bounds import get_bounds
from .safety import clamp, safe_float
from .scenario_io import scenario_to_flat_params
from .schema import CivilizationState, PlanetState, ScenarioModel


def _classify_import(eqt_k: float | None) -> str:
    if eqt_k is None:
        return "unknown"
    eqt_c = float(eqt_k) - 273.15
    if eqt_c <= -20.0:
        return "cold"
    if eqt_c >= 35.0:
        return "hot"
    return "temperate"


def normalize_imported_planet(raw: dict[str, Any], atmosphere_assumption: str = "Minimal") -> ScenarioModel:
    insol_raw = safe_float(raw.get("pl_insol"), float("nan"))
    eqt_k_raw = safe_float(raw.get("pl_eqt"), float("nan"))

    insol = insol_raw if insol_raw == insol_raw else None
    eqt_k = eqt_k_raw if eqt_k_raw == eqt_k_raw else None

    stellar_lo, stellar_hi = get_bounds("stellar_flux_multiplier")
    co2_lo, co2_hi = get_bounds("initial_co2_ppm")

    if atmosphere_assumption == "Earth-like":
        initial_co2 = 280.0
    elif atmosphere_assumption == "Dense":
        initial_co2 = 5000.0
    elif atmosphere_assumption == "Custom":
        initial_co2 = safe_float(raw.get("initial_co2_ppm"), co2_lo)
    else:
        atmosphere_assumption = "Minimal"
        initial_co2 = 1.0

    planet = PlanetState(
        stellar_flux_multiplier=clamp(insol if insol is not None else 1.0, stellar_lo, stellar_hi),
        enable_seasonality=bool(safe_float(raw.get("pl_orbeccen"), 0.0) >= 0.08),
        warm_albedo=0.30,
        ice_albedo=0.62,
        initial_co2_ppm=clamp(initial_co2, co2_lo, co2_hi),
        habitability_profile="Liquid Water",
        habitable_temp_min_c=0.0,
        habitable_temp_max_c=45.0,
        imported_equilibrium_temperature_k=eqt_k,
        imported_stellar_flux_multiplier=insol,
        import_classification=_classify_import(eqt_k),
        natural_planet_mode=True,
        atmosphere_assumption=atmosphere_assumption,
    )
    civ = CivilizationState(
        emissions_rate=0.0,
        emissions_growth_mode="Constant",
        mitigation_start_year=0,
        mitigation_strength=0.0,
    )
    return ScenarioModel(planet=planet, civilization=civ, preset_name=str(raw.get("pl_name", "")).strip() or None)


def normalize_imported_planet_flat(raw: dict[str, Any], atmosphere_assumption: str = "Minimal") -> dict[str, Any]:
    scenario = normalize_imported_planet(raw, atmosphere_assumption=atmosphere_assumption)
    return scenario_to_flat_params(scenario)

