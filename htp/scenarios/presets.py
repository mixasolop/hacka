"""Canonical scenario preset definitions and key sets."""

from __future__ import annotations

from htp.model.constants import DEFAULT_K_CO2

EXOPLANET_PRESET_PREFIX = "Exoplanet: "
EXOPLANET_ATMOSPHERE_OPTIONS = ("Minimal", "Earth-like", "Dense", "Custom")

REMOVED_UI_KEYS = (
    "planet_radius_km",
    "orbital_distance_au",
    "seasonality_amplitude",
    "obliquity_deg",
    "ice_transition_temp_c",
    "heat_transport_strength",
    "climate_heat_capacity",
    "volcanic_outgassing",
    "weathering_strength",
    "biosphere_uptake_strength",
    "carbon_removal_sensitivity",
    "carbon_capture_rate",
    "seasonality_enabled",
)

EARTHLIKE_PLANET_BASE = {
    "stellar_flux_multiplier": 1.00,
    "enable_seasonality": True,
    "warm_albedo": 0.30,
    "ice_albedo": 0.62,
    "initial_co2_ppm": 420.0,
    "habitability_profile": "Liquid Water",
    "habitable_temp_min_c": 0.0,
    "habitable_temp_max_c": 45.0,
    "K_CO2": DEFAULT_K_CO2,
    "seed": 42,
    "natural_planet_mode": False,
    "atmosphere_assumption": None,
    "imported_equilibrium_temperature_k": None,
    "imported_stellar_flux_multiplier": None,
    "import_classification": None,
}
EARTHLIKE_CIV_BASELINE = {
    "emissions_rate": 1.0,
    "emissions_growth_mode": "Constant",
    "mitigation_start_year": 40,
    "mitigation_strength": 0.15,
}
EARTHLIKE_CIV_CAREFREE = {
    "emissions_rate": 4.8,
    "emissions_growth_mode": "Carefree",
    "mitigation_start_year": 120,
    "mitigation_strength": 0.02,
}
EARTHLIKE_CIV_STABILIZATION = {
    "emissions_rate": 2.0,
    "emissions_growth_mode": "Stabilization",
    "mitigation_start_year": 5,
    "mitigation_strength": 0.78,
}

EARTHLIKE_SHARED_INITIAL_KEYS = tuple(EARTHLIKE_PLANET_BASE.keys())
EARTHLIKE_POLICY_KEYS = (
    "emissions_rate",
    "emissions_growth_mode",
    "mitigation_start_year",
    "mitigation_strength",
)

SCENARIO_INPUT_KEYS = (
    "stellar_flux_multiplier",
    "enable_seasonality",
    "warm_albedo",
    "ice_albedo",
    "initial_co2_ppm",
    "emissions_rate",
    "emissions_growth_mode",
    "mitigation_start_year",
    "mitigation_strength",
    "habitability_profile",
    "habitable_temp_min_c",
    "habitable_temp_max_c",
)
SCENARIO_HIDDEN_KEYS = (
    "K_CO2",
    "seed",
    "natural_planet_mode",
    "atmosphere_assumption",
    "imported_equilibrium_temperature_k",
    "imported_stellar_flux_multiplier",
    "import_classification",
)
SCENARIO_ALL_KEYS = SCENARIO_INPUT_KEYS + SCENARIO_HIDDEN_KEYS


def _merge_planet_and_civ(planet: dict[str, object], civ: dict[str, object]) -> dict[str, object]:
    merged = dict(planet)
    merged.update(civ)
    return merged


PRESETS = {
    "Earth-like Baseline": _merge_planet_and_civ(EARTHLIKE_PLANET_BASE, EARTHLIKE_CIV_BASELINE),
    "Carefree Civilization": _merge_planet_and_civ(EARTHLIKE_PLANET_BASE, EARTHLIKE_CIV_CAREFREE),
    "Stabilization Policy": _merge_planet_and_civ(EARTHLIKE_PLANET_BASE, EARTHLIKE_CIV_STABILIZATION),
    "Snowball-Prone World": _merge_planet_and_civ(
        {
            **EARTHLIKE_PLANET_BASE,
            "stellar_flux_multiplier": 0.86,
            "warm_albedo": 0.34,
            "ice_albedo": 0.78,
            "initial_co2_ppm": 260.0,
            "habitability_profile": "Conservative Biosphere",
            "habitable_temp_min_c": -5.0,
            "habitable_temp_max_c": 35.0,
            "seed": 43,
        },
        {
            "emissions_rate": 1.0,
            "emissions_growth_mode": "Constant",
            "mitigation_start_year": 40,
            "mitigation_strength": 0.25,
        },
    ),
    "Runaway-Prone World": _merge_planet_and_civ(
        {
            **EARTHLIKE_PLANET_BASE,
            "stellar_flux_multiplier": 1.12,
            "warm_albedo": 0.23,
            "ice_albedo": 0.52,
            "initial_co2_ppm": 700.0,
            "habitability_profile": "Broad Microbial Tolerance",
            "habitable_temp_min_c": 0.0,
            "habitable_temp_max_c": 50.0,
            "seed": 44,
        },
        {
            "emissions_rate": 5.0,
            "emissions_growth_mode": "Growing",
            "mitigation_start_year": 120,
            "mitigation_strength": 0.08,
        },
    ),
}


def is_exoplanet_preset_name(name: str) -> bool:
    return isinstance(name, str) and name.startswith(EXOPLANET_PRESET_PREFIX)


def is_known_preset_name(name: str) -> bool:
    return isinstance(name, str) and (name in PRESETS or is_exoplanet_preset_name(name))
