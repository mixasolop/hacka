"""Scenario parsing, conversion, signatures, and JSON roundtrips."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping

from .constants import SCENARIO_VERSION
from .schema import ScenarioModel, default_scenario_model

PLANET_KEYS = {
    "stellar_flux_multiplier",
    "enable_seasonality",
    "warm_albedo",
    "ice_albedo",
    "initial_co2_ppm",
    "habitability_profile",
    "habitable_temp_min_c",
    "habitable_temp_max_c",
    "K_CO2",
    "seed",
    "imported_equilibrium_temperature_k",
    "imported_stellar_flux_multiplier",
    "import_classification",
    "natural_planet_mode",
    "atmosphere_assumption",
}

CIV_KEYS = {
    "emissions_rate",
    "emissions_growth_mode",
    "mitigation_start_year",
    "mitigation_strength",
}

LEGACY_FLAT_KEYS = PLANET_KEYS | CIV_KEYS


def _scenario_dict_from_flat(flat: Mapping[str, Any]) -> dict[str, Any]:
    planet = {k: flat.get(k) for k in PLANET_KEYS if k in flat}
    civ = {k: flat.get(k) for k in CIV_KEYS if k in flat}
    return {
        "planet": planet,
        "civilization": civ,
        "preset_name": flat.get("preset_name"),
        "scenario_version": flat.get("scenario_version", SCENARIO_VERSION),
        "created_at": flat.get("created_at"),
    }


def _unwrap_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, ScenarioModel):
        return raw.model_dump(mode="json")
    if not isinstance(raw, Mapping):
        return {}

    if "planet" in raw and "civilization" in raw:
        return dict(raw)
    if "scenario" in raw and isinstance(raw["scenario"], Mapping):
        nested = dict(raw["scenario"])
        if "planet" in nested and "civilization" in nested:
            return nested
        if LEGACY_FLAT_KEYS.intersection(nested.keys()):
            return _scenario_dict_from_flat(nested)
    if LEGACY_FLAT_KEYS.intersection(raw.keys()):
        return _scenario_dict_from_flat(raw)
    return {}


def scenario_from_any(raw: Any) -> ScenarioModel:
    if isinstance(raw, ScenarioModel):
        return raw
    payload = _unwrap_payload(raw)
    if payload:
        return ScenarioModel.model_validate(payload)
    return default_scenario_model()


def scenario_from_flat_params(flat: Mapping[str, Any], *, preset_name: str | None = None) -> ScenarioModel:
    data = _scenario_dict_from_flat(flat)
    if preset_name is not None:
        data["preset_name"] = preset_name
    return ScenarioModel.model_validate(data)


def load_scenario_from_session(session_state: Mapping[str, Any]) -> ScenarioModel:
    submitted_snapshot = session_state.get("submitted_scenario_snapshot")
    params_payload = session_state.get("params")

    for candidate in (submitted_snapshot, params_payload):
        payload = _unwrap_payload(candidate)
        if payload:
            if payload.get("preset_name") is None:
                payload["preset_name"] = session_state.get("builder_persisted_preset_name")
            return ScenarioModel.model_validate(payload)

    fallback = default_scenario_model()
    fallback.preset_name = str(session_state.get("builder_persisted_preset_name") or "") or None
    return fallback


def scenario_to_flat_params(scenario: ScenarioModel) -> dict[str, Any]:
    s = scenario_from_any(scenario)
    flat = {
        "stellar_flux_multiplier": s.planet.stellar_flux_multiplier,
        "enable_seasonality": s.planet.enable_seasonality,
        "warm_albedo": s.planet.warm_albedo,
        "ice_albedo": s.planet.ice_albedo,
        "initial_co2_ppm": s.planet.initial_co2_ppm,
        "habitability_profile": s.planet.habitability_profile,
        "habitable_temp_min_c": s.planet.habitable_temp_min_c,
        "habitable_temp_max_c": s.planet.habitable_temp_max_c,
        "K_CO2": s.planet.K_CO2,
        "seed": s.planet.seed,
        "imported_equilibrium_temperature_k": s.planet.imported_equilibrium_temperature_k,
        "imported_stellar_flux_multiplier": s.planet.imported_stellar_flux_multiplier,
        "import_classification": s.planet.import_classification,
        "natural_planet_mode": s.planet.natural_planet_mode,
        "atmosphere_assumption": s.planet.atmosphere_assumption,
        "emissions_rate": s.civilization.emissions_rate,
        "emissions_growth_mode": s.civilization.emissions_growth_mode,
        "mitigation_start_year": s.civilization.mitigation_start_year,
        "mitigation_strength": s.civilization.mitigation_strength,
        "preset_name": s.preset_name,
        "scenario_version": s.scenario_version,
        "created_at": s.created_at,
    }
    return flat


def scenario_signature(scenario: ScenarioModel | Mapping[str, Any] | dict[str, Any]) -> str:
    s = scenario_from_any(scenario)
    payload = s.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def export_scenario_json(scenario: ScenarioModel | Mapping[str, Any]) -> str:
    s = scenario_from_any(scenario)
    return json.dumps(s.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False)


def import_scenario_json(text: str) -> ScenarioModel:
    payload = json.loads(text)
    return scenario_from_any(payload)
