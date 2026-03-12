"""Pydantic schema for scenarios."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .bounds import BOUNDS
from .constants import (
    DEFAULT_K_CO2,
    DEFAULT_SEED,
    EMISSIONS_MODES,
    HABITABILITY_PROFILES,
    SCENARIO_VERSION,
)
from .safety import clamp, safe_float


def _clamp_bound(name: str, value: Any) -> float:
    lo, hi = BOUNDS[name]
    return clamp(safe_float(value, lo), lo, hi)


class PlanetState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    stellar_flux_multiplier: float = 1.0
    enable_seasonality: bool = True
    warm_albedo: float = 0.30
    ice_albedo: float = 0.62
    initial_co2_ppm: float = 420.0
    habitability_profile: str = "Liquid Water"
    habitable_temp_min_c: float = 0.0
    habitable_temp_max_c: float = 45.0
    K_CO2: float = DEFAULT_K_CO2
    seed: int = DEFAULT_SEED

    imported_equilibrium_temperature_k: float | None = None
    imported_stellar_flux_multiplier: float | None = None
    import_classification: str | None = None
    natural_planet_mode: bool = False
    atmosphere_assumption: str | None = None

    @field_validator("stellar_flux_multiplier", mode="before")
    @classmethod
    def _v_stellar(cls, v: Any) -> float:
        return _clamp_bound("stellar_flux_multiplier", v)

    @field_validator("warm_albedo", mode="before")
    @classmethod
    def _v_warm_albedo(cls, v: Any) -> float:
        return _clamp_bound("warm_albedo", v)

    @field_validator("ice_albedo", mode="before")
    @classmethod
    def _v_ice_albedo(cls, v: Any) -> float:
        return _clamp_bound("ice_albedo", v)

    @field_validator("initial_co2_ppm", mode="before")
    @classmethod
    def _v_initial_co2(cls, v: Any) -> float:
        return _clamp_bound("initial_co2_ppm", v)

    @field_validator("habitable_temp_min_c", mode="before")
    @classmethod
    def _v_hab_min(cls, v: Any) -> float:
        return _clamp_bound("habitable_temp_min_c", v)

    @field_validator("habitable_temp_max_c", mode="before")
    @classmethod
    def _v_hab_max(cls, v: Any) -> float:
        return _clamp_bound("habitable_temp_max_c", v)

    @field_validator("K_CO2", mode="before")
    @classmethod
    def _v_k_co2(cls, v: Any) -> float:
        return _clamp_bound("K_CO2", v)

    @field_validator("seed", mode="before")
    @classmethod
    def _v_seed(cls, v: Any) -> int:
        lo, hi = BOUNDS["seed"]
        return int(round(clamp(safe_float(v, lo), lo, hi)))

    @field_validator("habitability_profile", mode="before")
    @classmethod
    def _v_profile(cls, v: Any) -> str:
        value = str(v) if v is not None else "Liquid Water"
        return value if value in HABITABILITY_PROFILES else "Liquid Water"

    @field_validator("imported_equilibrium_temperature_k", mode="before")
    @classmethod
    def _v_imported_teq(cls, v: Any) -> float | None:
        if v is None:
            return None
        value = safe_float(v, float("nan"))
        return value if value == value else None

    @field_validator("imported_stellar_flux_multiplier", mode="before")
    @classmethod
    def _v_imported_flux(cls, v: Any) -> float | None:
        if v is None:
            return None
        value = safe_float(v, float("nan"))
        return value if value == value else None

    @field_validator("import_classification", mode="before")
    @classmethod
    def _v_classification(cls, v: Any) -> str | None:
        if v is None:
            return None
        normalized = str(v).strip().lower()
        if normalized in {"hot", "cold", "temperate", "unknown"}:
            return normalized
        return "unknown"

    @field_validator("atmosphere_assumption", mode="before")
    @classmethod
    def _v_assumption(cls, v: Any) -> str | None:
        if v is None:
            return None
        value = str(v).strip()
        if value in {"Minimal", "Earth-like", "Dense", "Custom"}:
            return value
        return None

    @model_validator(mode="after")
    def _normalize_habitability_range(self) -> "PlanetState":
        if self.habitable_temp_max_c <= self.habitable_temp_min_c:
            lo, hi = BOUNDS["habitable_temp_max_c"]
            self.habitable_temp_max_c = clamp(self.habitable_temp_min_c + 1.0, lo, hi)
            if self.habitable_temp_max_c <= self.habitable_temp_min_c:
                self.habitable_temp_min_c = clamp(self.habitable_temp_max_c - 1.0, *BOUNDS["habitable_temp_min_c"])
        return self


class CivilizationState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emissions_rate: float = 1.0
    emissions_growth_mode: str = "Constant"
    mitigation_start_year: int = 40
    mitigation_strength: float = 0.15

    @field_validator("emissions_rate", mode="before")
    @classmethod
    def _v_emissions(cls, v: Any) -> float:
        return _clamp_bound("emissions_rate", v)

    @field_validator("mitigation_start_year", mode="before")
    @classmethod
    def _v_mitigation_start(cls, v: Any) -> int:
        lo, hi = BOUNDS["mitigation_start_year"]
        return int(round(clamp(safe_float(v, lo), lo, hi)))

    @field_validator("mitigation_strength", mode="before")
    @classmethod
    def _v_mitigation_strength(cls, v: Any) -> float:
        return _clamp_bound("mitigation_strength", v)

    @field_validator("emissions_growth_mode", mode="before")
    @classmethod
    def _v_mode(cls, v: Any) -> str:
        mode = str(v) if v is not None else "Constant"
        return mode if mode in EMISSIONS_MODES else "Constant"


class ScenarioModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    planet: PlanetState = Field(default_factory=PlanetState)
    civilization: CivilizationState = Field(default_factory=CivilizationState)
    preset_name: str | None = None
    scenario_version: str = SCENARIO_VERSION
    created_at: str | None = None

    @model_validator(mode="after")
    def _normalize_model(self) -> "ScenarioModel":
        if not self.scenario_version:
            self.scenario_version = SCENARIO_VERSION
        if self.created_at == "":
            self.created_at = None
        return self


def default_scenario_model() -> ScenarioModel:
    return ScenarioModel(created_at=datetime.now(timezone.utc).isoformat())

