"""Unified hard bounds used across UI, validation, imports, and simulation."""

from __future__ import annotations

from typing import Final

BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "stellar_flux_multiplier": (0.01, 1000.0),
    "warm_albedo": (0.0, 1.0),
    "ice_albedo": (0.0, 1.0),
    "initial_co2_ppm": (1.0, 100000.0),
    "emissions_rate": (0.0, 100.0),
    "mitigation_start_year": (0.0, 50000.0),
    "mitigation_strength": (0.0, 1.0),
    "habitable_temp_min_c": (-150.0, 1500.0),
    "habitable_temp_max_c": (-150.0, 1500.0),
    "K_CO2": (0.0, 20.0),
    "temperature_clamp_c": (-200.0, 2000.0),
    "seed": (0.0, 2147483647.0),
}

CO2_min: Final[float] = BOUNDS["initial_co2_ppm"][0]
CO2_max_hard: Final[float] = BOUNDS["initial_co2_ppm"][1]
temperature_clamp_min_c: Final[float] = BOUNDS["temperature_clamp_c"][0]
temperature_clamp_max_c: Final[float] = BOUNDS["temperature_clamp_c"][1]


def get_bounds(name: str) -> tuple[float, float]:
    return BOUNDS[name]

