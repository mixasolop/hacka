"""Canonical Earth geometry and surface model shared by all screens."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

CANONICAL_LAT_MIN_DEG = -90.0
CANONICAL_LAT_MAX_DEG = 90.0
CANONICAL_LON_MIN_DEG = -180.0
CANONICAL_LON_MAX_DEG = 180.0

SURFACE_CLASS_OCEAN = 0
SURFACE_CLASS_TEMPERATE = 1
SURFACE_CLASS_DESERT = 2
SURFACE_CLASS_SNOW_ICE = 3

SURFACE_CLASS_VEGETATED = SURFACE_CLASS_TEMPERATE

EARTH_LAND_MASK_PATH = Path(__file__).resolve().parent / "data" / "earth_land_mask_360x720.npz"


@lru_cache(maxsize=1)
def _load_land_mask() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not EARTH_LAND_MASK_PATH.exists():
        raise FileNotFoundError(f"Missing Earth land mask file: {EARTH_LAND_MASK_PATH}")

    with np.load(EARTH_LAND_MASK_PATH) as data:
        lat = np.asarray(data["lat"], dtype=float)
        lon = np.asarray(data["lon"], dtype=float)
        land_mask = np.asarray(data["land_mask"], dtype=float)

    lon = wrap_longitude_deg(lon)
    land_mask = land_mask >= 0.5

    if land_mask.shape != (lat.size, lon.size):
        raise ValueError(
            f"Invalid land mask shape {land_mask.shape}; expected {(lat.size, lon.size)}"
        )

    return lat, lon, land_mask


def wrap_longitude_deg(lon_deg: np.ndarray | float) -> np.ndarray:
    lon = np.asarray(lon_deg, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def latlon_to_xyz(
    lat_deg: np.ndarray | float,
    lon_deg: np.ndarray | float,
    radius: np.ndarray | float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.clip(np.asarray(lat_deg, dtype=float), CANONICAL_LAT_MIN_DEG, CANONICAL_LAT_MAX_DEG)
    lon = wrap_longitude_deg(np.asarray(lon_deg, dtype=float))
    r = np.asarray(radius, dtype=float)

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    cos_lat = np.cos(lat_rad)
    x = r * cos_lat * np.cos(lon_rad)
    y = r * cos_lat * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return x, y, z


def surface_grid_to_xyz(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    radius: np.ndarray | float = 1.0,
) -> dict[str, np.ndarray]:
    lat_1d = np.asarray(lat_deg, dtype=float)
    lon_1d = np.asarray(lon_deg, dtype=float)

    lon2_deg, lat2_deg = np.meshgrid(lon_1d, lat_1d)
    x, y, z = latlon_to_xyz(lat2_deg, lon2_deg, radius=radius)

    return {
        "lat2_deg": np.asarray(lat2_deg, dtype=float),
        "lon2_deg": np.asarray(lon2_deg, dtype=float),
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=float),
        "z": np.asarray(z, dtype=float),
    }


def prepare_surface_render_grid(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    *grid_fields: np.ndarray,
    add_polar_caps: bool = True,
    wrap_longitude: bool = True,
) -> tuple[np.ndarray, ...]:
    """
    Close a render grid for globe plotting.

    Important for Plotly:
    - duplicate the first longitude column at +360°
    - add explicit polar cap rows based on nearest valid rows
    """
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    fields = [np.asarray(field) for field in grid_fields]

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("lat_deg and lon_deg must be 1D arrays")

    for field in fields:
        if field.shape != (lat.size, lon.size):
            raise ValueError(
                f"Grid field shape {field.shape} does not match expected {(lat.size, lon.size)}"
            )

    if add_polar_caps:
        lat = np.concatenate((
            np.array([CANONICAL_LAT_MIN_DEG], dtype=float),
            lat,
            np.array([CANONICAL_LAT_MAX_DEG], dtype=float),
        ))
        new_fields: list[np.ndarray] = []
        for field in fields:
            south_cap = field[0:1, :].copy()
            north_cap = field[-1:, :].copy()
            new_fields.append(np.vstack((south_cap, field, north_cap)))
        fields = new_fields

    if wrap_longitude:
        lon = np.concatenate((lon, np.array([lon[0] + 360.0], dtype=float)))
        fields = [
            np.concatenate((field, field[:, :1]), axis=1)
            for field in fields
        ]

    return (lat, lon, *fields)


def earth_land_mask_sampled(lat_deg: np.ndarray | float, lon_deg: np.ndarray | float) -> np.ndarray:
    lat_grid, lon_grid, land_mask = _load_land_mask()

    lat_arr, lon_arr = np.broadcast_arrays(
        np.asarray(lat_deg, dtype=float),
        np.asarray(lon_deg, dtype=float),
    )
    lat_arr = np.clip(lat_arr, CANONICAL_LAT_MIN_DEG, CANONICAL_LAT_MAX_DEG)
    lon_arr = wrap_longitude_deg(lon_arr)

    nlat = lat_grid.size
    nlon = lon_grid.size

    lat_pos = (lat_arr - CANONICAL_LAT_MIN_DEG) / (CANONICAL_LAT_MAX_DEG - CANONICAL_LAT_MIN_DEG)
    lon_pos = (lon_arr - CANONICAL_LON_MIN_DEG) / (CANONICAL_LON_MAX_DEG - CANONICAL_LON_MIN_DEG)

    lat_idx = np.clip(np.floor(lat_pos * nlat).astype(int), 0, nlat - 1)
    lon_idx = np.mod(np.floor(lon_pos * nlon).astype(int), nlon)

    return land_mask[lat_idx, lon_idx]


def _gaussian(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    center_lat: float,
    center_lon: float,
    sigma_lat: float,
    sigma_lon: float,
) -> np.ndarray:
    dlat = (lat_deg - center_lat) / sigma_lat
    dlon = wrap_longitude_deg(lon_deg - center_lon) / sigma_lon
    return np.exp(-0.5 * (dlat * dlat + dlon * dlon))


def _desert_score(lat_deg: np.ndarray, lon_deg: np.ndarray, land: np.ndarray) -> np.ndarray:
    abs_lat = np.abs(lat_deg)

    subtropical_belt = np.exp(-0.5 * ((abs_lat - 24.0) / 8.5) ** 2)
    score = 0.55 * subtropical_belt

    score += 1.10 * _gaussian(lat_deg, lon_deg, 23.0, 13.0, 9.5, 18.0)    # Sahara
    score += 0.80 * _gaussian(lat_deg, lon_deg, 24.0, 45.0, 8.0, 14.0)    # Arabia
    score += 0.45 * _gaussian(lat_deg, lon_deg, 42.0, 103.0, 7.0, 11.0)   # Gobi
    score += 0.75 * _gaussian(lat_deg, lon_deg, -24.0, 133.0, 9.0, 14.0)  # Australia
    score += 0.55 * _gaussian(lat_deg, lon_deg, -23.0, -69.0, 6.0, 8.0)   # Atacama
    score += 0.35 * _gaussian(lat_deg, lon_deg, -23.0, 22.0, 7.0, 10.0)   # Kalahari
    score += 0.32 * _gaussian(lat_deg, lon_deg, 34.0, -112.0, 7.0, 10.0)  # SW US / N Mexico

    score = np.where(land, score, 0.0)
    return np.clip(score, 0.0, 2.0)


def _elevation_field(lat_rad: np.ndarray, lon_rad: np.ndarray, land: np.ndarray) -> np.ndarray:
    wave_1 = 0.5 + 0.5 * np.sin(1.6 * lon_rad + 0.5 * np.sin(2.1 * lat_rad))
    wave_2 = 0.5 + 0.5 * np.cos(3.4 * lon_rad - 1.7 * lat_rad)
    ridge = np.abs(np.sin(2.9 * lon_rad) * np.cos(2.0 * lat_rad))

    terrain = np.clip(0.45 * wave_1 + 0.30 * wave_2 + 0.25 * ridge, 0.0, 1.0)
    ocean_depth = np.clip(0.35 + 0.65 * (1.0 - wave_1), 0.0, 1.0)

    elevation_km = np.where(
        land,
        0.05 + 1.2 * terrain + 0.9 * ridge * ridge,
        -0.35 - 4.8 * ocean_depth,
    )
    return np.clip(elevation_km, -6.0, 5.5)


def earth_surface_fields(
    lon_rad: np.ndarray,
    lat_rad: np.ndarray,
    temp_c: float,
) -> dict[str, np.ndarray]:
    lon_rad = np.asarray(lon_rad, dtype=float)
    lat_rad = np.asarray(lat_rad, dtype=float)

    if lon_rad.shape != lat_rad.shape:
        raise ValueError("lon_rad and lat_rad must have the same shape")

    lon_deg = wrap_longitude_deg(np.rad2deg(lon_rad))
    lat_deg = np.clip(np.rad2deg(lat_rad), CANONICAL_LAT_MIN_DEG, CANONICAL_LAT_MAX_DEG)

    land = earth_land_mask_sampled(lat_deg, lon_deg)
    elevation_km = _elevation_field(lat_rad, lon_rad, land)

    # Stable polar logic
    antarctica = land & (lat_deg <= -60.0)
    greenland = land & (lat_deg >= 60.0) & (lat_deg <= 85.0) & (lon_deg >= -74.0) & (lon_deg <= -10.0)

    polar_limit_deg = np.clip(73.0 + 0.15 * float(temp_c), 67.0, 84.0)
    polar_belt = land & (np.abs(lat_deg) >= polar_limit_deg)
    mountain_snow = land & (np.abs(lat_deg) >= 50.0) & (elevation_km >= 3.4)

    snow_ice = antarctica | greenland | polar_belt | mountain_snow

    # Force the very last southern ring to snow on land so the south cap never opens visually
    south_cap_land = land & (lat_deg <= -88.0)
    snow_ice = snow_ice | south_cap_land

    desert_score = _desert_score(lat_deg, lon_deg, land)
    desert_lat_ok = (np.abs(lat_deg) >= 8.0) & (np.abs(lat_deg) <= 38.0)
    desert = land & (~snow_ice) & desert_lat_ok & (desert_score >= 0.95)

    temperate = land & (~snow_ice) & (~desert)

    land_norm = np.clip((elevation_km - 0.05) / 5.45, 0.0, 1.0)
    ocean_norm = np.clip((-elevation_km - 0.35) / 5.65, 0.0, 1.0)

    ocean_texture = 0.10 + 0.16 * (1.0 - ocean_norm)

    # More vivid green for vegetated land
    temperate_texture = 0.56 + 0.18 * land_norm

    desert_texture = 0.72 + 0.07 * np.clip(desert_score / 1.6, 0.0, 1.0)
    snow_texture = 0.95 + 0.015 * np.clip(np.abs(lat_deg) / 90.0, 0.0, 1.0)

    surface_texture = np.where(land, temperate_texture, ocean_texture)
    surface_texture = np.where(desert, desert_texture, surface_texture)
    surface_texture = np.where(snow_ice, snow_texture, surface_texture)
    surface_texture = np.clip(surface_texture, 0.0, 1.0)

    relief = np.where(land, 0.010 + 0.045 * land_norm, -0.004 * ocean_norm)
    relief = np.where(snow_ice, relief + 0.008, relief)

    lat_cooling = -0.012 * np.abs(lat_deg)
    zonal_structure = 0.55 * np.sin(2.4 * lon_rad) * np.cos(lat_rad)
    meridional_structure = 0.18 * np.sin(4.0 * lat_rad)
    elevation_cooling = -0.28 * np.clip(elevation_km, 0.0, 6.0)
    ocean_softening = 0.08 * np.clip(-elevation_km, 0.0, 6.0)

    microclimate_c = lat_cooling + zonal_structure + meridional_structure + elevation_cooling + ocean_softening
    microclimate_c = np.where(desert, microclimate_c + 2.0, microclimate_c)
    microclimate_c = np.where(snow_ice, microclimate_c - 3.0, microclimate_c)

    surface_class = np.full(land.shape, SURFACE_CLASS_OCEAN, dtype=np.int8)
    surface_class = np.where(temperate, SURFACE_CLASS_TEMPERATE, surface_class)
    surface_class = np.where(desert, SURFACE_CLASS_DESERT, surface_class)
    surface_class = np.where(snow_ice, SURFACE_CLASS_SNOW_ICE, surface_class)

    return {
        "land_mask": np.asarray(land, dtype=bool),
        "temperate_mask": np.asarray(temperate, dtype=bool),
        "vegetated_mask": np.asarray(temperate, dtype=bool),
        "desert_mask": np.asarray(desert, dtype=bool),
        "snow_ice_mask": np.asarray(snow_ice, dtype=bool),
        "surface_class": np.asarray(surface_class, dtype=np.int8),
        "surface_texture": np.asarray(surface_texture, dtype=float),
        "relief": np.asarray(relief, dtype=float),
        "elevation_km": np.asarray(elevation_km, dtype=float),
        "microclimate_c": np.asarray(microclimate_c, dtype=float),
    }


def earth_geography_grid(
    texture_temp_c: float,
    nlat: int = 72,
    nlon: int = 144,
) -> dict[str, np.ndarray]:
    nlat_i = max(8, int(nlat))
    nlon_i = max(16, int(nlon))

    lat_step = 180.0 / nlat_i
    lon_step = 360.0 / nlon_i

    # Move slightly away from exact poles to reduce cap artefacts in the base grid.
    lat_deg = np.linspace(
        -89.0,
        89.0,
        nlat_i,
        dtype=float,
    )
    lon_deg = np.linspace(
        CANONICAL_LON_MIN_DEG + 0.5 * lon_step,
        CANONICAL_LON_MAX_DEG - 0.5 * lon_step,
        nlon_i,
        dtype=float,
    )

    lon2_rad, lat2_rad = np.meshgrid(np.deg2rad(lon_deg), np.deg2rad(lat_deg))
    fields = earth_surface_fields(lon2_rad, lat2_rad, temp_c=float(texture_temp_c))

    return {
        "lat_deg": np.asarray(lat_deg, dtype=float),
        "lon_deg": np.asarray(lon_deg, dtype=float),
        "elevation_km": np.asarray(fields["elevation_km"], dtype=float),
        "microclimate_c": np.asarray(fields["microclimate_c"], dtype=float),
        "land_mask": np.asarray(fields["land_mask"], dtype=bool),
        "temperate_mask": np.asarray(fields["temperate_mask"], dtype=bool),
        "vegetated_mask": np.asarray(fields["vegetated_mask"], dtype=bool),
        "desert_mask": np.asarray(fields["desert_mask"], dtype=bool),
        "snow_ice_mask": np.asarray(fields["snow_ice_mask"], dtype=bool),
        "surface_class": np.asarray(fields["surface_class"], dtype=np.int8),
        "surface_texture": np.asarray(fields["surface_texture"], dtype=float),
        "relief": np.asarray(fields["relief"], dtype=float),
    }


def visual_alignment_report(
    land_mask: np.ndarray,
    surface_class: np.ndarray,
) -> dict[str, float | int | bool]:
    land = np.asarray(land_mask, dtype=bool)
    visible_land = np.asarray(surface_class, dtype=int) != SURFACE_CLASS_OCEAN

    if land.shape != visible_land.shape:
        raise ValueError(
            f"Shape mismatch: land_mask {land.shape} vs surface_class {visible_land.shape}"
        )

    direct_match = float(np.mean(land == visible_land))
    mirror_match = float(np.mean(land == np.flip(visible_land, axis=-1)))

    phase_scores = [
        float(np.mean(land == np.roll(visible_land, shift=shift, axis=-1)))
        for shift in range(int(land.shape[-1]))
    ]
    best_shift = int(np.argmax(phase_scores))
    mismatches = land ^ visible_land

    return {
        "cells": int(land.size),
        "mismatch_cells": int(np.sum(mismatches)),
        "mismatch_fraction": float(np.mean(mismatches)),
        "direct_match_fraction": direct_match,
        "mirror_match_fraction": mirror_match,
        "is_lon_mirrored": bool(mirror_match > direct_match + 1e-12),
        "best_lon_phase_shift_cells": best_shift,
        "best_lon_phase_shift_deg": float(best_shift * 360.0 / max(1, int(land.shape[-1]))),
    }