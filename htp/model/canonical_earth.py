"""Canonical Earth model shared by rendering and local map calculations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

CANONICAL_LAT_MIN_DEG = -90.0
CANONICAL_LAT_MAX_DEG = 90.0
CANONICAL_LON_MIN_DEG = -180.0
CANONICAL_LON_MAX_DEG = 180.0

SURFACE_CLASS_OCEAN = 0
SURFACE_CLASS_VEGETATED = 1
SURFACE_CLASS_DESERT = 2
SURFACE_CLASS_SNOW_ICE = 3

# Backward-compatible alias used by existing imports.
SURFACE_CLASS_TEMPERATE = SURFACE_CLASS_VEGETATED

EARTH_LAND_MASK_PATH = Path(__file__).resolve().parent / "data" / "earth_land_mask_360x720.npz"


@dataclass(frozen=True)
class EarthSurfaceMesh:
    radius_scale: float
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    surface_texture: np.ndarray
    relief: np.ndarray
    elevation_km: np.ndarray
    land_mask: np.ndarray
    vegetated_mask: np.ndarray
    desert_mask: np.ndarray
    snow_ice_mask: np.ndarray
    surface_class: np.ndarray
    microclimate_c: np.ndarray


@lru_cache(maxsize=1)
def _load_land_mask() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not EARTH_LAND_MASK_PATH.exists():
        raise FileNotFoundError(f"Missing Earth land mask file: {EARTH_LAND_MASK_PATH}")
    with np.load(EARTH_LAND_MASK_PATH) as data:
        lat = np.asarray(data["lat"], dtype=float)
        lon = np.asarray(data["lon"], dtype=float)
        land_mask = np.asarray(data["land_mask"], dtype=float) >= 0.5

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Earth land mask lat/lon axes must be 1D")
    if land_mask.shape != (lat.size, lon.size):
        raise ValueError(
            f"Invalid Earth land mask shape {land_mask.shape}; expected {(lat.size, lon.size)}"
        )
    if lat.size < 2 or lon.size < 2:
        raise ValueError("Earth land mask axes are too small")
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
    rad = np.asarray(radius, dtype=float)

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    cos_lat = np.cos(lat_rad)
    x = rad * cos_lat * np.cos(lon_rad)
    y = rad * cos_lat * np.sin(lon_rad)
    z = rad * np.sin(lat_rad)
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
    """Compatibility helper that closes a 2D lat/lon grid for rendering."""
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
        lat = np.concatenate(
            (
                np.array([CANONICAL_LAT_MIN_DEG], dtype=float),
                lat,
                np.array([CANONICAL_LAT_MAX_DEG], dtype=float),
            )
        )
        fields = [np.vstack((field[0:1, :], field, field[-1:, :])) for field in fields]

    if wrap_longitude:
        lon = np.concatenate((lon, np.array([lon[0] + 360.0], dtype=float)))
        fields = [np.concatenate((field, field[:, :1]), axis=1) for field in fields]

    return (lat, lon, *fields)


def earth_land_mask_sampled(lat_deg: np.ndarray | float, lon_deg: np.ndarray | float) -> np.ndarray:
    lat_axis, lon_axis, land_mask = _load_land_mask()
    lat_arr, lon_arr = np.broadcast_arrays(
        np.asarray(lat_deg, dtype=float),
        np.asarray(lon_deg, dtype=float),
    )
    lat_arr = np.clip(lat_arr, CANONICAL_LAT_MIN_DEG, CANONICAL_LAT_MAX_DEG)
    lon_arr = wrap_longitude_deg(lon_arr)

    lat_step = float(lat_axis[1] - lat_axis[0])
    lon_step = float(lon_axis[1] - lon_axis[0])

    lat_idx = np.rint((lat_arr - float(lat_axis[0])) / lat_step).astype(int)
    lon_idx = np.rint((lon_arr - float(lon_axis[0])) / lon_step).astype(int)
    lat_idx = np.clip(lat_idx, 0, lat_axis.size - 1)
    lon_idx = np.mod(lon_idx, lon_axis.size)
    return land_mask[lat_idx, lon_idx]


def _gaussian_region(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    center_lat: float,
    center_lon: float,
    sigma_lat: float,
    sigma_lon: float,
) -> np.ndarray:
    dlat = (lat_deg - float(center_lat)) / float(sigma_lat)
    dlon = wrap_longitude_deg(lon_deg - float(center_lon)) / float(sigma_lon)
    return np.exp(-0.5 * (dlat * dlat + dlon * dlon))


def _mountain_intensity(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    score = np.zeros_like(lat_deg, dtype=float)
    score += 1.00 * _gaussian_region(lat_deg, lon_deg, -20.0, -70.0, 12.0, 8.0)   # Andes
    score += 0.78 * _gaussian_region(lat_deg, lon_deg, 42.0, -112.0, 10.0, 12.0)  # Rockies
    score += 1.25 * _gaussian_region(lat_deg, lon_deg, 32.0, 85.0, 7.0, 14.0)     # Himalaya/Tibet
    score += 0.68 * _gaussian_region(lat_deg, lon_deg, 46.0, 12.0, 6.0, 7.0)      # Alps/Caucasus
    score += 0.52 * _gaussian_region(lat_deg, lon_deg, 8.0, 39.0, 8.0, 7.0)       # East Africa
    score += 0.44 * _gaussian_region(lat_deg, lon_deg, -44.0, 170.0, 5.0, 6.0)    # New Zealand
    return np.clip(score, 0.0, 2.5)


def _elevation_field_km(lat_deg: np.ndarray, lon_deg: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    wave_a = 0.5 + 0.5 * np.sin(1.5 * lon_rad + 0.6 * np.sin(2.3 * lat_rad))
    wave_b = 0.5 + 0.5 * np.cos(3.0 * lon_rad - 1.2 * lat_rad)
    mountain = _mountain_intensity(lat_deg, lon_deg)

    land_elevation = 0.08 + 1.35 * wave_a + 0.95 * wave_b + 2.85 * mountain
    land_elevation = np.clip(land_elevation, 0.02, 6.4)

    abyss = np.clip(0.45 + 0.55 * (1.0 - wave_b), 0.0, 1.0)
    ocean_depth = -0.35 - 5.75 * abyss
    ocean_depth = np.clip(ocean_depth, -6.2, -0.05)

    return np.where(land_mask, land_elevation, ocean_depth)


def _desert_score(lat_deg: np.ndarray, lon_deg: np.ndarray, land_mask: np.ndarray, elevation_km: np.ndarray) -> np.ndarray:
    abs_lat = np.abs(lat_deg)
    subtropical = np.exp(-0.5 * ((abs_lat - 24.0) / 7.5) ** 2)
    score = 0.58 * subtropical

    score += 1.10 * _gaussian_region(lat_deg, lon_deg, 23.0, 13.0, 9.0, 18.0)    # Sahara
    score += 0.82 * _gaussian_region(lat_deg, lon_deg, 24.0, 45.0, 8.0, 13.0)    # Arabia
    score += 0.75 * _gaussian_region(lat_deg, lon_deg, -24.0, 133.0, 10.0, 15.0) # Australia
    score += 0.60 * _gaussian_region(lat_deg, lon_deg, -23.0, -69.0, 6.0, 9.0)   # Atacama
    score += 0.46 * _gaussian_region(lat_deg, lon_deg, 42.0, 103.0, 7.0, 11.0)   # Gobi
    score += 0.35 * _gaussian_region(lat_deg, lon_deg, 34.0, -112.0, 7.0, 11.0)  # SW North America
    score += 0.33 * _gaussian_region(lat_deg, lon_deg, -23.0, 22.0, 8.0, 11.0)   # Kalahari/Namib

    mountain_penalty = 0.25 * np.clip(elevation_km, 0.0, 4.0)
    score = np.where(land_mask, score - mountain_penalty, 0.0)
    return np.clip(score, 0.0, 2.2)


def _surface_classification(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    land_mask: np.ndarray,
    elevation_km: np.ndarray,
    temp_c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    polar_limit_deg = float(np.clip(72.0 + 0.30 * (float(temp_c) - 15.0), 64.0, 83.0))

    antarctica = land_mask & (lat_deg <= -60.0)
    greenland = land_mask & (lat_deg >= 60.0) & (lat_deg <= 85.0) & (lon_deg >= -74.0) & (lon_deg <= -10.0)
    polar_land = land_mask & (np.abs(lat_deg) >= polar_limit_deg)
    mid_lat_mountain_snow = land_mask & (elevation_km >= 4.8) & (np.abs(lat_deg) >= 42.0)
    low_lat_mountain_snow = land_mask & (elevation_km >= 5.6) & (np.abs(lat_deg) >= 25.0)
    mountain_snow = mid_lat_mountain_snow | low_lat_mountain_snow
    snow_ice_mask = antarctica | greenland | polar_land | mountain_snow

    desert_score = _desert_score(lat_deg, lon_deg, land_mask, elevation_km)
    desert_lat_ok = (np.abs(lat_deg) >= 8.0) & (np.abs(lat_deg) <= 42.0)
    desert_mask = land_mask & (~snow_ice_mask) & desert_lat_ok & (desert_score >= 0.95)

    vegetated_mask = land_mask & (~snow_ice_mask) & (~desert_mask)
    return vegetated_mask, desert_mask, snow_ice_mask


def _surface_texture_field(
    lat_deg: np.ndarray,
    elevation_km: np.ndarray,
    vegetated_mask: np.ndarray,
    desert_mask: np.ndarray,
    snow_ice_mask: np.ndarray,
) -> np.ndarray:
    land_norm = np.clip(elevation_km / 6.4, 0.0, 1.0)
    ocean_norm = np.clip((-elevation_km) / 6.2, 0.0, 1.0)
    lat_norm = np.clip(np.abs(lat_deg) / 90.0, 0.0, 1.0)

    ocean_texture = 0.07 + 0.18 * (1.0 - ocean_norm)
    vegetated_texture = 0.54 + 0.18 * land_norm + 0.05 * (1.0 - lat_norm)
    desert_texture = 0.72 + 0.10 * (1.0 - lat_norm)
    snow_texture = 0.93 + 0.06 * lat_norm

    texture = np.array(ocean_texture, dtype=float, copy=True)
    texture = np.where(vegetated_mask, vegetated_texture, texture)
    texture = np.where(desert_mask, desert_texture, texture)
    texture = np.where(snow_ice_mask, snow_texture, texture)
    return np.clip(texture, 0.0, 1.0)


def _relief_field(elevation_km: np.ndarray, land_mask: np.ndarray, snow_ice_mask: np.ndarray) -> np.ndarray:
    land_relief = 0.006 + 0.030 * np.clip(elevation_km / 6.4, 0.0, 1.0)
    ocean_relief = -0.006 * np.clip((-elevation_km) / 6.2, 0.0, 1.0)
    relief = np.where(land_mask, land_relief, ocean_relief)
    relief = np.where(snow_ice_mask, relief + 0.004, relief)
    return np.clip(relief, -0.02, 0.05)


def _microclimate_field_c(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    elevation_km: np.ndarray,
    desert_mask: np.ndarray,
    snow_ice_mask: np.ndarray,
) -> np.ndarray:
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    lat_component = -0.012 * np.abs(lat_deg)
    zonal_component = 0.65 * np.sin(2.2 * lon_rad) * np.cos(lat_rad)
    meridional_component = 0.24 * np.sin(4.0 * lat_rad)
    elevation_component = -0.58 * np.clip(elevation_km, 0.0, 6.0)
    marine_component = 0.10 * np.clip(-elevation_km, 0.0, 6.0)

    micro = lat_component + zonal_component + meridional_component + elevation_component + marine_component
    micro = np.where(desert_mask, micro + 2.2, micro)
    micro = np.where(snow_ice_mask, micro - 3.4, micro)
    return np.clip(micro, -18.0, 8.0)


def earth_surface_fields(
    lon_rad: np.ndarray,
    lat_rad: np.ndarray,
    temp_c: float,
) -> dict[str, np.ndarray]:
    lon_rad_arr = np.asarray(lon_rad, dtype=float)
    lat_rad_arr = np.asarray(lat_rad, dtype=float)
    if lon_rad_arr.shape != lat_rad_arr.shape:
        raise ValueError("lon_rad and lat_rad must have the same shape")

    lon_deg = wrap_longitude_deg(np.rad2deg(lon_rad_arr))
    lat_deg = np.clip(np.rad2deg(lat_rad_arr), CANONICAL_LAT_MIN_DEG, CANONICAL_LAT_MAX_DEG)

    land_mask = earth_land_mask_sampled(lat_deg, lon_deg)
    elevation_km = _elevation_field_km(lat_deg, lon_deg, land_mask)
    vegetated_mask, desert_mask, snow_ice_mask = _surface_classification(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        land_mask=land_mask,
        elevation_km=elevation_km,
        temp_c=float(temp_c),
    )

    surface_class = np.full(land_mask.shape, SURFACE_CLASS_OCEAN, dtype=np.int8)
    surface_class = np.where(vegetated_mask, SURFACE_CLASS_VEGETATED, surface_class)
    surface_class = np.where(desert_mask, SURFACE_CLASS_DESERT, surface_class)
    surface_class = np.where(snow_ice_mask, SURFACE_CLASS_SNOW_ICE, surface_class)

    surface_texture = _surface_texture_field(
        lat_deg=lat_deg,
        elevation_km=elevation_km,
        vegetated_mask=vegetated_mask,
        desert_mask=desert_mask,
        snow_ice_mask=snow_ice_mask,
    )
    relief = _relief_field(
        elevation_km=elevation_km,
        land_mask=land_mask,
        snow_ice_mask=snow_ice_mask,
    )
    microclimate_c = _microclimate_field_c(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        elevation_km=elevation_km,
        desert_mask=desert_mask,
        snow_ice_mask=snow_ice_mask,
    )

    return {
        "land_mask": np.asarray(land_mask, dtype=bool),
        "vegetated_mask": np.asarray(vegetated_mask, dtype=bool),
        "temperate_mask": np.asarray(vegetated_mask, dtype=bool),
        "desert_mask": np.asarray(desert_mask, dtype=bool),
        "snow_ice_mask": np.asarray(snow_ice_mask, dtype=bool),
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
    lat_step = 180.0 / float(nlat_i)
    lon_step = 360.0 / float(nlon_i)
    lat_deg = np.linspace(-90.0 + 0.5 * lat_step, 90.0 - 0.5 * lat_step, nlat_i, dtype=float)
    lon_deg = np.linspace(-180.0 + 0.5 * lon_step, 180.0 - 0.5 * lon_step, nlon_i, dtype=float)

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


def build_earth_surface_mesh(
    *,
    temp_c: float,
    radius_km: float,
    nlat: int = 120,
    nlon: int = 240,
) -> EarthSurfaceMesh:
    """Build a closed Earth mesh with poles and seam included in the base geometry."""
    nlat_i = max(24, int(nlat))
    nlon_i = max(48, int(nlon))

    lat_deg = np.linspace(-90.0, 90.0, nlat_i, dtype=float)
    lon_deg = np.linspace(-180.0, 180.0, nlon_i + 1, dtype=float)
    lon2_rad, lat2_rad = np.meshgrid(np.deg2rad(lon_deg), np.deg2rad(lat_deg))
    fields = earth_surface_fields(lon2_rad, lat2_rad, temp_c=float(temp_c))

    radius_scale = float(np.clip(float(radius_km) / 6371.0, 0.35, 2.2))
    relief = np.asarray(fields["relief"], dtype=float).copy()
    # Collapse both poles to a single radius each to guarantee closed, pinhole-free poles.
    relief[0, :] = float(np.mean(relief[0, :]))
    relief[-1, :] = float(np.mean(relief[-1, :]))
    surface_texture = np.asarray(fields["surface_texture"], dtype=float).copy()
    surface_texture[0, :] = float(np.mean(surface_texture[0, :]))
    surface_texture[-1, :] = float(np.mean(surface_texture[-1, :]))
    radius_field = radius_scale * (1.0 + relief)
    lon2_deg, lat2_deg = np.meshgrid(lon_deg, lat_deg)
    x, y, z = latlon_to_xyz(lat2_deg, lon2_deg, radius=radius_field)

    return EarthSurfaceMesh(
        radius_scale=radius_scale,
        lat_deg=np.asarray(lat_deg, dtype=float),
        lon_deg=np.asarray(lon_deg, dtype=float),
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        z=np.asarray(z, dtype=float),
        surface_texture=np.asarray(surface_texture, dtype=float),
        relief=np.asarray(relief, dtype=float),
        elevation_km=np.asarray(fields["elevation_km"], dtype=float),
        land_mask=np.asarray(fields["land_mask"], dtype=bool),
        vegetated_mask=np.asarray(fields["vegetated_mask"], dtype=bool),
        desert_mask=np.asarray(fields["desert_mask"], dtype=bool),
        snow_ice_mask=np.asarray(fields["snow_ice_mask"], dtype=bool),
        surface_class=np.asarray(fields["surface_class"], dtype=np.int8),
        microclimate_c=np.asarray(fields["microclimate_c"], dtype=float),
    )


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


__all__ = [
    "CANONICAL_LAT_MAX_DEG",
    "CANONICAL_LAT_MIN_DEG",
    "CANONICAL_LON_MAX_DEG",
    "CANONICAL_LON_MIN_DEG",
    "EARTH_LAND_MASK_PATH",
    "EarthSurfaceMesh",
    "SURFACE_CLASS_DESERT",
    "SURFACE_CLASS_OCEAN",
    "SURFACE_CLASS_SNOW_ICE",
    "SURFACE_CLASS_TEMPERATE",
    "SURFACE_CLASS_VEGETATED",
    "build_earth_surface_mesh",
    "earth_geography_grid",
    "earth_land_mask_sampled",
    "earth_surface_fields",
    "latlon_to_xyz",
    "prepare_surface_render_grid",
    "surface_grid_to_xyz",
    "visual_alignment_report",
    "wrap_longitude_deg",
]
