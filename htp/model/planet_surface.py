"""Shared assembly of planet surface geometry and texture fields for 3D rendering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .earth import earth_surface_fields, prepare_surface_render_grid, surface_grid_to_xyz


@dataclass(frozen=True)
class PlanetSurfaceMesh:
    radius_scale: float
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    texture: np.ndarray
    south_cap_x: np.ndarray
    south_cap_y: np.ndarray
    south_cap_z: np.ndarray
    south_cap_texture: np.ndarray


def build_planet_surface_mesh(
    *,
    temp_c: float,
    radius_km: float,
    nlat: int = 100,
    nlon: int = 100,
) -> PlanetSurfaceMesh:
    lat_count = max(16, int(nlat))
    lon_count = max(32, int(nlon))
    lat_deg = np.linspace(-89.0, 89.0, lat_count, dtype=float)
    lon_deg = np.linspace(-180.0, 180.0, lon_count, endpoint=False, dtype=float)
    lon2_deg, lat2_deg = np.meshgrid(lon_deg, lat_deg)
    lon = np.deg2rad(np.asarray(lon2_deg, dtype=float))
    lat = np.deg2rad(np.asarray(lat2_deg, dtype=float))
    radius_scale = float(np.clip(radius_km / 6371.0, 0.35, 2.2))

    surface_fields = earth_surface_fields(lon, lat, temp_c=temp_c)
    texture = np.asarray(surface_fields["surface_texture"], dtype=float)
    relief = np.asarray(surface_fields["relief"], dtype=float)

    # Close longitude seam and add true pole rows so the surface has no gaps.
    lat_closed_deg, lon_closed_deg, texture_closed, relief_closed = prepare_surface_render_grid(
        lat_deg,
        lon_deg,
        texture,
        relief,
        add_polar_caps=True,
        wrap_longitude=True,
    )
    texture_closed[0, :] = 0.985
    texture_closed[-1, :] = float(np.mean(texture[-1, :]))
    relief_closed[0, :] = max(float(np.mean(relief[0, :])), 0.010)
    relief_closed[-1, :] = float(np.mean(relief[-1, :]))

    xyz_base = surface_grid_to_xyz(
        lat_deg=lat_closed_deg,
        lon_deg=lon_closed_deg,
        radius=1.0,
    )
    x0 = np.asarray(xyz_base["x"], dtype=float)
    y0 = np.asarray(xyz_base["y"], dtype=float)
    z0 = np.asarray(xyz_base["z"], dtype=float)
    rfield = radius_scale * (1.0 + relief_closed)
    x = rfield * x0
    y = rfield * y0
    z = rfield * z0

    # Extra south-pole cap hides potential pinholes and keeps a visually solid snow circle.
    south_cap_lat = np.array([-90.0, -72.0], dtype=float)
    south_cap_tex = np.full((south_cap_lat.size, lon_deg.size), 0.985, dtype=float)
    south_cap_rel = np.full((south_cap_lat.size, lon_deg.size), max(float(np.mean(relief_closed[0, :])), 0.010), dtype=float)
    south_cap_lat_closed, south_cap_lon_closed, south_cap_tex_closed, south_cap_rel_closed = prepare_surface_render_grid(
        south_cap_lat,
        lon_deg,
        south_cap_tex,
        south_cap_rel,
        add_polar_caps=False,
        wrap_longitude=True,
    )
    south_cap_xyz = surface_grid_to_xyz(
        lat_deg=south_cap_lat_closed,
        lon_deg=south_cap_lon_closed,
        radius=1.0,
    )
    south_cap_r = radius_scale * (1.0 + south_cap_rel_closed + 0.008)
    south_cap_x = south_cap_r * np.asarray(south_cap_xyz["x"], dtype=float)
    south_cap_y = south_cap_r * np.asarray(south_cap_xyz["y"], dtype=float)
    south_cap_z = south_cap_r * np.asarray(south_cap_xyz["z"], dtype=float)

    return PlanetSurfaceMesh(
        radius_scale=radius_scale,
        x=x,
        y=y,
        z=z,
        texture=texture_closed,
        south_cap_x=south_cap_x,
        south_cap_y=south_cap_y,
        south_cap_z=south_cap_z,
        south_cap_texture=south_cap_tex_closed,
    )
