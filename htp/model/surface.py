"""Legacy import compatibility for Earth surface mesh helpers."""

from .canonical_earth import EarthSurfaceMesh, build_earth_surface_mesh
from .planet_surface import PlanetSurfaceMesh, build_planet_surface_mesh

__all__ = [
    "EarthSurfaceMesh",
    "PlanetSurfaceMesh",
    "build_earth_surface_mesh",
    "build_planet_surface_mesh",
]
