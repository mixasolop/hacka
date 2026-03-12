import numpy as np

from htp.model.latitude import evaluate_local_human_habitability
from htp.model.earth import (
    SURFACE_CLASS_DESERT,
    SURFACE_CLASS_SNOW_ICE,
    SURFACE_CLASS_VEGETATED,
    earth_geography_grid,
)


def _earthlike_params():
    return {
        "warm_albedo": 0.30,
        "ice_albedo": 0.62,
        "enable_seasonality": True,
        "habitable_temp_min_c": 0.0,
        "habitable_temp_max_c": 45.0,
    }


def test_local_habitability_field_is_not_collapsed_uniform_red():
    geo = earth_geography_grid(texture_temp_c=16.5, nlat=72, nlon=144)
    land = np.asarray(geo["land_mask"], dtype=bool)
    result = evaluate_local_human_habitability(
        global_T_c=16.5,
        global_CO2_ppm=420.0,
        lat_deg=np.asarray(geo["lat_deg"], dtype=float),
        elevation_km=np.asarray(geo["elevation_km"], dtype=float),
        microclimate_c=np.asarray(geo["microclimate_c"], dtype=float),
        planet_params=_earthlike_params(),
        year=0,
    )
    score = np.asarray(result["score"], dtype=float)
    land_scores = score[land]

    assert np.all(np.isfinite(land_scores))
    assert float(np.min(land_scores)) >= 0.0
    assert float(np.max(land_scores)) <= 1.0
    # Ensure we retain regional structure instead of collapsing nearly all land to one low value.
    assert float(np.quantile(land_scores, 0.9) - np.quantile(land_scores, 0.1)) > 0.35
    assert 0.20 < float(np.mean(land_scores < 0.1)) < 0.80


def test_local_habitability_changes_coherently_with_global_warming():
    geo = earth_geography_grid(texture_temp_c=16.5, nlat=72, nlon=144)
    lat = np.asarray(geo["lat_deg"], dtype=float)
    elev = np.asarray(geo["elevation_km"], dtype=float)
    micro = np.asarray(geo["microclimate_c"], dtype=float)
    land = np.asarray(geo["land_mask"], dtype=bool)
    params = _earthlike_params()

    cool = evaluate_local_human_habitability(15.5, 420.0, lat, elev, micro, params, year=0)
    warm = evaluate_local_human_habitability(19.5, 520.0, lat, elev, micro, params, year=120)
    score_cool = np.asarray(cool["score"], dtype=float)
    score_warm = np.asarray(warm["score"], dtype=float)

    assert float(np.mean(score_warm[land])) > float(np.mean(score_cool[land]))
    assert float(np.mean(np.abs(score_warm - score_cool))) > 0.03


def _nearest_grid_indices(lat_axis: np.ndarray, lon_axis: np.ndarray, lat_deg: float, lon_deg: float) -> tuple[int, int]:
    lat_idx = int(np.argmin(np.abs(lat_axis - float(lat_deg))))
    lon_delta = (lon_axis - float(lon_deg) + 180.0) % 360.0 - 180.0
    lon_idx = int(np.argmin(np.abs(lon_delta)))
    return lat_idx, lon_idx


def test_surface_classes_drive_distinct_desert_vs_snow_temperature_behavior():
    geo = earth_geography_grid(texture_temp_c=15.0, nlat=72, nlon=144)
    lat = np.asarray(geo["lat_deg"], dtype=float)
    lon = np.asarray(geo["lon_deg"], dtype=float)
    land = np.asarray(geo["land_mask"], dtype=bool)
    desert_mask = np.asarray(geo["desert_mask"], dtype=bool)
    snow_mask = np.asarray(geo["snow_ice_mask"], dtype=bool)
    vegetated_mask = np.asarray(geo["vegetated_mask"], dtype=bool)
    surface_class = np.asarray(geo["surface_class"], dtype=int)
    elev = np.asarray(geo["elevation_km"], dtype=float)
    micro = np.asarray(geo["microclimate_c"], dtype=float)

    assert np.any(desert_mask)
    assert np.any(snow_mask)
    assert np.any(vegetated_mask)
    assert not np.any(desert_mask & snow_mask)

    result = evaluate_local_human_habitability(
        global_T_c=15.0,
        global_CO2_ppm=420.0,
        lat_deg=lat,
        elevation_km=elev,
        microclimate_c=micro,
        planet_params=_earthlike_params(),
        surface_class=surface_class,
        year=0,
    )
    temp = np.asarray(result["local_temp_c"], dtype=float)
    class_adjustment = np.asarray(result["surface_class_adjustment_c"], dtype=float)

    desert_temp = float(np.mean(temp[desert_mask & land]))
    snow_temp = float(np.mean(temp[snow_mask & land]))
    assert desert_temp > snow_temp + 8.0

    assert float(np.mean(class_adjustment[desert_mask & land])) > 0.0
    assert float(np.mean(class_adjustment[snow_mask & land])) < 0.0

    sahara_i, sahara_j = _nearest_grid_indices(lat, lon, lat_deg=23.0, lon_deg=13.0)
    greenland_i, greenland_j = _nearest_grid_indices(lat, lon, lat_deg=73.0, lon_deg=-40.0)
    europe_i, europe_j = _nearest_grid_indices(lat, lon, lat_deg=50.0, lon_deg=10.0)

    assert int(surface_class[sahara_i, sahara_j]) == SURFACE_CLASS_DESERT
    assert int(surface_class[greenland_i, greenland_j]) == SURFACE_CLASS_SNOW_ICE
    assert int(surface_class[europe_i, europe_j]) == SURFACE_CLASS_VEGETATED

    assert float(temp[sahara_i, sahara_j]) > float(temp[europe_i, europe_j])
    assert float(temp[europe_i, europe_j]) > float(temp[greenland_i, greenland_j])
