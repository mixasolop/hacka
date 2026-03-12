import numpy as np

from htp.model.earth import (
    SURFACE_CLASS_DESERT,
    SURFACE_CLASS_OCEAN,
    SURFACE_CLASS_SNOW_ICE,
    SURFACE_CLASS_TEMPERATE,
    earth_geography_grid,
    earth_surface_fields,
    earth_land_mask_sampled,
    latlon_to_xyz,
    prepare_surface_render_grid,
    surface_grid_to_xyz,
    visual_alignment_report,
)


def test_latlon_to_xyz_cardinal_orientation_default():
    x0, y0, z0 = latlon_to_xyz(lat_deg=0.0, lon_deg=0.0)
    xn, yn, zn = latlon_to_xyz(lat_deg=90.0, lon_deg=0.0)
    xs, ys, zs = latlon_to_xyz(lat_deg=-90.0, lon_deg=0.0)
    _, ye, _ = latlon_to_xyz(lat_deg=0.0, lon_deg=20.0)
    _, yw, _ = latlon_to_xyz(lat_deg=0.0, lon_deg=-20.0)

    assert np.isclose(float(x0), 1.0, atol=1e-12)
    assert np.isclose(float(y0), 0.0, atol=1e-12)
    assert np.isclose(float(z0), 0.0, atol=1e-12)
    assert float(zn) > 0.0 and float(zs) < 0.0
    assert float(ye) > float(yw)
    assert np.isclose(float(xn), 0.0, atol=1e-12)
    assert np.isclose(float(xs), 0.0, atol=1e-12)
    assert np.isclose(float(yn), 0.0, atol=1e-12)
    assert np.isclose(float(ys), 0.0, atol=1e-12)


def test_surface_grid_xyz_matches_pointwise_transform():
    lat_deg = np.linspace(-80.0, 80.0, 9, dtype=float)
    lon_deg = np.linspace(-180.0, 180.0, 12, endpoint=False, dtype=float)
    grid = surface_grid_to_xyz(lat_deg=lat_deg, lon_deg=lon_deg, radius=1.0)
    x2, y2, z2 = latlon_to_xyz(
        lat_deg=np.asarray(grid["lat2_deg"], dtype=float),
        lon_deg=np.asarray(grid["lon2_deg"], dtype=float),
        radius=1.0,
    )

    assert np.allclose(np.asarray(grid["x"], dtype=float), x2)
    assert np.allclose(np.asarray(grid["y"], dtype=float), y2)
    assert np.allclose(np.asarray(grid["z"], dtype=float), z2)


def test_prepare_surface_render_grid_adds_poles_and_lon_wrap():
    lat = np.array([-45.0, 0.0, 45.0], dtype=float)
    lon = np.array([-180.0, -90.0, 0.0, 90.0], dtype=float)
    field = np.arange(lat.size * lon.size, dtype=float).reshape(lat.size, lon.size)
    lat_out, lon_out, field_out = prepare_surface_render_grid(lat, lon, field)

    assert np.isclose(float(lat_out[0]), -90.0)
    assert np.isclose(float(lat_out[-1]), 90.0)
    assert field_out.shape == (field.shape[0] + 2, field.shape[1] + 1)
    assert np.allclose(field_out[0, :-1], field[0, :])
    assert np.allclose(field_out[-1, :-1], field[-1, :])
    assert np.allclose(field_out[:, -1], field_out[:, 0])
    assert np.isclose(float(lon_out[-1] - lon_out[0]), 360.0)


def test_earth_surface_classes_match_land_mask_without_lon_gap():
    geo = earth_geography_grid(texture_temp_c=15.0, nlat=72, nlon=144)
    report = visual_alignment_report(
        land_mask=np.asarray(geo["land_mask"], dtype=bool),
        surface_class=np.asarray(geo["surface_class"], dtype=int),
    )
    assert int(report["mismatch_cells"]) == 0
    assert int(report["best_lon_phase_shift_cells"]) == 0
    assert bool(report["is_lon_mirrored"]) is False


def test_earth_surface_fields_consistent_with_geography_grid():
    geo = earth_geography_grid(texture_temp_c=22.0, nlat=60, nlon=120)
    lat = np.asarray(geo["lat_deg"], dtype=float)
    lon = np.asarray(geo["lon_deg"], dtype=float)
    lon2, lat2 = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    fields = earth_surface_fields(lon2, lat2, temp_c=22.0)

    assert np.array_equal(
        np.asarray(geo["land_mask"], dtype=bool),
        np.asarray(fields["land_mask"], dtype=bool),
    )
    assert np.array_equal(
        np.asarray(geo["surface_class"], dtype=int),
        np.asarray(fields["surface_class"], dtype=int),
    )


def test_earth_mask_has_plausible_continent_and_ocean_points():
    assert bool(earth_land_mask_sampled(50.0, 10.0)) is True       # Europe
    assert bool(earth_land_mask_sampled(23.0, 13.0)) is True       # North Africa
    assert bool(earth_land_mask_sampled(-25.0, 133.0)) is True     # Australia
    assert bool(earth_land_mask_sampled(0.0, -140.0)) is False     # Pacific
    assert bool(earth_land_mask_sampled(0.0, -30.0)) is False      # Atlantic


def test_polar_snow_does_not_spill_into_central_europe():
    lat = np.array([50.0, 73.0, 23.0], dtype=float)
    lon = np.array([10.0, -40.0, 13.0], dtype=float)
    lon2, lat2 = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    fields = earth_surface_fields(lon2, lat2, temp_c=15.0)
    classes = np.asarray(fields["surface_class"], dtype=int)

    europe = int(classes[0, 0])
    greenland = int(classes[1, 1])
    sahara = int(classes[2, 2])

    assert europe == SURFACE_CLASS_TEMPERATE
    assert greenland == SURFACE_CLASS_SNOW_ICE
    assert sahara == SURFACE_CLASS_DESERT
    assert europe != SURFACE_CLASS_OCEAN
