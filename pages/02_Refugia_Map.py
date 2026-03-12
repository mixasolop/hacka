import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from htp.model.constants import SIM_DT_YEARS_DEFAULT, SIM_YEARS_DEFAULT
from htp.model.latitude import evaluate_local_human_habitability
from htp.model.scenario_io import (
    import_scenario_json,
    load_scenario_from_session,
    scenario_signature,
    scenario_to_flat_params,
)
from htp.model.safety import safe_exp
from htp.model.simulate import simulate_time_series
from htp.model.earth import (
    SURFACE_CLASS_OCEAN,
    earth_geography_grid,
    latlon_to_xyz,
    prepare_surface_render_grid,
    surface_grid_to_xyz,
    visual_alignment_report,
)
from htp.ui.spinning_plot import render_spinning_surface

st.set_page_config(page_title="Refugia Map", layout="wide")

CLIMATE_TWIN_SERIES_CACHE_KEY = "climate_twin_series_cache"
CLIMATE_TWIN_YEAR_KEY = "climate_twin_year"
MAP_SPIN_SPEED_DEG_PER_SEC = 10.2
DEFAULT_LIVABLE_THRESHOLD = 0.45
MAP_SAMPLE_CHECK_COUNT = 8
MAP_BODY_RADIUS = 1.0
MAP_TEXTURE_RADIUS = 1.002
MAP_DOT_RADIUS = 1.006
MAP_DEBUG_DOT_RADIUS = 1.014
MAP_INITIAL_VIEW_LON_DEG = 38.0
MAP_TILT_DEG = 23.44
MAP_EYE_RADIUS = 2.05
MAP_EYE_Z = 1.08

HUMAN_HABITABILITY_COLORSCALE = [
    [0.00, "#8b0000"],
    [0.20, "#cc3434"],
    [0.40, "#e67e22"],
    [0.60, "#d9c447"],
    [0.80, "#62cc5f"],
    [1.00, "#00ff40"],
]
SURFACE_TEXTURE_COLORSCALE = [
    [0.00, "#091326"],
    [0.22, "#12305a"],
    [0.35, "#1f4d84"],
    [0.50, "#345a33"],
    [0.68, "#5c6d38"],
    [0.82, "#8b7b58"],
    [1.00, "#f0f3ff"],
]


def _load_climate_twin_series_cache(signature: str):
    cache = st.session_state.get(CLIMATE_TWIN_SERIES_CACHE_KEY)
    if not isinstance(cache, dict):
        return None

    if str(cache.get("scenario_signature", "")) != signature:
        return None

    try:
        years = np.asarray(cache.get("time_years", []), dtype=float)
        global_temp_c = np.asarray(cache.get("global_temperature_c", []), dtype=float)
        global_co2_ppm = np.asarray(cache.get("co2_ppm", []), dtype=float)
        habitable_surface_percent = np.asarray(cache.get("habitable_surface_percent", []), dtype=float)
    except Exception:
        return None

    n = int(len(years))
    if n == 0:
        return None
    if len(global_temp_c) != n or len(global_co2_ppm) != n or len(habitable_surface_percent) != n:
        return None
    return {
        "years": years,
        "global_temperature_c": global_temp_c,
        "co2_ppm": global_co2_ppm,
        "habitable_surface_percent": habitable_surface_percent,
    }


@st.cache_data(show_spinner=False)
def _simulate_global_series(signature: str, scenario_json: str):
    scenario = import_scenario_json(scenario_json)
    series = simulate_time_series(scenario, years=SIM_YEARS_DEFAULT, dt_years=SIM_DT_YEARS_DEFAULT)
    return {
        "years": np.asarray(series["time_years"], dtype=float).tolist(),
        "global_temperature_c": np.asarray(series["global_temperature_c"], dtype=float).tolist(),
        "co2_ppm": np.asarray(series["co2_ppm"], dtype=float).tolist(),
        "habitable_surface_percent": np.asarray(series["habitable_surface_percent"], dtype=float).tolist(),
    }


@st.cache_data(show_spinner=False)
def _generate_geography(texture_temp_c: float):
    geo = earth_geography_grid(texture_temp_c=float(texture_temp_c))
    return {
        "lat_deg": np.asarray(geo["lat_deg"], dtype=float).tolist(),
        "lon_deg": np.asarray(geo["lon_deg"], dtype=float).tolist(),
        "elevation_km": np.asarray(geo["elevation_km"], dtype=float).tolist(),
        "microclimate_c": np.asarray(geo["microclimate_c"], dtype=float).tolist(),
        "land_mask": np.asarray(geo["land_mask"], dtype=float).tolist(),
        "vegetated_mask": np.asarray(geo["vegetated_mask"], dtype=float).tolist(),
        "desert_mask": np.asarray(geo["desert_mask"], dtype=float).tolist(),
        "snow_ice_mask": np.asarray(geo["snow_ice_mask"], dtype=float).tolist(),
        "surface_class": np.asarray(geo["surface_class"], dtype=float).tolist(),
        "surface_texture": np.asarray(geo["surface_texture"], dtype=float).tolist(),
    }


def _tile_co2_ppm(global_co2_ppm: float, elevation_km: np.ndarray):
    exponent = -np.clip(elevation_km, -1.0, 8.0) / 8.5
    pressure_factor = np.vectorize(safe_exp)(exponent)
    basin_boost = 1.0 + 0.04 * np.clip(-elevation_km, 0.0, 4.0)
    return np.clip(global_co2_ppm * pressure_factor * basin_boost, 1.0, None)


def _rotate_xyz_about_y(x: np.ndarray, y: np.ndarray, z: np.ndarray, angle_deg: float):
    angle_rad = np.deg2rad(float(angle_deg))
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    x_rot = cos_a * x_arr + sin_a * z_arr
    y_rot = y_arr
    z_rot = -sin_a * x_arr + cos_a * z_arr
    return x_rot, y_rot, z_rot


def _sample_points_for_mask(
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
    land_mask: np.ndarray,
    visible_land_mask: np.ndarray,
    score: np.ndarray,
    target_is_land: bool,
    sample_count: int,
):
    target = np.asarray(land_mask, dtype=bool) if target_is_land else ~np.asarray(land_mask, dtype=bool)
    flat_idx = np.flatnonzero(target.ravel())
    if flat_idx.size == 0:
        return []
    rng = np.random.default_rng(20260311 + (1 if target_is_land else 2))
    take = flat_idx if flat_idx.size <= int(sample_count) else rng.choice(flat_idx, size=int(sample_count), replace=False)
    out = []
    for idx in np.asarray(take, dtype=int):
        out.append(
            {
                "lat_deg": float(lat2_deg.ravel()[idx]),
                "lon_deg": float(lon2_deg.ravel()[idx]),
                "mask_is_land": bool(land_mask.ravel()[idx]),
                "visible_is_land": bool(visible_land_mask.ravel()[idx]),
                "habitability_score": float(score.ravel()[idx]),
            }
        )
    return out


def _build_alignment_debug(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    land_mask: np.ndarray,
    surface_class: np.ndarray,
    score: np.ndarray,
):
    lon2_deg, lat2_deg = np.meshgrid(lon_deg, lat_deg)
    visible_land_mask = np.asarray(surface_class, dtype=int) != SURFACE_CLASS_OCEAN
    report = visual_alignment_report(
        land_mask=np.asarray(land_mask, dtype=bool),
        surface_class=np.asarray(surface_class, dtype=int),
    )
    north = latlon_to_xyz(
        lat_deg=90.0,
        lon_deg=0.0,
    )
    south = latlon_to_xyz(
        lat_deg=-90.0,
        lon_deg=0.0,
    )
    land_samples = _sample_points_for_mask(
        lat2_deg=lat2_deg,
        lon2_deg=lon2_deg,
        land_mask=land_mask,
        visible_land_mask=visible_land_mask,
        score=score,
        target_is_land=True,
        sample_count=MAP_SAMPLE_CHECK_COUNT,
    )
    ocean_samples = _sample_points_for_mask(
        lat2_deg=lat2_deg,
        lon2_deg=lon2_deg,
        land_mask=land_mask,
        visible_land_mask=visible_land_mask,
        score=score,
        target_is_land=False,
        sample_count=MAP_SAMPLE_CHECK_COUNT,
    )
    return {
        "report": report,
        "land_samples": land_samples,
        "ocean_samples": ocean_samples,
        "north_has_positive_z": bool(float(np.asarray(north[2])) > 0.0),
        "south_has_negative_z": bool(float(np.asarray(south[2])) < 0.0),
    }


def _habitability_map(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    score: np.ndarray,
    land_mask: np.ndarray,
    livable_mask: np.ndarray,
    surface_class: np.ndarray,
    surface_texture: np.ndarray,
    year: int,
    show_debug_overlay: bool,
    show_dots: bool,
    alignment_debug: dict,
):
    lat_surface_deg, lon_surface_deg, surface_texture_full = prepare_surface_render_grid(
        lat_deg,
        lon_deg,
        np.asarray(surface_texture, dtype=float),
        add_polar_caps=True,
        wrap_longitude=True,
    )
    xyz_body = surface_grid_to_xyz(
        lat_deg=lat_surface_deg,
        lon_deg=lon_surface_deg,
        radius=MAP_BODY_RADIUS,
    )
    xyz_texture = surface_grid_to_xyz(
        lat_deg=lat_surface_deg,
        lon_deg=lon_surface_deg,
        radius=MAP_TEXTURE_RADIUS,
    )
    body_x, body_y, body_z = _rotate_xyz_about_y(
        xyz_body["x"],
        xyz_body["y"],
        xyz_body["z"],
        MAP_TILT_DEG,
    )
    texture_x, texture_y, texture_z = _rotate_xyz_about_y(
        xyz_texture["x"],
        xyz_texture["y"],
        xyz_texture["z"],
        MAP_TILT_DEG,
    )
    lon2_deg, lat2_deg = np.meshgrid(lon_deg, lat_deg)
    land_flat = np.asarray(land_mask, dtype=bool).ravel()
    lat_land = lat2_deg.ravel()[land_flat]
    lon_land = lon2_deg.ravel()[land_flat]
    score_land = np.asarray(score, dtype=float).ravel()[land_flat]
    livable_land = np.asarray(livable_mask, dtype=bool).ravel()[land_flat]
    x_land, y_land, z_land = latlon_to_xyz(
        lat_deg=lat_land,
        lon_deg=lon_land,
        radius=MAP_DOT_RADIUS,
    )
    x_land, y_land, z_land = _rotate_xyz_about_y(x_land, y_land, z_land, MAP_TILT_DEG)
    custom = np.column_stack((score_land, livable_land.astype(int), lat_land, lon_land))

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=body_x,
            y=body_y,
            z=body_z,
            surfacecolor=np.zeros_like(np.asarray(body_x, dtype=float), dtype=float),
            colorscale=[[0.0, "#0b1830"], [1.0, "#0b1830"]],
            cmin=0.0,
            cmax=1.0,
            showscale=False,
            opacity=1.0,
            hoverinfo="skip",
            lighting=dict(ambient=0.95, diffuse=0.25, specular=0.04, roughness=1.0),
            lightposition=dict(x=130, y=85, z=210),
        )
    )
    fig.add_trace(
        go.Surface(
            x=texture_x,
            y=texture_y,
            z=texture_z,
            surfacecolor=np.asarray(surface_texture_full, dtype=float),
            colorscale=SURFACE_TEXTURE_COLORSCALE,
            cmin=0.0,
            cmax=1.0,
            showscale=False,
            opacity=1.0,
            hoverinfo="skip",
            lighting=dict(ambient=0.82, diffuse=0.58, specular=0.10, roughness=0.92),
            lightposition=dict(x=130, y=85, z=210),
        )
    )
    if show_dots:
        fig.add_trace(
            go.Scatter3d(
                x=x_land,
                y=y_land,
                z=z_land,
                mode="markers",
                marker=dict(
                    size=2.8,
                    color=score_land,
                    colorscale=HUMAN_HABITABILITY_COLORSCALE,
                    cmin=0.0,
                    cmax=1.0,
                    opacity=1.0,
                    line=dict(width=0),
                    colorbar=dict(
                        title="Human Habitability",
                        x=1.0,
                        tickmode="array",
                        tickvals=[0.0, 0.25, 0.50, 0.75, 1.0],
                        ticktext=["0.00", "0.25", "0.50", "0.75", "1.00"],
                    ),
                ),
                customdata=custom,
                hovertemplate=(
                    "Lat %{customdata[2]:.1f}<br>Lon %{customdata[3]:.1f}"
                    "<br>Surface Land"
                    "<br>Habitability %{customdata[0]:.2f}"
                    "<br>Livable %{customdata[1]:.0f}"
                    "<extra></extra>"
                ),
            ),
        )

    if show_debug_overlay:
        visible_land_mask = np.asarray(surface_class, dtype=int).ravel() != SURFACE_CLASS_OCEAN
        x_all, y_all, z_all = latlon_to_xyz(
            lat_deg=lat2_deg.ravel(),
            lon_deg=lon2_deg.ravel(),
            radius=MAP_DEBUG_DOT_RADIUS,
        )
        x_all, y_all, z_all = _rotate_xyz_about_y(x_all, y_all, z_all, MAP_TILT_DEG)
        fig.add_trace(
            go.Scatter3d(
                x=x_all,
                y=y_all,
                z=z_all,
                mode="markers",
                marker=dict(
                    size=2.1,
                    color=np.where(visible_land_mask, "#72d34d", "#3b82f6"),
                    opacity=0.35,
                    line=dict(width=0),
                ),
                name="Debug: Texture Land/Ocean",
                hoverinfo="skip",
                showlegend=True,
            )
        )
        sample_points = [*alignment_debug["land_samples"], *alignment_debug["ocean_samples"]]
        if sample_points:
            sample_lat = np.asarray([point["lat_deg"] for point in sample_points], dtype=float)
            sample_lon = np.asarray([point["lon_deg"] for point in sample_points], dtype=float)
            sample_is_land = np.asarray([point["mask_is_land"] for point in sample_points], dtype=bool)
            xs, ys, zs = latlon_to_xyz(
                lat_deg=sample_lat,
                lon_deg=sample_lon,
                radius=MAP_DEBUG_DOT_RADIUS + 0.018,
            )
            xs, ys, zs = _rotate_xyz_about_y(xs, ys, zs, MAP_TILT_DEG)
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers",
                    marker=dict(
                        size=5.4,
                        color=np.where(sample_is_land, "#ffd54f", "#7dd3fc"),
                        opacity=1.0,
                        line=dict(color="#0f1626", width=1),
                    ),
                    name="Debug: Sample Checks",
                    customdata=np.column_stack((sample_lat, sample_lon, sample_is_land.astype(int))),
                    hovertemplate=(
                        "Sample Lat %{customdata[0]:.1f}<br>"
                        "Sample Lon %{customdata[1]:.1f}<br>"
                        "Mask class %{customdata[2]:.0f}"
                        "<extra></extra>"
                    ),
                    showlegend=True,
                )
            )

    view_phase_rad = np.deg2rad(float(MAP_INITIAL_VIEW_LON_DEG))
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=590,
        title=f"Human Livability Map - Year {year}",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d9e3f5"),
        clickmode="event+select",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            bgcolor="rgb(10, 14, 24)",
            camera=dict(
                eye=dict(
                    x=MAP_EYE_RADIUS * float(np.cos(view_phase_rad)),
                    y=MAP_EYE_RADIUS * float(np.sin(view_phase_rad)),
                    z=MAP_EYE_Z,
                ),
                up=dict(x=0.0, y=0.0, z=1.0),
            ),
        ),
        showlegend=bool(show_debug_overlay),
        legend=dict(
            bgcolor="rgba(15,22,38,0.70)",
            bordercolor="rgba(120,140,180,0.25)",
            borderwidth=1,
        ),
    )
    return fig


def render_map_page():
    st.markdown("<style>[data-testid='stHeaderActionElements']{display:none;}</style>", unsafe_allow_html=True)
    st.title("Refugia Map")
    st.caption("Uses Climate Twin outputs for global climate values and maps local livability on top.")

    scenario = load_scenario_from_session(st.session_state)
    params = scenario_to_flat_params(scenario)
    signature = scenario_signature(scenario)
    scenario_json = json.dumps(scenario.model_dump(mode="json"), sort_keys=True)
    climate_cache = _load_climate_twin_series_cache(signature)
    if climate_cache is None:
        climate_cache = _simulate_global_series(signature, scenario_json)
        st.session_state[CLIMATE_TWIN_SERIES_CACHE_KEY] = {
            "scenario_signature": signature,
            "time_years": list(climate_cache["years"]),
            "global_temperature_c": list(climate_cache["global_temperature_c"]),
            "co2_ppm": list(climate_cache["co2_ppm"]),
            "habitable_surface_percent": list(climate_cache["habitable_surface_percent"]),
            "meta": {"seed": int(params.get("seed", 42))},
        }

    years = np.asarray(climate_cache["years"], dtype=float)
    global_temp_c = np.asarray(climate_cache["global_temperature_c"], dtype=float)
    global_co2_ppm = np.asarray(climate_cache["co2_ppm"], dtype=float)
    global_habitable_pct = np.asarray(climate_cache["habitable_surface_percent"], dtype=float)

    geo = _generate_geography(texture_temp_c=float(global_temp_c[0]))
    lat_deg = np.array(geo["lat_deg"], dtype=float)
    lon_deg = np.array(geo["lon_deg"], dtype=float)
    elevation_km = np.array(geo["elevation_km"], dtype=float)
    microclimate_c = np.array(geo["microclimate_c"], dtype=float)
    land_mask = np.array(geo["land_mask"], dtype=float) >= 0.5
    vegetated_mask = np.array(geo["vegetated_mask"], dtype=float) >= 0.5
    desert_mask = np.array(geo["desert_mask"], dtype=float) >= 0.5
    snow_ice_mask = np.array(geo["snow_ice_mask"], dtype=float) >= 0.5
    surface_class = np.array(geo["surface_class"], dtype=float).astype(int)
    surface_texture = np.array(geo["surface_texture"], dtype=float)

    controls = st.columns([2.0, 1.6, 1.3, 1.3, 1.2])
    default_year = int(
        np.clip(
            st.session_state.get(CLIMATE_TWIN_YEAR_KEY, min(100, int(round(years[-1])))),
            int(round(years[0])),
            int(round(years[-1])),
        )
    )
    year = controls[0].slider(
        "Year",
        min_value=int(round(years[0])),
        max_value=int(round(years[-1])),
        value=default_year,
        step=1,
    )
    st.session_state[CLIMATE_TWIN_YEAR_KEY] = int(year)
    livable_threshold = controls[1].slider(
        "Livable Threshold",
        min_value=0.30,
        max_value=0.70,
        value=DEFAULT_LIVABLE_THRESHOLD,
        step=0.01,
        help="Higher threshold marks more regions as inhabitable (red).",
    )
    show_livable_only = controls[2].toggle("Show Livable Only", value=False)
    show_debug_overlay = controls[3].toggle(
        "Debug Alignment",
        value=False,
        help="Shows texture-class dots and sampled land/ocean checks on top of the globe.",
    )
    show_dots = controls[4].toggle(
        "Show Dots",
        value=True,
        help="Toggle livability dots. The globe body remains fully rendered.",
    )

    idx = int(np.argmin(np.abs(years - year)))
    climate_twin_habitable_pct = float(global_habitable_pct[idx])
    climate_twin_habitability_score = climate_twin_habitable_pct / 100.0

    local_eval = evaluate_local_human_habitability(
        global_T_c=float(global_temp_c[idx]),
        global_CO2_ppm=float(global_co2_ppm[idx]),
        lat_deg=lat_deg,
        elevation_km=elevation_km,
        microclimate_c=microclimate_c,
        surface_class=surface_class,
        planet_params=params,
        year=int(round(years[idx])),
    )
    temp_grid_c = np.asarray(local_eval["local_temp_c"], dtype=float)
    tile_co2 = _tile_co2_ppm(float(global_co2_ppm[idx]), elevation_km)
    score = np.asarray(local_eval["score"], dtype=float)
    temp_score = np.asarray(local_eval["temp_score"], dtype=float)
    co2_score = np.asarray(local_eval["co2_score"], dtype=float)
    temp_ok = np.asarray(local_eval["temp_ok"], dtype=bool)
    score = np.where(land_mask, score, 0.0)
    livable_mask = (score >= livable_threshold) & land_mask
    plot_score = np.where(livable_mask, score, 0.0) if show_livable_only else score

    alignment_debug = _build_alignment_debug(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        land_mask=land_mask,
        surface_class=surface_class,
        score=plot_score,
    )
    fig = _habitability_map(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        score=plot_score,
        land_mask=land_mask,
        livable_mask=livable_mask,
        surface_class=surface_class,
        surface_texture=surface_texture,
        year=int(round(years[idx])),
        show_debug_overlay=show_debug_overlay,
        show_dots=show_dots,
        alignment_debug=alignment_debug,
    )
    render_spinning_surface(
        fig,
        component_key="refugia_map_globe",
        height_px=590,
        speed_deg_per_sec=MAP_SPIN_SPEED_DEG_PER_SEC,
    )

    alignment_report = alignment_debug["report"]
    land_samples = alignment_debug["land_samples"]
    ocean_samples = alignment_debug["ocean_samples"]
    land_sample_ok = sum(int(item["mask_is_land"] == item["visible_is_land"]) for item in land_samples)
    ocean_sample_ok = sum(int(item["mask_is_land"] == item["visible_is_land"]) for item in ocean_samples)

    with st.container(border=True):
        st.subheader("Map Statistics")
        perfect_mask = (score >= 0.80) & land_mask
        hard_unlivable_mask = (~temp_ok) & land_mask
        desert_land_mask = desert_mask & land_mask
        snow_land_mask = snow_ice_mask & land_mask
        land_fraction = 100.0 * float(np.mean(land_mask))
        livable_on_land = 100.0 * float(np.sum(livable_mask) / max(1.0, np.sum(land_mask)))
        hard_unlivable_on_land = 100.0 * float(np.sum(hard_unlivable_mask) / max(1.0, np.sum(land_mask)))
        map_livable_global_pct = float(climate_twin_habitable_pct)
        threshold_livable_global_pct = 100.0 * float(np.mean(livable_mask))
        stats_lines = [
            "- Geography mode: Shared Earth land/ocean surface (Natural Earth source).",
            "- Canonical geometry: lat [-90,+90], lon [-180,+180), no longitude flipping.",
            f"- Land area: {land_fraction:.1f}% (water is forced unlivable)",
            f"- Global temperature: {float(global_temp_c[idx]):.2f} C",
            f"- Global CO2: {float(global_co2_ppm[idx]):.1f} ppm",
            f"- Mean habitability score (from Climate Twin page): {climate_twin_habitability_score:.3f}",
            f"- Non-red area (from Climate Twin page): {map_livable_global_pct:.1f}%",
            f"- Map livable area (score >= {livable_threshold:.2f}): {threshold_livable_global_pct:.1f}%",
            f"- Livable fraction of land: {livable_on_land:.1f}%",
            f"- Perfect area (score >= 0.80): {100.0 * float(np.mean(perfect_mask)):.1f}%",
            (
                f"- Hard red area on land (temp outside {params['habitable_temp_min_c']:.1f} to "
                f"{params['habitable_temp_max_c']:.1f} C): "
                f"{hard_unlivable_on_land:.1f}%"
            ),
            f"- Tile temperature range: {float(np.min(temp_grid_c)):.2f} to {float(np.max(temp_grid_c)):.2f} C",
            f"- Tile CO2 range: {float(np.min(tile_co2)):.0f} to {float(np.max(tile_co2)):.0f} ppm",
            f"- Mean temperature suitability: {float(np.mean(temp_score)):.3f}",
            f"- Mean CO2 suitability: {float(np.mean(co2_score)):.3f}",
            f"- Global habitability stress multiplier: {float(local_eval['stress']):.3f}",
            f"- Local soft margin used: {float(local_eval['soft_margin_c']):.2f} C",
            f"- Surface classes (land): vegetated {100.0 * float(np.mean(vegetated_mask)):.1f}%, desert {100.0 * float(np.mean(desert_mask)):.1f}%, snow/ice {100.0 * float(np.mean(snow_ice_mask)):.1f}%",
            (
                "- Class temperature means (C): "
                f"desert {float(np.mean(temp_grid_c[desert_land_mask])) if np.any(desert_land_mask) else float('nan'):.2f}, "
                f"snow/ice {float(np.mean(temp_grid_c[snow_land_mask])) if np.any(snow_land_mask) else float('nan'):.2f}"
            ),
            (
                "- Alignment checks: "
                f"mismatch={alignment_report['mismatch_cells']}/{alignment_report['cells']} "
                f"({100.0 * float(alignment_report['mismatch_fraction']):.2f}%), "
                f"best_lon_phase_shift={int(alignment_report['best_lon_phase_shift_cells'])} cells "
                f"({float(alignment_report['best_lon_phase_shift_deg']):.1f} deg), "
                f"mirrored={bool(alignment_report['is_lon_mirrored'])}"
            ),
            (
                "- Sample checks: "
                f"land {land_sample_ok}/{len(land_samples)}, "
                f"ocean {ocean_sample_ok}/{len(ocean_samples)}, "
                f"north_z_ok={alignment_debug['north_has_positive_z']}, "
                f"south_z_ok={alignment_debug['south_has_negative_z']}"
            ),
            f"- Dots layer visible: {bool(show_dots)}",
        ]
        st.markdown("\n".join(stats_lines))

    if show_debug_overlay:
        with st.expander("Alignment Debug Details", expanded=True):
            st.markdown(
                "\n".join(
                    [
                        f"- Direct match: {100.0 * float(alignment_report['direct_match_fraction']):.2f}%",
                        f"- Mirror match: {100.0 * float(alignment_report['mirror_match_fraction']):.2f}%",
                        f"- Mismatch cells: {int(alignment_report['mismatch_cells'])} / {int(alignment_report['cells'])}",
                    ]
                )
            )
            sample_rows = []
            for item in land_samples:
                sample_rows.append(
                    {
                        "sample_type": "land",
                        "lat_deg": item["lat_deg"],
                        "lon_deg": item["lon_deg"],
                        "mask_is_land": item["mask_is_land"],
                        "visible_is_land": item["visible_is_land"],
                        "habitability_score": item["habitability_score"],
                    }
                )
            for item in ocean_samples:
                sample_rows.append(
                    {
                        "sample_type": "ocean",
                        "lat_deg": item["lat_deg"],
                        "lon_deg": item["lon_deg"],
                        "mask_is_land": item["mask_is_land"],
                        "visible_is_land": item["visible_is_land"],
                        "habitability_score": item["habitability_score"],
                    }
                )
            if sample_rows:
                st.dataframe(sample_rows, use_container_width=True)


render_map_page()
