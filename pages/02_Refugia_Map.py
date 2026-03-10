import hashlib
import json
from math import log

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Refugia Map", layout="wide")

SIM_YEARS = 500
SIM_DT_YEARS = 1.0
TEMPERATURE_RESPONSE_YEARS = 12.0

CO2_BASELINE_PPM = 280.0
REFERENCE_TEMP_C = 14.8
ALBEDO_REF = 0.30
DEFAULT_SEASONALITY_AMPLITUDE = 0.12
DEFAULT_ICE_TRANSITION_TEMP_C = -10.0
ICE_TRANSITION_WIDTH_C = 5.0
NATURAL_OUTGASSING = 0.9
DEFAULT_WEATHERING_STRENGTH = 0.55
DEFAULT_BIOSPHERE_UPTAKE_STRENGTH = 0.50
BIOSPHERE_OPT_TEMP_C = 15.0
BIOSPHERE_TEMP_WIDTH_C = 15.0

K_FLUX = 55.0
K_ALBEDO = 85.0
K_CO2 = 5.5
WEATHERING_COEFF = 0.94 * DEFAULT_WEATHERING_STRENGTH * 0.50
WEATHERING_TEMP_SENS = 0.045
BIOSPHERE_COEFF = 0.94 * DEFAULT_BIOSPHERE_UPTAKE_STRENGTH * 0.50
EMISSIONS_TO_PPM_PER_YEAR = 1.0
CO2_HUMAN_MAX_SAFE_PPM = 1000.0
CLIMATE_TWIN_SERIES_CACHE_KEY = "climate_twin_series_cache"
CLIMATE_TWIN_YEAR_KEY = "climate_twin_year"

SCENARIO_KEYS = (
    "stellar_flux_multiplier",
    "enable_seasonality",
    "warm_albedo",
    "ice_albedo",
    "initial_co2_ppm",
    "emissions_rate",
    "emissions_growth_mode",
    "mitigation_start_year",
    "mitigation_strength",
    "habitability_profile",
    "habitable_temp_min_c",
    "habitable_temp_max_c",
)

DEFAULT_SCENARIO = {
    "stellar_flux_multiplier": 1.00,
    "enable_seasonality": True,
    "warm_albedo": 0.30,
    "ice_albedo": 0.62,
    "initial_co2_ppm": 420.0,
    "emissions_rate": 1.0,
    "emissions_growth_mode": "Constant",
    "mitigation_start_year": 40,
    "mitigation_strength": 0.15,
    "habitability_profile": "Liquid Water",
    "habitable_temp_min_c": 0.0,
    "habitable_temp_max_c": 45.0,
}

EMISSIONS_MODES = ("Constant", "Growing", "Carefree", "Stabilization", "Aggressive Mitigation")

HUMAN_HABITABILITY_COLORSCALE = [
    [0.00, "#8b0000"],
    [0.20, "#cc3434"],
    [0.40, "#e67e22"],
    [0.60, "#d9c447"],
    [0.80, "#62cc5f"],
    [1.00, "#00ff40"],
]
PLANET_VIEW_LAT = 26
PLANET_VIEW_LON = -39


def _load_current_scenario():
    params = dict(DEFAULT_SCENARIO)

    scenario_source = None
    payload = st.session_state.get("params")
    if isinstance(payload, dict):
        payload_scenario = payload.get("scenario")
        if isinstance(payload_scenario, dict):
            scenario_source = payload_scenario

    if scenario_source is None:
        submitted_snapshot = st.session_state.get("submitted_scenario_snapshot")
        if isinstance(submitted_snapshot, dict):
            scenario_source = submitted_snapshot

    if isinstance(scenario_source, dict):
        for key in SCENARIO_KEYS:
            if key in scenario_source:
                params[key] = scenario_source[key]

    params["stellar_flux_multiplier"] = float(params["stellar_flux_multiplier"])
    params["enable_seasonality"] = bool(params["enable_seasonality"])
    params["warm_albedo"] = float(params["warm_albedo"])
    params["ice_albedo"] = float(params["ice_albedo"])
    params["initial_co2_ppm"] = max(1.0, float(params["initial_co2_ppm"]))
    params["emissions_rate"] = max(0.0, float(params["emissions_rate"]))
    params["emissions_growth_mode"] = (
        str(params["emissions_growth_mode"]) if str(params["emissions_growth_mode"]) in EMISSIONS_MODES else "Constant"
    )
    params["mitigation_start_year"] = max(0, int(params["mitigation_start_year"]))
    params["mitigation_strength"] = float(np.clip(params["mitigation_strength"], 0.0, 1.0))
    params["habitable_temp_min_c"] = float(params["habitable_temp_min_c"])
    params["habitable_temp_max_c"] = float(params["habitable_temp_max_c"])
    if params["habitable_temp_max_c"] <= params["habitable_temp_min_c"]:
        params["habitable_temp_max_c"] = params["habitable_temp_min_c"] + 1.0
    return params


def _scenario_signature(params):
    payload = {key: params[key] for key in SCENARIO_KEYS if key in params}
    return json.dumps(payload, sort_keys=True)


def _load_climate_twin_series_cache(params):
    cache = st.session_state.get(CLIMATE_TWIN_SERIES_CACHE_KEY)
    if not isinstance(cache, dict):
        return None

    if str(cache.get("scenario_signature", "")) != _scenario_signature(params):
        return None

    try:
        years = np.asarray(cache.get("time_years", []), dtype=int)
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


def _soft_step(x):
    safe_x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-safe_x))


def _effective_ice_weight(temp_c: float):
    exponent = np.clip((temp_c - DEFAULT_ICE_TRANSITION_TEMP_C) / ICE_TRANSITION_WIDTH_C, -40.0, 40.0)
    return float(1.0 / (1.0 + np.exp(exponent)))


def _effective_albedo(temp_c: float, warm_albedo: float, ice_albedo: float):
    w_ice = _effective_ice_weight(temp_c)
    alpha = warm_albedo * (1.0 - w_ice) + ice_albedo * w_ice
    return float(np.clip(alpha, 0.0, 1.0))


def _equilibrium_temperature_c(stellar_flux_multiplier: float, warm_albedo: float, ice_albedo: float, co2_ppm: float):
    temp_c = REFERENCE_TEMP_C
    safe_co2 = max(float(co2_ppm), 1.0)
    albedo_contrast = np.clip((ice_albedo - warm_albedo - 0.18) / 0.45, 0.0, 1.0)
    low_flux = np.clip((1.0 - stellar_flux_multiplier) / 0.22, 0.0, 1.0)
    for _ in range(3):
        alpha_eff = _effective_albedo(temp_c, warm_albedo, ice_albedo)
        cold_index = _soft_step((6.0 - temp_c) / 4.0)
        cold_locking_feedback = 6.5 * cold_index * (0.55 * albedo_contrast + 0.45 * low_flux)
        temp_c = (
            REFERENCE_TEMP_C
            + K_FLUX * (stellar_flux_multiplier - 1.0)
            - K_ALBEDO * (alpha_eff - ALBEDO_REF)
            + K_CO2 * log(safe_co2 / CO2_BASELINE_PPM)
            - cold_locking_feedback
        )
    return float(temp_c)


def _human_emissions_rate(mode: str, emissions_rate: float, year: int, mitigation_start_year: int, mitigation_strength: float):
    if mode == "Growing":
        base = emissions_rate * (1.0 + 0.010 * year)
    elif mode == "Carefree":
        base = emissions_rate * (1.0 + 0.018 * year) / (1.0 + 0.0012 * year)
    elif mode == "Stabilization":
        pre_growth = emissions_rate * (1.0 + 0.004 * min(year, mitigation_start_year))
        if year < mitigation_start_year:
            base = pre_growth
        else:
            decay = 0.035 * max(0.1, mitigation_strength) * (year - mitigation_start_year)
            base = pre_growth * np.exp(-decay)
    elif mode == "Aggressive Mitigation":
        if year < mitigation_start_year:
            base = emissions_rate
        else:
            decay = 0.065 * max(0.2, mitigation_strength) * (year - mitigation_start_year)
            base = emissions_rate * np.exp(-decay)
    else:
        base = emissions_rate

    if year >= mitigation_start_year and mode in {"Constant", "Growing", "Carefree"}:
        base *= max(0.05, 1.0 - 0.85 * mitigation_strength)
    return float(EMISSIONS_TO_PPM_PER_YEAR * max(0.0, base))


def _weathering_sink(temp_c: float, co2_ppm: float):
    temp_factor = np.exp(np.clip(WEATHERING_TEMP_SENS * (temp_c - REFERENCE_TEMP_C), -5.0, 5.0))
    return float(WEATHERING_COEFF * (co2_ppm / CO2_BASELINE_PPM) * temp_factor)


def _biosphere_sink(temp_c: float, co2_ppm: float):
    gaussian = -((temp_c - BIOSPHERE_OPT_TEMP_C) ** 2) / (2.0 * BIOSPHERE_TEMP_WIDTH_C**2)
    temp_factor = np.exp(np.clip(gaussian, -20.0, 0.0))
    return float(BIOSPHERE_COEFF * (co2_ppm / CO2_BASELINE_PPM) * temp_factor)


def _cold_trap_cooling(temp_c: float, co2_ppm: float, params):
    if temp_c >= 9.0:
        return 0.0
    cold_severity = np.clip((9.0 - temp_c) / 30.0, 0.0, 1.0)
    albedo_contrast = np.clip((params["ice_albedo"] - params["warm_albedo"] - 0.18) / 0.45, 0.0, 1.0)
    low_flux = np.clip((1.0 - params["stellar_flux_multiplier"]) / 0.22, 0.0, 1.0)
    if albedo_contrast <= 0.0 and low_flux <= 0.0:
        return 0.0
    co2_escape = np.clip((co2_ppm - CO2_BASELINE_PPM) / 1500.0, 0.0, 1.0)
    persistence = 1.0 - 0.40 * co2_escape
    return float(0.40 * cold_severity * (0.65 * albedo_contrast + 0.35 * low_flux) * persistence)


@st.cache_data(show_spinner=False)
def _simulate_global_series(params_payload: str):
    params = json.loads(params_payload)
    years = np.arange(0, SIM_YEARS + 1, dtype=int)
    n = len(years)
    temp_c_series = np.zeros(n, dtype=float)
    co2_series = np.zeros(n, dtype=float)

    co2_ppm = max(float(params["initial_co2_ppm"]), 1.0)
    temp_c = _equilibrium_temperature_c(
        params["stellar_flux_multiplier"], params["warm_albedo"], params["ice_albedo"], co2_ppm
    )

    for i, year in enumerate(years):
        temp_eq_c = _equilibrium_temperature_c(
            params["stellar_flux_multiplier"], params["warm_albedo"], params["ice_albedo"], co2_ppm
        )
        if year > 0:
            cold_severity = np.clip((8.0 - temp_c) / 20.0, 0.0, 1.0)
            contrast = np.clip((params["ice_albedo"] - params["warm_albedo"] - 0.18) / 0.45, 0.0, 1.0)
            response_years = TEMPERATURE_RESPONSE_YEARS * (1.0 + 1.40 * cold_severity * contrast)
            temp_c += (temp_eq_c - temp_c) / response_years
            temp_c -= _cold_trap_cooling(temp_c, co2_ppm, params)

        temp_c = float(np.clip(temp_c, -120.0, 120.0))
        co2_ppm = float(max(1.0, co2_ppm))
        temp_c_series[i] = temp_c
        co2_series[i] = co2_ppm

        if i == n - 1:
            continue

        e_human = _human_emissions_rate(
            params["emissions_growth_mode"],
            params["emissions_rate"],
            int(year),
            params["mitigation_start_year"],
            params["mitigation_strength"],
        )
        sink_weathering = _weathering_sink(temp_c, co2_ppm)
        sink_biosphere = _biosphere_sink(temp_c, co2_ppm)
        dco2_dt = e_human + NATURAL_OUTGASSING - sink_weathering - sink_biosphere
        co2_next = co2_ppm + dco2_dt * SIM_DT_YEARS
        if not np.isfinite(co2_next):
            co2_next = co2_ppm
        co2_ppm = float(max(1.0, co2_next))

    return {"years": years.tolist(), "temp_c": temp_c_series.tolist(), "co2_ppm": co2_series.tolist()}


def _seed_from_payload(payload: str):
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _planet_surface_fields(lon: np.ndarray, lat: np.ndarray, seed: int, temp_c: float):
    # This matches the procedural texture logic used in screen 1 (Scenario Builder).
    rng = np.random.default_rng(seed)
    n = np.zeros_like(lon, dtype=float)
    for _ in range(9):
        k1 = int(rng.integers(1, 7))
        k2 = int(rng.integers(1, 7))
        p1 = float(rng.uniform(0.0, 2.0 * np.pi))
        p2 = float(rng.uniform(0.0, 2.0 * np.pi))
        amp = float(rng.uniform(0.05, 0.22))
        n += amp * np.sin(k1 * lon + p1) * np.cos(k2 * lat + p2)

    n += 0.20 * np.sin(2.0 * lon + float(rng.uniform(0.0, 2.0 * np.pi)))
    n -= 0.16 * np.cos(3.0 * lat + float(rng.uniform(0.0, 2.0 * np.pi)))
    n = (n - np.min(n)) / (np.max(n) - np.min(n) + 1e-9)

    sea_level = 0.54
    land = n > sea_level
    land_height = np.clip((n - sea_level) / (1.0 - sea_level + 1e-9), 0.0, 1.0)
    ocean_depth = np.clip((sea_level - n) / (sea_level + 1e-9), 0.0, 1.0)

    ridges = np.abs(np.sin(9.0 * lon + float(rng.uniform(0.0, 2.0 * np.pi))) * np.cos(11.0 * lat))
    ridges *= np.abs(np.sin(15.0 * lon + float(rng.uniform(0.0, 2.0 * np.pi))))
    ridges = (ridges - np.min(ridges)) / (np.max(ridges) - np.min(ridges) + 1e-9)
    mountain = np.clip(0.65 * land_height + 0.55 * ridges, 0.0, 1.0)

    abs_lat = np.abs(lat) / (np.pi / 2.0)
    ice_threshold = np.clip(0.72 + 0.004 * temp_c, 0.56, 0.90)
    ice = abs_lat > ice_threshold

    relief = np.where(land, 0.02 + 0.05 * mountain, -0.004 * ocean_depth)
    relief = np.where(ice, relief + 0.012, relief)
    return land, land_height, ocean_depth, mountain, relief


def _texture_seed_for_map(params_payload: str):
    try:
        submitted_seed = st.session_state.get("submitted_texture_seed")
        if submitted_seed is not None:
            return int(submitted_seed)
        if "texture_seed" not in st.session_state:
            # Keep the same "current planet" seed model as screen 1.
            st.session_state["texture_seed"] = int(np.random.randint(0, 1_000_000_000))
        return int(st.session_state["texture_seed"])
    except Exception:
        return _seed_from_payload(params_payload)


@st.cache_data(show_spinner=False)
def _generate_geography(texture_seed: int, texture_temp_c: float):
    nlat = 72
    nlon = 144
    lat_deg = np.linspace(-89.0, 89.0, nlat, dtype=float)
    lon_deg = np.linspace(-180.0, 180.0, nlon, endpoint=False, dtype=float)
    lon2, lat2 = np.meshgrid(np.deg2rad(lon_deg), np.deg2rad(lat_deg))
    land, land_height, ocean_depth, mountain, relief = _planet_surface_fields(
        lon=lon2,
        lat=lat2,
        seed=int(texture_seed),
        temp_c=float(texture_temp_c),
    )

    elevation_km = np.where(
        land,
        0.25 + 5.8 * land_height + 2.2 * mountain,
        -5.2 * ocean_depth,
    )
    elevation_km = np.clip(elevation_km, -5.5, 8.0)
    microclimate_c = 1.1 * np.sin(4.0 * lon2) * np.cos(lat2) + 0.8 * mountain - 0.35 * ocean_depth + 0.6 * relief

    return {
        "lat_deg": lat_deg.tolist(),
        "lon_deg": lon_deg.tolist(),
        "elevation_km": elevation_km.tolist(),
        "microclimate_c": microclimate_c.tolist(),
        "land_mask": land.astype(float).tolist(),
    }


def _local_temperature_c(global_temp_c: float, lat_deg: np.ndarray, elevation_km: np.ndarray, microclimate_c: np.ndarray, params):
    lat_rad = np.deg2rad(lat_deg)[:, None]
    hot_anomaly = max(0.0, global_temp_c - 22.0)
    cold_anomaly = max(0.0, 8.0 - global_temp_c)
    albedo_contrast = max(0.0, params["ice_albedo"] - params["warm_albedo"])
    seasonality = DEFAULT_SEASONALITY_AMPLITUDE if params["enable_seasonality"] else 0.0

    equatorial_heat_excess = float(np.clip(4.8 + 1.05 * hot_anomaly - 0.08 * cold_anomaly + 6.0 * seasonality, 2.5, 18.0))
    polar_cooling_strength = float(
        np.clip(20.0 + 0.55 * cold_anomaly + 0.65 * hot_anomaly + 6.0 * max(0.0, albedo_contrast - 0.20) + 12.0 * seasonality, 12.0, 42.0)
    )
    lat_band_temp_c = global_temp_c + equatorial_heat_excess * (np.cos(lat_rad) ** 2) - polar_cooling_strength * (np.sin(lat_rad) ** 2)
    return lat_band_temp_c + microclimate_c - 6.5 * elevation_km


def _tile_co2_ppm(global_co2_ppm: float, elevation_km: np.ndarray):
    pressure_factor = np.exp(-np.clip(elevation_km, -1.0, 8.0) / 8.5)
    basin_boost = 1.0 + 0.04 * np.clip(-elevation_km, 0.0, 4.0)
    return np.clip(global_co2_ppm * pressure_factor * basin_boost, 1.0, None)


def _temperature_suitability(local_temp_c: np.ndarray, tmin_c: float, tmax_c: float):
    center_c = 0.5 * (tmin_c + tmax_c)
    half_width_c = max(1.0, 0.5 * (tmax_c - tmin_c))
    distance = np.abs(local_temp_c - center_c) / half_width_c
    inside_score = 1.0 - 0.30 * np.power(np.clip(distance, 0.0, 1.0), 1.5)
    outside_score = 0.30 * np.exp(-np.clip(distance - 1.0, 0.0, None) / 0.45)
    return np.clip(np.where(distance <= 1.0, inside_score, outside_score), 0.0, 1.0)


def _co2_suitability(tile_co2_ppm: np.ndarray):
    score = np.ones_like(tile_co2_ppm, dtype=float)

    mid = (tile_co2_ppm > 450.0) & (tile_co2_ppm <= 1000.0)
    score[mid] = 1.0 - 0.45 * np.clip((tile_co2_ppm[mid] - 450.0) / 550.0, 0.0, 1.0)

    high = (tile_co2_ppm > 1000.0) & (tile_co2_ppm <= 2000.0)
    score[high] = 0.55 - 0.45 * np.clip((tile_co2_ppm[high] - 1000.0) / 1000.0, 0.0, 1.0)

    extreme = tile_co2_ppm > 2000.0
    score[extreme] = 0.10 * np.exp(-(tile_co2_ppm[extreme] - 2000.0) / 500.0)

    return np.clip(score, 0.0, 1.0)


def _temperature_hard_constraint(local_temp_c: np.ndarray, tmin_c: float, tmax_c: float):
    return (local_temp_c >= tmin_c) & (local_temp_c <= tmax_c)


def _co2_hard_constraint(tile_co2_ppm: np.ndarray):
    return tile_co2_ppm <= CO2_HUMAN_MAX_SAFE_PPM


def _human_habitability(local_temp_c: np.ndarray, tile_co2_ppm: np.ndarray, params):
    temp_ok = _temperature_hard_constraint(local_temp_c, params["habitable_temp_min_c"], params["habitable_temp_max_c"])
    co2_ok = _co2_hard_constraint(tile_co2_ppm)
    temp_score = _temperature_suitability(local_temp_c, params["habitable_temp_min_c"], params["habitable_temp_max_c"])
    co2_score = _co2_suitability(tile_co2_ppm)
    score = np.clip(temp_score * co2_score, 0.0, 1.0)
    score = np.where(temp_ok & co2_ok, score, 0.0)
    return score, temp_score, co2_score, temp_ok, co2_ok


def _habitability_map(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    score: np.ndarray,
    land_mask: np.ndarray,
    livable_threshold: float,
    year: int,
):
    lon2, lat2 = np.meshgrid(lon_deg, lat_deg)
    plot_score = np.where(land_mask, score, np.nan)
    is_livable = (np.nan_to_num(plot_score.ravel(), nan=-1.0) >= livable_threshold).astype(int)
    surface_label = np.where(land_mask.ravel() > 0.5, "Land", "Water")
    custom = np.column_stack((np.nan_to_num(plot_score.ravel(), nan=0.0), is_livable, surface_label))
    fig = go.Figure(
        data=[
            go.Scattergeo(
                lat=lat2.ravel(),
                lon=lon2.ravel(),
                mode="markers",
                marker=dict(
                    size=3.8,
                    color=plot_score.ravel(),
                    colorscale=HUMAN_HABITABILITY_COLORSCALE,
                    cmin=0.0,
                    cmax=1.0,
                    line=dict(width=0),
                    opacity=0.96,
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
                    "Lat %{lat:.1f}<br>Lon %{lon:.1f}"
                    "<br>Surface %{customdata[2]}"
                    "<br>Habitability %{customdata[0]:.2f}"
                    "<br>Livable %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=590,
        title=f"Human Livability Map - Year {year}",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d9e3f5"),
        geo=dict(
            projection_type="orthographic",
            showland=False,
            showcoastlines=False,
            showocean=True,
            oceancolor="#0f1626",
            projection_rotation=dict(lat=PLANET_VIEW_LAT, lon=PLANET_VIEW_LON),
            bgcolor="rgba(0,0,0,0)",
            lataxis_showgrid=False,
            lonaxis_showgrid=False,
        ),
    )
    return fig


def render_map_page():
    st.markdown("<style>[data-testid='stHeaderActionElements']{display:none;}</style>", unsafe_allow_html=True)
    st.title("Refugia Map")
    st.caption("Shows where humankind can live based on tile temperature and tile CO2, with static red-to-green scale.")

    params = _load_current_scenario()
    payload = json.dumps(params, sort_keys=True)
    climate_cache = _load_climate_twin_series_cache(params)
    if climate_cache is None:
        sim = _simulate_global_series(payload)
        years = np.array(sim["years"], dtype=int)
        global_temp_c = np.array(sim["temp_c"], dtype=float)
        global_co2_ppm = np.array(sim["co2_ppm"], dtype=float)
        official_habitable_pct_series = None
    else:
        years = np.asarray(climate_cache["years"], dtype=int)
        global_temp_c = np.asarray(climate_cache["global_temperature_c"], dtype=float)
        global_co2_ppm = np.asarray(climate_cache["co2_ppm"], dtype=float)
        official_habitable_pct_series = np.asarray(climate_cache["habitable_surface_percent"], dtype=float)

    texture_seed = _texture_seed_for_map(payload)
    # Use the same texture seed as screen 1; water/land boundaries then match that planet.
    geo = _generate_geography(
        texture_seed=texture_seed,
        texture_temp_c=float(global_temp_c[min(100, len(global_temp_c) - 1)]),
    )

    lat_deg = np.array(geo["lat_deg"], dtype=float)
    lon_deg = np.array(geo["lon_deg"], dtype=float)
    elevation_km = np.array(geo["elevation_km"], dtype=float)
    microclimate_c = np.array(geo["microclimate_c"], dtype=float)
    land_mask = np.array(geo["land_mask"], dtype=float) >= 0.5

    controls = st.columns([2.0, 1.7, 1.4, 1.2])
    default_year = int(
        np.clip(
            st.session_state.get(CLIMATE_TWIN_YEAR_KEY, min(100, int(years[-1]))),
            int(years[0]),
            int(years[-1]),
        )
    )
    year = controls[0].slider(
        "Year",
        min_value=int(years[0]),
        max_value=int(years[-1]),
        value=default_year,
        step=1,
    )
    st.session_state[CLIMATE_TWIN_YEAR_KEY] = int(year)
    livable_threshold = controls[1].slider(
        "Livable Threshold",
        min_value=0.30,
        max_value=0.70,
        value=0.45,
        step=0.01,
        help="Tile is counted as livable when habitability score is >= threshold.",
    )
    show_livable_only = controls[2].toggle("Show Livable Only", value=False)
    auto_center = controls[3].toggle("Auto Center", value=True)

    idx = int(np.argmin(np.abs(years - year)))
    official_habitable_pct = None
    official_habitability_score = None
    if official_habitable_pct_series is not None and idx < len(official_habitable_pct_series):
        official_habitable_pct = float(np.clip(official_habitable_pct_series[idx], 0.0, 100.0))
        official_habitability_score = official_habitable_pct / 100.0

    temp_grid_c = _local_temperature_c(global_temp_c[idx], lat_deg, elevation_km, microclimate_c, params)
    tile_co2 = _tile_co2_ppm(float(global_co2_ppm[idx]), elevation_km)
    score, temp_score, co2_score, temp_ok, co2_ok = _human_habitability(temp_grid_c, tile_co2, params)
    # Hard constraint from user requirement: water is always unlivable.
    score = np.where(land_mask, score, 0.0)

    if show_livable_only:
        plot_score = np.where(score >= livable_threshold, score, 0.0)
    else:
        plot_score = score

    fig = _habitability_map(lat_deg, lon_deg, plot_score, land_mask, livable_threshold, int(years[idx]))
    if not auto_center:
        center_col1, center_col2 = st.columns(2)
        center_lat = center_col1.slider("Center Latitude", min_value=-90, max_value=90, value=PLANET_VIEW_LAT)
        center_lon = center_col2.slider("Center Longitude", min_value=-180, max_value=180, value=PLANET_VIEW_LON)
        fig.update_geos(projection_rotation=dict(lat=center_lat, lon=center_lon))
    st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.subheader("Map Statistics")
        livable_mask = (score >= livable_threshold) & land_mask
        perfect_mask = score >= 0.80
        hard_unlivable_mask = (~temp_ok | ~co2_ok) & land_mask
        land_fraction = 100.0 * float(np.mean(land_mask))
        livable_on_land = 100.0 * float(np.sum(livable_mask) / max(1.0, np.sum(land_mask)))
        hard_unlivable_on_land = 100.0 * float(np.sum(hard_unlivable_mask) / max(1.0, np.sum(land_mask)))
        map_livable_global_pct = 100.0 * float(np.mean(livable_mask))
        map_mean_score = float(np.mean(score))
        stats_lines = [
            f"- Texture seed (same as screen 1): {texture_seed}",
            f"- Land area: {land_fraction:.1f}% (water is forced unlivable)",
            f"- Global temperature: {float(global_temp_c[idx]):.2f} C",
            f"- Global CO2: {float(global_co2_ppm[idx]):.1f} ppm",
        ]
        if official_habitable_pct is not None and official_habitability_score is not None:
            stats_lines.extend(
                [
                    f"- Mean habitability score (from Climate Twin): {official_habitability_score:.3f}",
                    f"- Livable area (from Climate Twin): {official_habitable_pct:.1f}%",
                ]
            )
        else:
            stats_lines.extend(
                [
                    f"- Mean habitability score (map fallback): {map_mean_score:.3f}",
                    f"- Livable area (map fallback, score >= {livable_threshold:.2f}): {map_livable_global_pct:.1f}%",
                    "- Climate Twin cache not found for this scenario. Open screen 01 to sync exact habitability values.",
                ]
            )
        stats_lines.extend(
            [
                f"- Map livable area (score >= {livable_threshold:.2f}): {map_livable_global_pct:.1f}%",
                f"- Livable fraction of land: {livable_on_land:.1f}%",
                f"- Perfect area (score >= 0.80): {100.0 * float(np.mean(perfect_mask)):.1f}%",
                (
                    f"- Hard red area on land (temp outside {params['habitable_temp_min_c']:.1f} to "
                    f"{params['habitable_temp_max_c']:.1f} C OR CO2 > {CO2_HUMAN_MAX_SAFE_PPM:.0f} ppm): "
                    f"{hard_unlivable_on_land:.1f}%"
                ),
                f"- Tile temperature range: {float(np.min(temp_grid_c)):.2f} to {float(np.max(temp_grid_c)):.2f} C",
                f"- Tile CO2 range: {float(np.min(tile_co2)):.0f} to {float(np.max(tile_co2)):.0f} ppm",
                f"- Mean temperature suitability: {float(np.mean(temp_score)):.3f}",
                f"- Mean CO2 suitability: {float(np.mean(co2_score)):.3f}",
            ]
        )
        st.markdown(
            "\n".join(stats_lines)
        )


render_map_page()
