import json

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
import urllib.parse
import urllib.request

from htp.model.bounds import BOUNDS
from htp.model.constants import (
    ALBEDO_REF,
    BIOSPHERE_COEFF,
    BIOSPHERE_OPT_TEMP_C,
    BIOSPHERE_TEMP_WIDTH_C,
    CO2_BASELINE_PPM,
    DEFAULT_ICE_TRANSITION_TEMP_C,
    DEFAULT_K_CO2,
    DEFAULT_PLANET_RADIUS_KM,
    DEFAULT_SEASONALITY_AMPLITUDE,
    EMISSIONS_MODES,
    HABITABILITY_PROFILES,
    ICE_TRANSITION_WIDTH_C,
    K_ALBEDO,
    K_FLUX,
    NATURAL_OUTGASSING_PPM_PER_YEAR,
    PREVIEW_HORIZON_YEARS,
    REFERENCE_TEMP_C,
    SOLAR_CONSTANT,
    WEATHERING_COEFF,
    WEATHERING_TEMP_SENS,
)
from htp.model.imports import normalize_imported_planet_flat
from htp.model.scenario_io import export_scenario_json, scenario_from_flat_params, scenario_to_flat_params
from htp.model.safety import safe_float
from htp.model.simulate import preview_state
from htp.model.earth import earth_surface_fields, prepare_surface_render_grid, surface_grid_to_xyz

st.set_page_config(page_title="Scenario Builder", layout="wide")

PLANET_RIGHT_TILT_DEG = 23.5
PLANET_INITIAL_VIEW_LON_DEG = 38.0
PLANET_SPIN_SPEED_DEG_PER_SEC = 10.2
STELLAR_FLUX_INPUT_MIN = BOUNDS["stellar_flux_multiplier"][0]
STELLAR_FLUX_INPUT_MAX = BOUNDS["stellar_flux_multiplier"][1]
CO2_INPUT_MIN = BOUNDS["initial_co2_ppm"][0]
CO2_INPUT_MAX = BOUNDS["initial_co2_ppm"][1]
EMISSIONS_INPUT_MIN = BOUNDS["emissions_rate"][0]
EMISSIONS_INPUT_MAX = BOUNDS["emissions_rate"][1]
MITIGATION_START_MIN = int(BOUNDS["mitigation_start_year"][0])
MITIGATION_START_MAX = int(BOUNDS["mitigation_start_year"][1])
HAB_TEMP_MIN = BOUNDS["habitable_temp_min_c"][0]
HAB_TEMP_MAX = BOUNDS["habitable_temp_max_c"][1]
SEED_MIN = int(BOUNDS["seed"][0])
SEED_MAX = int(BOUNDS["seed"][1])
EXOPLANET_ATMOSPHERE_OPTIONS = ("Minimal", "Earth-like", "Dense", "Custom")

INTERNAL_MODEL_CONSTANTS = {
    "co2_baseline_ppm": CO2_BASELINE_PPM,
    "preview_horizon_years": PREVIEW_HORIZON_YEARS,
    "reference_temp_c": REFERENCE_TEMP_C,
    "albedo_ref": ALBEDO_REF,
    "seasonality_amplitude": DEFAULT_SEASONALITY_AMPLITUDE,
    "ice_transition_temp_c": DEFAULT_ICE_TRANSITION_TEMP_C,
    "ice_transition_width_c": ICE_TRANSITION_WIDTH_C,
    "natural_outgassing_ppm_per_year": NATURAL_OUTGASSING_PPM_PER_YEAR,
    "biosphere_opt_temp_c": BIOSPHERE_OPT_TEMP_C,
    "biosphere_temp_width_c": BIOSPHERE_TEMP_WIDTH_C,
    "k_flux": K_FLUX,
    "k_albedo": K_ALBEDO,
    "k_co2": DEFAULT_K_CO2,
    "weathering_coeff": WEATHERING_COEFF,
    "weathering_temp_sens": WEATHERING_TEMP_SENS,
    "biosphere_coeff": BIOSPHERE_COEFF,
    "bounds": dict(BOUNDS),
    "planet_radius_km": DEFAULT_PLANET_RADIUS_KM,
}

REMOVED_UI_KEYS = (
    "planet_radius_km",
    "orbital_distance_au",
    "seasonality_amplitude",
    "obliquity_deg",
    "ice_transition_temp_c",
    "heat_transport_strength",
    "climate_heat_capacity",
    "volcanic_outgassing",
    "weathering_strength",
    "biosphere_uptake_strength",
    "carbon_removal_sensitivity",
    "carbon_capture_rate",
    "seasonality_enabled",
)

EARTHLIKE_PLANET_BASE = {
    "stellar_flux_multiplier": 1.00,
    "enable_seasonality": True,
    "warm_albedo": 0.30,
    "ice_albedo": 0.62,
    "initial_co2_ppm": 420.0,
    "habitability_profile": "Liquid Water",
    "habitable_temp_min_c": 0.0,
    "habitable_temp_max_c": 45.0,
    "K_CO2": DEFAULT_K_CO2,
    "seed": 42,
    "natural_planet_mode": False,
    "atmosphere_assumption": None,
    "imported_equilibrium_temperature_k": None,
    "imported_stellar_flux_multiplier": None,
    "import_classification": None,
}
EARTHLIKE_CIV_BASELINE = {
    "emissions_rate": 1.0,
    "emissions_growth_mode": "Constant",
    "mitigation_start_year": 40,
    "mitigation_strength": 0.15,
}
EARTHLIKE_CIV_CAREFREE = {
    "emissions_rate": 4.8,
    "emissions_growth_mode": "Carefree",
    "mitigation_start_year": 120,
    "mitigation_strength": 0.02,
}
EARTHLIKE_CIV_STABILIZATION = {
    "emissions_rate": 2.0,
    "emissions_growth_mode": "Stabilization",
    "mitigation_start_year": 5,
    "mitigation_strength": 0.78,
}

EARTHLIKE_SHARED_INITIAL_KEYS = tuple(EARTHLIKE_PLANET_BASE.keys())
EARTHLIKE_POLICY_KEYS = (
    "emissions_rate",
    "emissions_growth_mode",
    "mitigation_start_year",
    "mitigation_strength",
)
SCENARIO_INPUT_KEYS = (
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
SCENARIO_HIDDEN_KEYS = (
    "K_CO2",
    "seed",
    "natural_planet_mode",
    "atmosphere_assumption",
    "imported_equilibrium_temperature_k",
    "imported_stellar_flux_multiplier",
    "import_classification",
)
SCENARIO_ALL_KEYS = SCENARIO_INPUT_KEYS + SCENARIO_HIDDEN_KEYS


def _merge_planet_and_civ(planet: dict, civ: dict):
    merged = dict(planet)
    merged.update(civ)
    return merged

PRESETS = {
    "Earth-like Baseline": _merge_planet_and_civ(EARTHLIKE_PLANET_BASE, EARTHLIKE_CIV_BASELINE),
    "Carefree Civilization": _merge_planet_and_civ(EARTHLIKE_PLANET_BASE, EARTHLIKE_CIV_CAREFREE),
    "Stabilization Policy": _merge_planet_and_civ(EARTHLIKE_PLANET_BASE, EARTHLIKE_CIV_STABILIZATION),
    "Snowball-Prone World": _merge_planet_and_civ(
        {
            **EARTHLIKE_PLANET_BASE,
            "stellar_flux_multiplier": 0.86,
            "warm_albedo": 0.34,
            "ice_albedo": 0.78,
            "initial_co2_ppm": 260.0,
            "habitability_profile": "Conservative Biosphere",
            "habitable_temp_min_c": -5.0,
            "habitable_temp_max_c": 35.0,
            "seed": 43,
        },
        {
            "emissions_rate": 1.0,
            "emissions_growth_mode": "Constant",
            "mitigation_start_year": 40,
            "mitigation_strength": 0.25,
        },
    ),
    "Runaway-Prone World": _merge_planet_and_civ(
        {
            **EARTHLIKE_PLANET_BASE,
            "stellar_flux_multiplier": 1.12,
            "warm_albedo": 0.23,
            "ice_albedo": 0.52,
            "initial_co2_ppm": 700.0,
            "habitability_profile": "Broad Microbial Tolerance",
            "habitable_temp_min_c": 0.0,
            "habitable_temp_max_c": 50.0,
            "seed": 44,
        },
        {
            "emissions_rate": 5.0,
            "emissions_growth_mode": "Growing",
            "mitigation_start_year": 120,
            "mitigation_strength": 0.08,
        },
    ),
}

EXOPLANET_ARCHIVE_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
EXOPLANET_ARCHIVE_QUERY = (
    "select pl_name,hostname,pl_insol,pl_eqt,pl_orbeccen "
    "from ps where default_flag=1 and pl_name is not null and pl_insol is not null"
)
EXOPLANET_PRESET_PREFIX = "Exoplanet: "
EXOPLANET_SAMPLE_SIZE = 100


def _safe_float(value, default=np.nan):
    return safe_float(value, default)


def _is_exoplanet_preset_name(name: str):
    return isinstance(name, str) and name.startswith(EXOPLANET_PRESET_PREFIX)


def _is_known_preset_name(name: str):
    return isinstance(name, str) and (name in PRESETS or _is_exoplanet_preset_name(name))


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_exoplanet_rows():
    params = urllib.parse.urlencode({"query": EXOPLANET_ARCHIVE_QUERY, "format": "json"})
    url = f"{EXOPLANET_ARCHIVE_TAP_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []

    rows = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        name = str(row.get("pl_name", "")).strip()
        if not name:
            continue
        rows.append(
            {
                "pl_name": name,
                "hostname": str(row.get("hostname", "")).strip(),
                "pl_insol": _safe_float(row.get("pl_insol")),
                "pl_eqt": _safe_float(row.get("pl_eqt")),
                "pl_orbeccen": _safe_float(row.get("pl_orbeccen")),
            }
        )
    return rows


def _refresh_exoplanet_sample():
    rows = _fetch_exoplanet_rows()
    if not rows:
        st.session_state["exoplanet_random_sample"] = []
        return
    take = min(EXOPLANET_SAMPLE_SIZE, len(rows))
    rng = np.random.default_rng()
    idx = rng.choice(len(rows), size=take, replace=False)
    sample = [rows[int(i)] for i in idx]
    sample.sort(key=lambda item: str(item.get("pl_name", "")).lower())
    st.session_state["exoplanet_random_sample"] = sample


def _format_exoplanet_option(row: dict):
    name = str(row.get("pl_name", "Unknown")).strip() or "Unknown"
    host = str(row.get("hostname", "")).strip()
    insol = _safe_float(row.get("pl_insol"))
    eqt_k = _safe_float(row.get("pl_eqt"))
    ecc = _safe_float(row.get("pl_orbeccen"))
    host_suffix = f" @ {host}" if host else ""
    insol_text = f"{insol:.2f} S_earth" if np.isfinite(insol) else "S_earth n/a"
    eqt_text = f"{eqt_k:.0f} K" if np.isfinite(eqt_k) else "Teq n/a"
    ecc_text = f"e={ecc:.2f}" if np.isfinite(ecc) else "e=n/a"
    return f"{EXOPLANET_PRESET_PREFIX}{name}{host_suffix} [{insol_text}, {eqt_text}, {ecc_text}]"


def _build_exoplanet_option_map():
    sample = st.session_state.get("exoplanet_random_sample", [])
    if not isinstance(sample, list):
        return {}
    mapping = {}
    for idx, row in enumerate(sample):
        if not isinstance(row, dict):
            continue
        label = _format_exoplanet_option(row)
        if label in mapping:
            label = f"{label} #{idx + 1}"
        mapping[label] = row
    return mapping


def _apply_exoplanet_preset(name: str, row: dict, atmosphere_assumption: str):
    raw = dict(row)
    if atmosphere_assumption == "Custom":
        raw["initial_co2_ppm"] = st.session_state.get("initial_co2_ppm", CO2_INPUT_MIN)
    preset = normalize_imported_planet_flat(raw, atmosphere_assumption=atmosphere_assumption)
    for key in SCENARIO_ALL_KEYS:
        if key in preset:
            st.session_state[key] = preset[key]
    st.session_state["builder_persisted_preset_name"] = name
    st.session_state["builder_persisted_inputs"] = {key: st.session_state.get(key) for key in SCENARIO_ALL_KEYS}
    st.session_state["selected_exoplanet_row"] = dict(row)
    scenario = scenario_from_flat_params(
        _collect_inputs(),
        preset_name=name,
    )
    st.session_state["submitted_scenario_snapshot"] = scenario.model_dump(mode="json")


def draw_planet(temp_c: float, co2_ppm: float, albedo: float, stellar_flux: float, radius_km: float):
    heat = max(0.0, min(1.0, (temp_c + 30.0) / 80.0))
    co2_factor = max(0.0, min(1.0, co2_ppm / 1400.0))
    light = max(0.2, min(1.3, stellar_flux / SOLAR_CONSTANT))

    lat_deg = np.linspace(-89.0, 89.0, 100, dtype=float)
    lon_deg = np.linspace(-180.0, 180.0, 100, endpoint=False, dtype=float)
    lon2_deg, lat2_deg = np.meshgrid(lon_deg, lat_deg)
    lon = np.deg2rad(np.asarray(lon2_deg, dtype=float))
    lat = np.deg2rad(np.asarray(lat2_deg, dtype=float))
    radius_scale = np.clip(radius_km / 6371.0, 0.35, 2.2)

    base_r = 0.17 + 0.76 * heat
    base_g = 0.58 - 0.26 * heat + 0.08 * (1.0 - co2_factor)
    base_b = 0.88 - 0.67 * heat
    brightness = (0.72 + 0.35 * (1.0 - albedo + 0.2)) * light
    color = np.clip(np.array([base_r, base_g, base_b]) * brightness, 0, 1)

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

    c0 = np.clip(color * np.array([0.15, 0.30, 0.90]), 0, 1)
    c1 = np.clip(color * np.array([0.30, 0.65, 1.05]), 0, 1)
    c2 = np.clip(color * np.array([0.92, 0.84, 0.56]), 0, 1)
    c3 = np.clip(color * np.array([0.42, 0.88, 0.42]), 0, 1)
    c4 = np.clip(color * np.array([0.23, 0.60, 0.28]), 0, 1)
    c5 = np.clip(color * np.array([0.60, 0.58, 0.55]), 0, 1)
    c6 = np.array([0.96, 0.97, 1.0])

    temp_norm = float(np.clip((temp_c + 40.0) / 110.0, 0.0, 1.0))
    cold_tint = np.array([0.22, 0.48, 1.00])
    hot_tint = np.array([1.00, 0.28, 0.10])
    temp_tint = (1.0 - temp_norm) * cold_tint + temp_norm * hot_tint

    c0 = np.clip(0.80 * c0 + 0.20 * temp_tint, 0, 1)
    c1 = np.clip(0.75 * c1 + 0.25 * temp_tint, 0, 1)
    c2 = np.clip(0.55 * c2 + 0.45 * temp_tint, 0, 1)
    c3 = np.clip(0.55 * c3 + 0.45 * temp_tint, 0, 1)
    c4 = np.clip(0.55 * c4 + 0.45 * temp_tint, 0, 1)
    c5 = np.clip(0.55 * c5 + 0.45 * temp_tint, 0, 1)

    colorscale = [
        [0.00, f"rgb({int(255 * c0[0])}, {int(255 * c0[1])}, {int(255 * c0[2])})"],
        [0.28, f"rgb({int(255 * c1[0])}, {int(255 * c1[1])}, {int(255 * c1[2])})"],
        [0.47, f"rgb({int(255 * c2[0])}, {int(255 * c2[1])}, {int(255 * c2[2])})"],
        [0.63, f"rgb({int(255 * c3[0])}, {int(255 * c3[1])}, {int(255 * c3[2])})"],
        [0.80, f"rgb({int(255 * c4[0])}, {int(255 * c4[1])}, {int(255 * c4[2])})"],
        [0.92, f"rgb({int(255 * c5[0])}, {int(255 * c5[1])}, {int(255 * c5[2])})"],
        [1.00, f"rgb({int(255 * c6[0])}, {int(255 * c6[1])}, {int(255 * c6[2])})"],
    ]

    # Extra south-pole surface cap to fully hide any pinhole and guarantee a visibly wide snow circle.
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
    # Lift the snow cap slightly above the globe so it cleanly overpaints the pole area.
    south_cap_r = radius_scale * (1.0 + south_cap_rel_closed + 0.008)
    south_cap_x = south_cap_r * np.asarray(south_cap_xyz["x"], dtype=float)
    south_cap_y = south_cap_r * np.asarray(south_cap_xyz["y"], dtype=float)
    south_cap_z = south_cap_r * np.asarray(south_cap_xyz["z"], dtype=float)
    south_cap_colorscale = [
        [0.0, f"rgb({int(255 * c6[0])}, {int(255 * c6[1])}, {int(255 * c6[2])})"],
        [1.0, f"rgb({int(255 * c6[0])}, {int(255 * c6[1])}, {int(255 * c6[2])})"],
    ]

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=texture_closed,
                colorscale=colorscale,
                showscale=False,
                lighting=dict(ambient=0.45, diffuse=0.8, specular=0.3, roughness=0.85),
                lightposition=dict(x=120, y=80, z=200),
            ),
            go.Surface(
                x=south_cap_x,
                y=south_cap_y,
                z=south_cap_z,
                surfacecolor=south_cap_tex_closed,
                colorscale=south_cap_colorscale,
                cmin=0.0,
                cmax=1.0,
                opacity=1.0,
                showscale=False,
                hoverinfo="skip",
                lighting=dict(ambient=0.62, diffuse=0.55, specular=0.06, roughness=0.96),
                lightposition=dict(x=120, y=80, z=200),
            ),
        ]
    )
    view_phase_rad = np.deg2rad(float(PLANET_INITIAL_VIEW_LON_DEG))
    eye_radius = 2.05 + 0.25 * radius_scale
    eye_x = eye_radius * float(np.cos(view_phase_rad))
    eye_y = eye_radius * float(np.sin(view_phase_rad))
    eye_z = 0.98 + 0.14 * radius_scale
    tilt_rad = np.deg2rad(PLANET_RIGHT_TILT_DEG)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=300,
        clickmode="event+select",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            bgcolor="rgb(10, 14, 24)",
            camera=dict(
                eye=dict(x=eye_x, y=eye_y, z=eye_z),
                up=dict(x=float(np.sin(tilt_rad)), y=0.0, z=float(np.cos(tilt_rad))),
            ),
        ),
    )
    return fig


def _risk_label(score: float):
    if score < 0.33:
        return "Low"
    if score < 0.66:
        return "Elevated"
    return "High"


def _status_color(label: str):
    normalized = str(label).strip().lower()
    if normalized in {"stable", "low", "mild"}:
        return "#2E8B57"
    if normalized in {"marginal", "elevated", "moderate"}:
        return "#D2A106"
    if normalized == "cold":
        return "#3B82C4"
    return "#B52A2A"


def _kpi_temperature_regime(temp_k: float):
    if temp_k < 283.0:
        return "Cold"
    if temp_k <= 295.0:
        return "Mild"
    return "Hot"


def _kpi_status_co2(co2_ppm: float):
    ratio = max(co2_ppm, 1.0) / CO2_BASELINE_PPM
    if ratio < 1.5:
        return "Low"
    if ratio < 2.5:
        return "Elevated"
    return "High"


def _kpi_habitability_level(habitable_surface_pct: float):
    if habitable_surface_pct < 40.0:
        return "Low"
    if habitable_surface_pct < 75.0:
        return "Moderate"
    return "High"


def _kpi_card(label: str, value: str, status: str, status_color: str | None = None):
    color = status_color or _status_color(status)
    st.markdown(
        (
            "<div style='padding:10px 12px;border-radius:10px;border:1px solid #273249;"
            "background:#0f1626;margin-bottom:8px;'>"
            f"<div style='font-size:11px;letter-spacing:0.04em;text-transform:uppercase;color:#96a3bd;'>{label}</div>"
            f"<div style='font-size:27px;font-weight:700;line-height:1.2;color:#f4f8ff;margin-top:4px;'>{value}</div>"
            f"<div style='margin-top:6px;font-size:12px;font-weight:600;color:{color};'>{status}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _badge(label: str, status: str):
    color = _status_color(status)
    st.markdown(
        (
            "<div style='padding:10px 12px;border-radius:10px;border:1px solid #273249;"
            "margin-bottom:8px;background:#10192b;display:flex;justify-content:space-between;align-items:center;'>"
            f"<span style='font-weight:600;color:#d9e3f5;'>{label}</span>"
            f"<span style='color:{color};font-weight:700;border:1px solid {color};"
            f"padding:2px 8px;border-radius:999px;background:{color}22;'>{status}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _section_header(title: str, first: bool = False):
    if not first:
        st.markdown("<div style='height:1.3rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<h3 style='font-size:1.45rem;line-height:1.2;margin:0 0 0.6rem 0;'>{title}</h3>",
        unsafe_allow_html=True,
    )


def _render_spinning_surface(fig: go.Figure, component_key: str, height_px: int):
    div_id = f"{component_key}_plot"
    # Client-side spin keeps Streamlit widgets responsive.
    post_script = f"""
const plot = document.getElementById('{div_id}');
if (plot) {{
  const speedDegPerSec = {PLANET_SPIN_SPEED_DEG_PER_SEC:.3f};
  const interactionPauseMs = 1300;
  const toRad = Math.PI / 180.0;
  const initialEye = plot.layout?.scene?.camera?.eye ?? {{x: 1.8, y: 1.4, z: 1.0}};
  const initialUp = plot.layout?.scene?.camera?.up ?? {{x: 0.0, y: 0.0, z: 1.0}};
  const radius = Math.max(0.5, Math.hypot(initialEye.x ?? 1.8, initialEye.y ?? 1.4));
  const z = Number.isFinite(initialEye.z) ? initialEye.z : 1.0;
  let angleDeg = (Math.atan2(initialEye.y ?? 1.4, initialEye.x ?? 1.8) / toRad + 360.0) % 360.0;
  let lastTs = performance.now();
  let userPauseUntil = 0;
  let internalUpdate = false;
  let pending = false;
  const pauseFromInteraction = () => {{ userPauseUntil = performance.now() + interactionPauseMs; }};
  plot.addEventListener('pointerdown', pauseFromInteraction, {{ passive: true }});
  plot.addEventListener('wheel', pauseFromInteraction, {{ passive: true }});
  plot.addEventListener('touchstart', pauseFromInteraction, {{ passive: true }});
  plot.on('plotly_click', pauseFromInteraction);
  plot.on('plotly_doubleclick', pauseFromInteraction);
  plot.on('plotly_relayouting', () => {{ if (!internalUpdate) pauseFromInteraction(); }});
  plot.on('plotly_relayout', (ev) => {{
    if (internalUpdate) return;
    if (!ev) return;
    if (Object.prototype.hasOwnProperty.call(ev, 'scene.camera') ||
        Object.prototype.hasOwnProperty.call(ev, 'scene.camera.eye') ||
        Object.prototype.hasOwnProperty.call(ev, 'scene.camera.up')) {{
      pauseFromInteraction();
    }}
  }});
  const applyCamera = () => {{
    if (pending) return;
    pending = true;
    internalUpdate = true;
    const finalize = () => {{
      pending = false;
      internalUpdate = false;
    }};
    const angleRad = angleDeg * toRad;
    const result = Plotly.relayout(plot, {{
      'scene.camera.eye': {{x: radius * Math.cos(angleRad), y: radius * Math.sin(angleRad), z: z}},
      'scene.camera.up': initialUp
    }});
    if (result && typeof result.then === 'function') {{
      result.then(finalize).catch(finalize);
    }} else {{
      finalize();
    }}
  }};
  const tick = (ts) => {{
    const dt = Math.max(0.0, Math.min(0.2, (ts - lastTs) / 1000.0));
    lastTs = ts;
    if (ts >= userPauseUntil) {{
      angleDeg = (angleDeg + speedDegPerSec * dt) % 360.0;
      applyCamera();
    }}
    requestAnimationFrame(tick);
  }};
  requestAnimationFrame(tick);
}}
"""
    html = pio.to_html(
        fig,
        include_plotlyjs="inline",
        full_html=False,
        default_width="100%",
        default_height=f"{int(height_px)}px",
        div_id=div_id,
        post_script=post_script,
        config={"displayModeBar": True, "responsive": True},
    )
    components.html(html, height=int(height_px) + 12, scrolling=False)


def _initialize_state():
    defaults = PRESETS["Earth-like Baseline"]
    forced_preset_name = st.session_state.pop("force_preset_name_once", None)
    persisted_inputs = st.session_state.get("builder_persisted_inputs")
    if not isinstance(persisted_inputs, dict):
        persisted_inputs = dict(defaults)
        st.session_state["builder_persisted_inputs"] = persisted_inputs

    for key in SCENARIO_ALL_KEYS:
        st.session_state.setdefault(key, persisted_inputs.get(key, defaults.get(key)))

    if isinstance(forced_preset_name, str) and forced_preset_name in PRESETS:
        desired_preset_name = forced_preset_name
        st.session_state["preset_name"] = desired_preset_name
    else:
        current_widget_preset = st.session_state.get("preset_name")
        if _is_known_preset_name(current_widget_preset):
            desired_preset_name = current_widget_preset
        else:
            desired_preset_name = str(st.session_state.get("builder_persisted_preset_name", "Earth-like Baseline"))
            if not _is_known_preset_name(desired_preset_name):
                desired_preset_name = "Earth-like Baseline"
            st.session_state["preset_name"] = desired_preset_name
    st.session_state["builder_persisted_preset_name"] = desired_preset_name
    st.session_state.setdefault("show_debug", False)
    st.session_state.setdefault("texture_seed", int(np.random.randint(0, 1_000_000_000)))
    st.session_state["texture_seed"] = int(st.session_state.get("seed", st.session_state["texture_seed"]))
    st.session_state.setdefault("submitted_scenario_snapshot", None)
    st.session_state.setdefault("show_exoplanet_presets", False)
    st.session_state.setdefault("exoplanet_random_sample", [])
    st.session_state.setdefault("selected_exoplanet_row", None)
    st.session_state.setdefault("exoplanet_atmosphere_assumption", EXOPLANET_ATMOSPHERE_OPTIONS[0])
    for key in REMOVED_UI_KEYS:
        st.session_state.pop(key, None)


def _apply_preset(name: str):
    preset = PRESETS.get(name)
    if not preset:
        return
    for key, value in preset.items():
        st.session_state[key] = value
    st.session_state["builder_persisted_preset_name"] = name
    st.session_state["builder_persisted_inputs"] = {key: preset.get(key) for key in SCENARIO_ALL_KEYS}
    st.session_state["selected_exoplanet_row"] = None


def _collect_inputs():
    data = {
        "stellar_flux_multiplier": float(st.session_state.stellar_flux_multiplier),
        "enable_seasonality": bool(st.session_state.enable_seasonality),
        "warm_albedo": float(st.session_state.warm_albedo),
        "ice_albedo": float(st.session_state.ice_albedo),
        "initial_co2_ppm": float(st.session_state.initial_co2_ppm),
        "emissions_rate": float(st.session_state.emissions_rate),
        "emissions_growth_mode": str(st.session_state.emissions_growth_mode),
        "mitigation_start_year": int(st.session_state.mitigation_start_year),
        "mitigation_strength": float(st.session_state.mitigation_strength),
        "habitability_profile": str(st.session_state.habitability_profile),
        "habitable_temp_min_c": float(st.session_state.habitable_temp_min_c),
        "habitable_temp_max_c": float(st.session_state.habitable_temp_max_c),
        "K_CO2": float(st.session_state.get("K_CO2", DEFAULT_K_CO2)),
        "seed": int(st.session_state.get("seed", 42)),
        "natural_planet_mode": bool(st.session_state.get("natural_planet_mode", False)),
        "atmosphere_assumption": st.session_state.get("atmosphere_assumption"),
        "imported_equilibrium_temperature_k": st.session_state.get("imported_equilibrium_temperature_k"),
        "imported_stellar_flux_multiplier": st.session_state.get("imported_stellar_flux_multiplier"),
        "import_classification": st.session_state.get("import_classification"),
    }
    if data["seed"] < SEED_MIN or data["seed"] > SEED_MAX:
        data["seed"] = 42
    return data


def _estimate_state(p):
    scenario = scenario_from_flat_params(p, preset_name=str(st.session_state.get("preset_name", "")))
    return preview_state(scenario, horizon_years=PREVIEW_HORIZON_YEARS)


def _build_preset_validation_rows():
    rows = []
    for preset_name, preset_inputs in PRESETS.items():
        d = _estimate_state(preset_inputs)
        rows.append(
            {
                "preset": preset_name,
                "temp_k": round(d["temperature_k"], 2),
                "projected_co2_y100": round(d["projected_co2_ppm"], 2),
                "habitable_pct": round(d["habitable_surface_pct"], 2),
                "system_state": d["system_state"],
                "tipping_risk": d["tipping_label"],
            }
        )
    return rows


def _run_sanity_checks():
    derived = {name: _estimate_state(preset) for name, preset in PRESETS.items()}
    baseline_inputs = PRESETS["Earth-like Baseline"]
    carefree_inputs = PRESETS["Carefree Civilization"]
    stabilization_inputs = PRESETS["Stabilization Policy"]
    baseline = derived["Earth-like Baseline"]
    carefree = derived["Carefree Civilization"]
    stabilization = derived["Stabilization Policy"]
    snowball = derived["Snowball-Prone World"]
    runaway = derived["Runaway-Prone World"]

    earthlike_same_initial_state = all(
        baseline_inputs[key] == carefree_inputs[key] == stabilization_inputs[key]
        for key in EARTHLIKE_SHARED_INITIAL_KEYS
    )
    earthlike_policy_only_diff = (
        set(baseline_inputs.keys()) == set(carefree_inputs.keys()) == set(stabilization_inputs.keys())
        and all(
            key in EARTHLIKE_POLICY_KEYS or (
                baseline_inputs[key] == carefree_inputs[key] == stabilization_inputs[key]
            )
            for key in baseline_inputs
        )
    )

    checks = {
        "earthlike_same_initial_state": earthlike_same_initial_state,
        "earthlike_differ_only_by_policy": earthlike_policy_only_diff,
        "snowball_temp_below_baseline": snowball["temperature_k"] < baseline["temperature_k"],
        "runaway_temp_above_baseline": runaway["temperature_k"] > baseline["temperature_k"],
        "carefree_co2_above_baseline": carefree["projected_co2_ppm"] > baseline["projected_co2_ppm"],
        "stabilization_co2_below_carefree": stabilization["projected_co2_ppm"] < carefree["projected_co2_ppm"],
        "baseline_habitability_plausible": 40.0 <= baseline["habitable_surface_pct"] <= 90.0,
        "snowball_habitability_below_baseline": snowball["habitable_surface_pct"] < baseline["habitable_surface_pct"],
        "runaway_habitability_not_overly_optimistic": runaway["habitable_surface_pct"] <= 85.0,
        "carefree_habitability_not_near_total": carefree["habitable_surface_pct"] <= 90.0,
        "unstable_not_low_tipping": all(
            d["tipping_label"] != "Low" for d in derived.values() if d["system_state"] == "Unstable"
        ),
        "habitable_bounds": all(0.0 <= d["habitable_surface_pct"] <= 100.0 for d in derived.values()),
        "finite_preview_values": all(
            np.isfinite(d["temperature_k"])
            and np.isfinite(d["projected_co2_ppm"])
            and np.isfinite(d["habitable_surface_pct"])
            for d in derived.values()
        ),
    }
    return checks


def _build_export_payload(scenario):
    return export_scenario_json(scenario)


def _build_runtime_payload(inputs, derived, scenario):
    return {
        "scenario": scenario.model_dump(mode="json"),
        "predicted_state": {
            "global_temp_k": derived["temperature_k"],
            "projected_co2_year100_ppm": derived["projected_co2_ppm"],
            "habitable_surface_pct": derived["habitable_surface_pct"],
            "system_state": derived["system_state"],
            "tipping_risk": derived["tipping_label"],
        },
        "initial_planet_parameters": {
            "radius": DEFAULT_PLANET_RADIUS_KM,
            "temperature_c": derived["T0_c"],
            "co2_ppm": derived["co2_ppm"],
            "albedo": derived["effective_albedo"],
            "stellar_energy_w_m2": derived["stellar_flux_w_m2"],
            "seasonal_change": "Yes" if inputs["enable_seasonality"] else "No",
        },
    }


def _build_debug_payload(inputs, derived, scenario):
    sanity_checks = _run_sanity_checks()
    return {
        "runtime_payload": _build_runtime_payload(inputs, derived, scenario),
        "internal_model_constants": INTERNAL_MODEL_CONSTANTS,
        "selected_input_snapshot": inputs,
        "sanity_checks": sanity_checks,
        "preset_validation_table": _build_preset_validation_rows(),
        "internal_scores": {
            "snowball_score": derived["snowball_score"],
            "runaway_score": derived["runaway_score"],
            "stability_index": derived["stability_index"],
            "net_emissions": derived["net_emissions"],
        },
    }


def render_scenario_builder_page():
    _initialize_state()
    st.markdown("<style>[data-testid='stHeaderActionElements']{display:none;}</style>", unsafe_allow_html=True)

    st.markdown("<h1 style='margin:0;'>Scenario Builder</h1>", unsafe_allow_html=True)
    st.caption("Define a planetary climate-biosphere-civilization scenario and launch the digital twin.")

    st.markdown("**Preset Library**")
    selected_preset_name = str(st.session_state.get("preset_name", "Earth-like Baseline"))
    show_exoplanet_presets = bool(st.session_state.get("show_exoplanet_presets", False))
    if _is_exoplanet_preset_name(selected_preset_name):
        show_exoplanet_presets = True
        st.session_state["show_exoplanet_presets"] = True

    if show_exoplanet_presets and not st.session_state.get("exoplanet_random_sample"):
        _refresh_exoplanet_sample()

    exoplanet_option_map = _build_exoplanet_option_map() if show_exoplanet_presets else {}
    preset_options = list(PRESETS.keys()) + list(exoplanet_option_map.keys())
    if selected_preset_name not in preset_options:
        st.session_state["preset_name"] = "Earth-like Baseline"

    preset_cols = st.columns([1.8, 1.0, 1.0])
    preset_name = preset_cols[0].selectbox(
        "Preset Library",
        preset_options,
        key="preset_name",
        label_visibility="collapsed",
    )
    if preset_cols[1].button("Load Preset", type="secondary", use_container_width=True):
        if preset_name in PRESETS:
            _apply_preset(preset_name)
            st.rerun()
        elif preset_name in exoplanet_option_map:
            assumption = str(st.session_state.get("exoplanet_atmosphere_assumption", EXOPLANET_ATMOSPHERE_OPTIONS[0]))
            _apply_exoplanet_preset(preset_name, exoplanet_option_map[preset_name], assumption)
            st.rerun()
        else:
            st.warning("Selected preset is unavailable. Use Show More and try Random 100 again.")
    if preset_cols[2].button("Reset to Default", type="secondary", use_container_width=True):
        st.session_state["force_preset_name_once"] = "Earth-like Baseline"
        _apply_preset("Earth-like Baseline")
        st.rerun()

    extra_cols = st.columns([1.0, 1.0, 2.8])
    if not show_exoplanet_presets:
        if extra_cols[0].button("Show More", type="secondary", use_container_width=True):
            st.session_state["show_exoplanet_presets"] = True
            _refresh_exoplanet_sample()
            st.rerun()
    else:
        if extra_cols[0].button("Hide Extras", type="secondary", use_container_width=True):
            st.session_state["show_exoplanet_presets"] = False
            if _is_exoplanet_preset_name(str(st.session_state.get("preset_name", ""))):
                _apply_preset("Earth-like Baseline")
            st.rerun()
        if extra_cols[1].button("Random 100", type="secondary", use_container_width=True):
            _refresh_exoplanet_sample()
            st.rerun()
        extra_cols[2].caption(
            f"Showing {len(exoplanet_option_map)} random planets from NASA Exoplanet Archive (PS table)."
        )
        st.selectbox(
            "Atmosphere Assumption",
            EXOPLANET_ATMOSPHERE_OPTIONS,
            key="exoplanet_atmosphere_assumption",
            help="Used when applying an imported exoplanet preset.",
        )

    selected_exoplanet = st.session_state.get("selected_exoplanet_row")
    if _is_exoplanet_preset_name(str(st.session_state.get("preset_name", ""))) and isinstance(selected_exoplanet, dict):
        pl_name = str(selected_exoplanet.get("pl_name", "Unknown")).strip() or "Unknown"
        host = str(selected_exoplanet.get("hostname", "")).strip()
        host_text = f" @ {host}" if host else ""
        insol = _safe_float(selected_exoplanet.get("pl_insol"))
        eqt_k = _safe_float(selected_exoplanet.get("pl_eqt"))
        eqt_c = eqt_k - 273.15 if np.isfinite(eqt_k) else np.nan
        ecc = _safe_float(selected_exoplanet.get("pl_orbeccen"))
        insol_text = f"{insol:.2f} S_earth" if np.isfinite(insol) else "S_earth n/a"
        eqt_text = f"{eqt_k:.0f} K ({eqt_c:.1f} C)" if np.isfinite(eqt_k) else "Teq n/a"
        ecc_text = f"{ecc:.3f}" if np.isfinite(ecc) else "n/a"
        st.caption(
            "Loaded from NASA Exoplanet Archive: "
            f"{pl_name}{host_text} | Insolation: {insol_text} | Equilibrium Temp: {eqt_text} | "
            f"Eccentricity: {ecc_text} | Atmosphere: {st.session_state.get('atmosphere_assumption') or st.session_state.get('exoplanet_atmosphere_assumption')}"
        )

    current_inputs = _collect_inputs()
    scenario = scenario_from_flat_params(
        current_inputs,
        preset_name=str(st.session_state.get("preset_name", "Earth-like Baseline")),
    )
    normalized_inputs = scenario_to_flat_params(scenario)
    for key in SCENARIO_ALL_KEYS:
        if key in normalized_inputs:
            st.session_state[key] = normalized_inputs[key]
    current_inputs = _collect_inputs()
    st.session_state["builder_persisted_inputs"] = dict(current_inputs)
    st.session_state["builder_persisted_preset_name"] = str(st.session_state.get("preset_name", "Earth-like Baseline"))
    derived = _estimate_state(current_inputs)

    st.subheader("Current Predicted State")
    st.caption("Estimated from the current parameter configuration before full simulation.")
    kpi_cols = st.columns(5)
    temp_regime = _kpi_temperature_regime(derived["temperature_k"])
    co2_level = _kpi_status_co2(derived["co2_ppm"])
    habitability_level = _kpi_habitability_level(derived["habitable_surface_pct"])
    habitability_color = {"Low": "#B52A2A", "Moderate": "#D2A106", "High": "#2E8B57"}[habitability_level]
    with kpi_cols[0]:
        _kpi_card(
            "Global Temp",
            f"{derived['temperature_c']:.1f} \u00b0C",
            temp_regime,
            _status_color(temp_regime),
        )
    with kpi_cols[1]:
        _kpi_card(
            "CO\u2082",
            f"{derived['co2_ppm']:.0f} ppm",
            co2_level,
        )
    with kpi_cols[2]:
        _kpi_card(
            "Habitable Surface",
            f"{derived['habitable_surface_pct']:.0f}%",
            habitability_level,
            habitability_color,
        )
    with kpi_cols[3]:
        _kpi_card("System State", derived["system_state"], derived["system_state"])
    with kpi_cols[4]:
        _kpi_card("Tipping Risk", derived["tipping_label"], derived["tipping_label"])

    left, right = st.columns([1.15, 1.0])

    with left:
        _section_header("Star & Orbit", first=True)
        st.number_input(
            "Stellar Flux Multiplier",
            min_value=STELLAR_FLUX_INPUT_MIN,
            max_value=STELLAR_FLUX_INPUT_MAX,
            step=0.01,
            key="stellar_flux_multiplier",
            help="Scales incoming stellar radiation from the host star.",
        )
        st.toggle("Enable Seasonality", key="enable_seasonality")

        _section_header("Climate")
        st.slider("Warm Albedo", min_value=0.00, max_value=1.00, step=0.01, key="warm_albedo")
        st.slider("Ice Albedo", min_value=0.00, max_value=1.00, step=0.01, key="ice_albedo")

        _section_header("Atmosphere & Carbon")
        st.number_input("Initial CO\u2082 (ppm)", min_value=CO2_INPUT_MIN, max_value=CO2_INPUT_MAX, step=10.0, key="initial_co2_ppm")

        _section_header("Civilization")
        st.slider("Emissions Rate", min_value=EMISSIONS_INPUT_MIN, max_value=EMISSIONS_INPUT_MAX, step=0.05, key="emissions_rate")
        st.selectbox(
            "Emissions Growth Mode",
            list(EMISSIONS_MODES),
            key="emissions_growth_mode",
        )
        st.number_input("Mitigation Start Year", min_value=MITIGATION_START_MIN, max_value=MITIGATION_START_MAX, step=5, key="mitigation_start_year")
        st.slider("Mitigation Strength", min_value=0.00, max_value=1.00, step=0.01, key="mitigation_strength")

        _section_header("Habitability Definition")
        habit_profile = st.selectbox("Profile", list(HABITABILITY_PROFILES.keys()), key="habitability_profile")
        if st.button("Apply Profile", use_container_width=False):
            tmin, tmax = HABITABILITY_PROFILES[habit_profile]
            st.session_state["habitable_temp_min_c"] = tmin
            st.session_state["habitable_temp_max_c"] = tmax
            st.rerun()
        st.number_input(
            "Habitable Temperature Min (\u00b0C)",
            min_value=HAB_TEMP_MIN,
            max_value=HAB_TEMP_MAX,
            step=1.0,
            key="habitable_temp_min_c",
        )
        st.number_input(
            "Habitable Temperature Max (\u00b0C)",
            min_value=HAB_TEMP_MIN,
            max_value=HAB_TEMP_MAX,
            step=1.0,
            key="habitable_temp_max_c",
        )

    with right:
        with st.container(border=True):
            st.subheader("3D Planet Preview")
            fig = draw_planet(
                temp_c=derived["temperature_c"],
                co2_ppm=derived["co2_ppm"],
                albedo=derived["effective_albedo"],
                stellar_flux=derived["stellar_flux_w_m2"],
                radius_km=DEFAULT_PLANET_RADIUS_KM,
            )
            _render_spinning_surface(fig, component_key="scenario_builder_planet", height_px=300)

        with st.container(border=True):
            st.subheader("Scenario Summary")
            st.markdown(
                "\n".join(
                    [
                        f"- Star flux: {derived['stellar_flux_w_m2'] / SOLAR_CONSTANT:.2f} S\u2295",
                        f"- Initial CO\u2082: {current_inputs['initial_co2_ppm']:.0f} ppm",
                        f"- Projected CO\u2082 (year 100): {derived['projected_co2_ppm']:.0f} ppm",
                        f"- Emissions mode: {current_inputs['emissions_growth_mode']}",
                        f"- Warm albedo: {current_inputs['warm_albedo']:.2f}",
                        f"- Ice albedo: {current_inputs['ice_albedo']:.2f}",
                        f"- Seasonality: {'On' if current_inputs['enable_seasonality'] else 'Off'}",
                        f"- Mitigation start: year {current_inputs['mitigation_start_year']}",
                        f"- Mitigation strength: {current_inputs['mitigation_strength']:.2f}",
                        f"- Habitability profile: {current_inputs['habitability_profile']}",
                        (
                            f"- Habitable range: {current_inputs['habitable_temp_min_c']:.0f}\u00b0C "
                            f"to {current_inputs['habitable_temp_max_c']:.0f}\u00b0C"
                        ),
                    ]
                )
            )

        with st.container(border=True):
            st.subheader("Quick Risk")
            _badge("Snowball Risk", _risk_label(derived["snowball_score"]))
            _badge("Runaway Risk", _risk_label(derived["runaway_score"]))
            _badge("Stability Outlook", derived["stability_outlook"])

        with st.container(border=True):
            export_payload = _build_export_payload(scenario)
            runtime_payload = _build_runtime_payload(current_inputs, derived, scenario)
            debug_payload = _build_debug_payload(current_inputs, derived, scenario)

            run_now = st.button("Run Simulation", type="primary", use_container_width=True)

            action_cols = st.columns(3)
            save_pressed = action_cols[0].button("Save Scenario", type="secondary", use_container_width=True)
            action_cols[1].download_button(
                "Export JSON",
                data=export_payload,
                file_name="scenario_v2.json",
                mime="application/json",
                use_container_width=True,
                type="secondary",
            )
            if action_cols[2].button("Show Debug", type="secondary", use_container_width=True):
                st.session_state["show_debug"] = not st.session_state.get("show_debug", False)

            if run_now:
                submitted_seed = int(current_inputs.get("seed", st.session_state.get("texture_seed", 42)))
                st.session_state["submitted"] = True
                st.session_state["submitted_texture_seed"] = submitted_seed
                st.session_state["seed"] = submitted_seed
                st.session_state["submitted_scenario_snapshot"] = scenario.model_dump(mode="json")
                st.session_state["params"] = runtime_payload
                if hasattr(st, "switch_page"):
                    st.switch_page("pages/01_Climate_Twin.py")
                else:
                    st.rerun()

            if save_pressed:
                saved = st.session_state.setdefault("saved_scenarios", [])
                saved.append(export_payload)
                st.info(f"Scenario saved ({len(saved)} total).")

            if st.session_state.get("show_debug", False):
                with st.expander("Debug / Advanced", expanded=False):
                    st.json(
                        {
                            "runtime_payload": debug_payload["runtime_payload"],
                            "internal_model_constants": debug_payload["internal_model_constants"],
                            "selected_input_snapshot": debug_payload["selected_input_snapshot"],
                            "internal_scores": debug_payload["internal_scores"],
                        }
                    )
                    st.markdown("**Preset Validation Table**")
                    st.dataframe(debug_payload["preset_validation_table"], use_container_width=True)
                    st.markdown("**Sanity Checks**")
                    checks_df = [
                        {"check": name, "pass": bool(value)} for name, value in debug_payload["sanity_checks"].items()
                    ]
                    st.dataframe(checks_df, use_container_width=True)

    if current_inputs["habitable_temp_max_c"] <= current_inputs["habitable_temp_min_c"]:
        st.warning("Habitability bounds are invalid: max temperature must be greater than min temperature.")


render_scenario_builder_page()
