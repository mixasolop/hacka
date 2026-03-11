import json
from math import log

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Scenario Builder", layout="wide")

SOLAR_CONSTANT = 1361.0
CO2_BASELINE_PPM = 280.0
PREVIEW_HORIZON_YEARS = 100
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
DEFAULT_PLANET_RADIUS_KM = 6371.0

# Reduced climate sensitivity coefficients for preview-only estimator.
K_FLUX = 55.0
K_ALBEDO = 85.0
K_CO2 = 4.5
WEATHERING_COEFF = 0.94 * DEFAULT_WEATHERING_STRENGTH
WEATHERING_TEMP_SENS = 0.045
BIOSPHERE_COEFF = 0.94 * DEFAULT_BIOSPHERE_UPTAKE_STRENGTH
EMISSIONS_TO_PPM_PER_YEAR = 0.8
LATITUDE_BASE_CONTRAST_C = 24.0
PLANET_RIGHT_TILT_DEG = 23.5
PLANET_INITIAL_VIEW_LON_DEG = 38.0
PLANET_SPIN_SPEED_DEG_PER_SEC = 6.0

INTERNAL_MODEL_CONSTANTS = {
    "co2_baseline_ppm": CO2_BASELINE_PPM,
    "preview_horizon_years": PREVIEW_HORIZON_YEARS,
    "reference_temp_c": REFERENCE_TEMP_C,
    "albedo_ref": ALBEDO_REF,
    "seasonality_amplitude": DEFAULT_SEASONALITY_AMPLITUDE,
    "ice_transition_temp_c": DEFAULT_ICE_TRANSITION_TEMP_C,
    "ice_transition_width_c": ICE_TRANSITION_WIDTH_C,
    "natural_outgassing_ppm_per_year": NATURAL_OUTGASSING,
    "weathering_strength": DEFAULT_WEATHERING_STRENGTH,
    "biosphere_strength": DEFAULT_BIOSPHERE_UPTAKE_STRENGTH,
    "biosphere_opt_temp_c": BIOSPHERE_OPT_TEMP_C,
    "biosphere_temp_width_c": BIOSPHERE_TEMP_WIDTH_C,
    "k_flux": K_FLUX,
    "k_albedo": K_ALBEDO,
    "k_co2": K_CO2,
    "weathering_coeff": WEATHERING_COEFF,
    "weathering_temp_sens": WEATHERING_TEMP_SENS,
    "biosphere_coeff": BIOSPHERE_COEFF,
    "emissions_to_ppm_per_year": EMISSIONS_TO_PPM_PER_YEAR,
    "latitude_base_contrast_c": LATITUDE_BASE_CONTRAST_C,
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

EMISSIONS_MODE_MULTIPLIER = {
    "Constant": 1.00,
    "Growing": 1.25,
    "Carefree": 1.55,
    "Stabilization": 0.85,
    "Aggressive Mitigation": 0.65,
}

EARTHLIKE_SHARED_INITIAL_KEYS = (
    "stellar_flux_multiplier",
    "enable_seasonality",
    "warm_albedo",
    "ice_albedo",
    "initial_co2_ppm",
    "habitability_profile",
    "habitable_temp_min_c",
    "habitable_temp_max_c",
)
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

PRESETS = {
    "Earth-like Baseline": {
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
    },
    "Carefree Civilization": {
        "stellar_flux_multiplier": 1.00,
        "enable_seasonality": True,
        "warm_albedo": 0.30,
        "ice_albedo": 0.62,
        "initial_co2_ppm": 420.0,
        "emissions_rate": 4.8,
        "emissions_growth_mode": "Carefree",
        "mitigation_start_year": 120,
        "mitigation_strength": 0.02,
        "habitability_profile": "Liquid Water",
        "habitable_temp_min_c": 0.0,
        "habitable_temp_max_c": 45.0,
    },
    "Stabilization Policy": {
        "stellar_flux_multiplier": 1.00,
        "enable_seasonality": True,
        "warm_albedo": 0.30,
        "ice_albedo": 0.62,
        "initial_co2_ppm": 420.0,
        "emissions_rate": 2.0,
        "emissions_growth_mode": "Stabilization",
        "mitigation_start_year": 5,
        "mitigation_strength": 0.78,
        "habitability_profile": "Liquid Water",
        "habitable_temp_min_c": 0.0,
        "habitable_temp_max_c": 45.0,
    },
    "Snowball-Prone World": {
        "stellar_flux_multiplier": 0.86,
        "enable_seasonality": True,
        "warm_albedo": 0.34,
        "ice_albedo": 0.78,
        "initial_co2_ppm": 260.0,
        "emissions_rate": 1.0,
        "emissions_growth_mode": "Constant",
        "mitigation_start_year": 40,
        "mitigation_strength": 0.25,
        "habitability_profile": "Conservative Biosphere",
        "habitable_temp_min_c": -5.0,
        "habitable_temp_max_c": 35.0,
    },
    "Runaway-Prone World": {
        "stellar_flux_multiplier": 1.12,
        "enable_seasonality": True,
        "warm_albedo": 0.23,
        "ice_albedo": 0.52,
        "initial_co2_ppm": 700.0,
        "emissions_rate": 5.0,
        "emissions_growth_mode": "Growing",
        "mitigation_start_year": 120,
        "mitigation_strength": 0.08,
        "habitability_profile": "Broad Microbial Tolerance",
        "habitable_temp_min_c": 0.0,
        "habitable_temp_max_c": 50.0,
    },
}

HABITABILITY_PROFILES = {
    "Liquid Water": (0.0, 45.0),
    "Conservative Biosphere": (5.0, 32.0),
    "Broad Microbial Tolerance": (-10.0, 55.0),
}


def _planet_surface(lon, lat, seed: int, temp_c: float):
    rng = np.random.default_rng(seed)
    n = np.zeros_like(lon, dtype=float)
    for _ in range(9):
        k1 = rng.integers(1, 7)
        k2 = rng.integers(1, 7)
        p1 = rng.uniform(0, 2 * np.pi)
        p2 = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.05, 0.22)
        n += amp * np.sin(k1 * lon + p1) * np.cos(k2 * lat + p2)

    n += 0.20 * np.sin(2 * lon + rng.uniform(0, 2 * np.pi))
    n -= 0.16 * np.cos(3 * lat + rng.uniform(0, 2 * np.pi))
    n = (n - n.min()) / (n.max() - n.min() + 1e-9)

    sea_level = 0.54
    land = n > sea_level
    land_height = np.clip((n - sea_level) / (1 - sea_level + 1e-9), 0, 1)
    ocean_depth = np.clip((sea_level - n) / (sea_level + 1e-9), 0, 1)

    ridges = np.abs(np.sin(9 * lon + rng.uniform(0, 2 * np.pi)) * np.cos(11 * lat))
    ridges *= np.abs(np.sin(15 * lon + rng.uniform(0, 2 * np.pi)))
    ridges = (ridges - ridges.min()) / (ridges.max() - ridges.min() + 1e-9)
    mountain = np.clip(0.65 * land_height + 0.55 * ridges, 0, 1)

    abs_lat = np.abs(lat) / (np.pi / 2)
    ice_threshold = np.clip(0.72 + 0.004 * temp_c, 0.56, 0.90)
    ice = abs_lat > ice_threshold

    surf = np.where(land, 0.50 + 0.32 * land_height, 0.06 + 0.26 * (1 - ocean_depth))
    surf = np.where(land & (mountain > 0.72), 0.84 + 0.10 * mountain, surf)
    surf = np.where(ice, 0.97, surf)
    surf = np.clip(surf, 0, 1)

    relief = np.where(land, 0.02 + 0.05 * mountain, -0.004 * ocean_depth)
    relief = np.where(ice, relief + 0.012, relief)
    return surf, relief


def draw_planet(temp_c: float, co2_ppm: float, albedo: float, stellar_flux: float, radius_km: float, seed: int):
    heat = max(0.0, min(1.0, (temp_c + 30.0) / 80.0))
    co2_factor = max(0.0, min(1.0, co2_ppm / 1400.0))
    light = max(0.2, min(1.3, stellar_flux / SOLAR_CONSTANT))

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    radius_scale = np.clip(radius_km / 6371.0, 0.35, 2.2)
    x0 = np.outer(np.cos(u), np.sin(v))
    y0 = np.outer(np.sin(u), np.sin(v))
    z0 = np.outer(np.ones_like(u), np.cos(v))
    lon = np.arctan2(y0, x0)
    lat = np.arcsin(np.clip(z0, -1, 1))

    base_r = 0.17 + 0.76 * heat
    base_g = 0.58 - 0.26 * heat + 0.08 * (1.0 - co2_factor)
    base_b = 0.88 - 0.67 * heat
    brightness = (0.72 + 0.35 * (1.0 - albedo + 0.2)) * light
    color = np.clip(np.array([base_r, base_g, base_b]) * brightness, 0, 1)

    texture, relief = _planet_surface(lon, lat, seed=seed, temp_c=temp_c)
    rfield = radius_scale * (1.0 + relief)
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

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=texture,
                colorscale=colorscale,
                showscale=False,
                lighting=dict(ambient=0.45, diffuse=0.8, specular=0.3, roughness=0.85),
                lightposition=dict(x=120, y=80, z=200),
            )
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
  let spinning = true;
  const speedDegPerSec = {PLANET_SPIN_SPEED_DEG_PER_SEC:.3f};
  const toRad = Math.PI / 180.0;
  const initialEye = plot.layout?.scene?.camera?.eye ?? {{x: 1.8, y: 1.4, z: 1.0}};
  const initialUp = plot.layout?.scene?.camera?.up ?? {{x: 0.0, y: 0.0, z: 1.0}};
  const radius = Math.max(0.5, Math.hypot(initialEye.x ?? 1.8, initialEye.y ?? 1.4));
  const z = Number.isFinite(initialEye.z) ? initialEye.z : 1.0;
  let angleDeg = (Math.atan2(initialEye.y ?? 1.4, initialEye.x ?? 1.8) / toRad + 360.0) % 360.0;
  let lastTs = performance.now();
  let internalUpdate = false;
  let pending = false;
  const stop = () => {{ spinning = false; }};
  plot.addEventListener('pointerdown', stop, {{ passive: true }});
  plot.addEventListener('wheel', stop, {{ passive: true }});
  plot.addEventListener('touchstart', stop, {{ passive: true }});
  plot.on('plotly_click', stop);
  plot.on('plotly_doubleclick', stop);
  plot.on('plotly_relayouting', () => {{ if (!internalUpdate) stop(); }});
  plot.on('plotly_relayout', (ev) => {{
    if (internalUpdate) return;
    if (!ev) return;
    if (Object.prototype.hasOwnProperty.call(ev, 'scene.camera') ||
        Object.prototype.hasOwnProperty.call(ev, 'scene.camera.eye') ||
        Object.prototype.hasOwnProperty.call(ev, 'scene.camera.up')) {{
      stop();
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
    if (spinning) {{
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

    for key in SCENARIO_INPUT_KEYS:
        st.session_state.setdefault(key, persisted_inputs.get(key, defaults[key]))

    if isinstance(forced_preset_name, str) and forced_preset_name in PRESETS:
        desired_preset_name = forced_preset_name
        st.session_state["preset_name"] = desired_preset_name
    else:
        current_widget_preset = st.session_state.get("preset_name")
        if isinstance(current_widget_preset, str) and current_widget_preset in PRESETS:
            desired_preset_name = current_widget_preset
        else:
            desired_preset_name = str(st.session_state.get("builder_persisted_preset_name", "Earth-like Baseline"))
            if desired_preset_name not in PRESETS:
                desired_preset_name = "Earth-like Baseline"
            st.session_state["preset_name"] = desired_preset_name
    st.session_state["builder_persisted_preset_name"] = desired_preset_name
    st.session_state.setdefault("show_debug", False)
    st.session_state.setdefault("texture_seed", int(np.random.randint(0, 1_000_000_000)))
    st.session_state.setdefault("submitted_scenario_snapshot", None)
    for key in REMOVED_UI_KEYS:
        st.session_state.pop(key, None)


def _apply_preset(name: str):
    preset = PRESETS.get(name)
    if not preset:
        return
    for key, value in preset.items():
        st.session_state[key] = value
    st.session_state["builder_persisted_preset_name"] = name
    st.session_state["builder_persisted_inputs"] = {key: preset[key] for key in SCENARIO_INPUT_KEYS}


def _collect_inputs():
    return {
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
    }


def _effective_ice_weight(temp_c: float):
    exponent = np.clip((temp_c - DEFAULT_ICE_TRANSITION_TEMP_C) / ICE_TRANSITION_WIDTH_C, -40.0, 40.0)
    return float(1.0 / (1.0 + np.exp(exponent)))


def _effective_albedo(temp_c: float, warm_albedo: float, ice_albedo: float):
    w_ice = _effective_ice_weight(temp_c)
    alpha = warm_albedo * (1.0 - w_ice) + ice_albedo * w_ice
    return float(np.clip(alpha, 0.0, 1.0))


def _seasonality_amplitude(enable_seasonality: bool):
    return DEFAULT_SEASONALITY_AMPLITUDE if enable_seasonality else 0.0


def _temperature_from_co2(stellar_flux_multiplier: float, warm_albedo: float, ice_albedo: float, co2_ppm: float):
    temp_c = REFERENCE_TEMP_C
    safe_co2 = max(float(co2_ppm), 1.0)
    for _ in range(3):
        alpha_eff = _effective_albedo(temp_c, warm_albedo, ice_albedo)
        temp_c = (
            REFERENCE_TEMP_C
            + K_FLUX * (stellar_flux_multiplier - 1.0)
            - K_ALBEDO * (alpha_eff - ALBEDO_REF)
            + K_CO2 * log(safe_co2 / CO2_BASELINE_PPM)
        )
    return float(temp_c), float(_effective_albedo(temp_c, warm_albedo, ice_albedo))


def _human_emissions_rate(mode: str, emissions_rate: float, year: int, mitigation_start_year: int, mitigation_strength: float):
    if mode == "Growing":
        base = emissions_rate * (1.0 + 0.010 * year)
    elif mode == "Carefree":
        base = emissions_rate * (1.0 + 0.020 * year)
    elif mode == "Stabilization":
        if year < mitigation_start_year:
            base = emissions_rate
        else:
            decay = 0.030 * max(0.1, mitigation_strength) * (year - mitigation_start_year)
            base = emissions_rate * np.exp(-decay)
    elif mode == "Aggressive Mitigation":
        if year < mitigation_start_year:
            base = emissions_rate
        else:
            decay = 0.060 * max(0.2, mitigation_strength) * (year - mitigation_start_year)
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
    gaussian = -((temp_c - BIOSPHERE_OPT_TEMP_C) ** 2) / (2.0 * BIOSPHERE_TEMP_WIDTH_C ** 2)
    temp_factor = np.exp(np.clip(gaussian, -20.0, 0.0))
    return float(BIOSPHERE_COEFF * (co2_ppm / CO2_BASELINE_PPM) * temp_factor)


def _integrate_projected_co2(p):
    co2_ppm = max(float(p["initial_co2_ppm"]), 1.0)
    total_human = 0.0

    for year in range(PREVIEW_HORIZON_YEARS):
        temp_c, _ = _temperature_from_co2(
            stellar_flux_multiplier=p["stellar_flux_multiplier"],
            warm_albedo=p["warm_albedo"],
            ice_albedo=p["ice_albedo"],
            co2_ppm=co2_ppm,
        )
        e_human = _human_emissions_rate(
            mode=p["emissions_growth_mode"],
            emissions_rate=p["emissions_rate"],
            year=year,
            mitigation_start_year=p["mitigation_start_year"],
            mitigation_strength=p["mitigation_strength"],
        )
        weathering = _weathering_sink(temp_c, co2_ppm)
        biosphere = _biosphere_sink(temp_c, co2_ppm)
        dco2_dt = e_human + NATURAL_OUTGASSING - weathering - biosphere
        co2_ppm = max(1.0, co2_ppm + dco2_dt)
        total_human += e_human

    avg_human_emissions = total_human / PREVIEW_HORIZON_YEARS
    return float(co2_ppm), float(avg_human_emissions)


def _habitability_percent(temp_global_c: float, projected_co2_ppm: float, p):
    lat_deg = np.linspace(-88.75, 88.75, 72)
    lat_rad = np.deg2rad(lat_deg)

    # Latitudinal preview field with asymmetric structure:
    # equator can overheat in warm climates while poles stay cooler.
    hot_anomaly = max(0.0, temp_global_c - 22.0)
    cold_anomaly = max(0.0, 8.0 - temp_global_c)
    albedo_contrast = max(0.0, p["ice_albedo"] - p["warm_albedo"])

    equatorial_heat_excess = 5.0 + 1.25 * hot_anomaly - 0.10 * cold_anomaly
    polar_cooling_strength = 22.0 + 0.70 * cold_anomaly + 0.90 * hot_anomaly + 8.0 * max(0.0, albedo_contrast - 0.20)
    if p["enable_seasonality"]:
        equatorial_heat_excess += 1.0
        polar_cooling_strength += 2.5

    equatorial_heat_excess = float(np.clip(equatorial_heat_excess, 2.5, 22.0))
    polar_cooling_strength = float(np.clip(polar_cooling_strength, 12.0, 46.0))
    lat_temp = (
        temp_global_c
        + equatorial_heat_excess * (np.cos(lat_rad) ** 2)
        - polar_cooling_strength * (np.sin(lat_rad) ** 2)
    )

    weights = np.cos(lat_rad)
    habitable = (lat_temp >= p["habitable_temp_min_c"]) & (lat_temp <= p["habitable_temp_max_c"])
    habitable_frac = 100.0 * float(np.sum(weights * habitable.astype(float)) / np.sum(weights))

    # Secondary correction to avoid optimistic near-total habitability in hot/high-CO2 states.
    overheat = np.clip((temp_global_c - 30.0) / 14.0, 0.0, 1.0)
    greenhouse_stress = np.clip((projected_co2_ppm - 600.0) / 1000.0, 0.0, 1.0)
    spread_stress = np.clip((equatorial_heat_excess + polar_cooling_strength - 42.0) / 22.0, 0.0, 1.0)
    habitable_frac *= 1.0 - np.clip(0.25 * overheat + 0.20 * greenhouse_stress + 0.12 * spread_stress, 0.0, 0.50)
    return float(np.clip(habitable_frac, 0.0, 100.0))


def _estimate_state(p):
    projected_co2_ppm, avg_human_emissions = _integrate_projected_co2(p)

    temp_c, effective_albedo = _temperature_from_co2(
        stellar_flux_multiplier=p["stellar_flux_multiplier"],
        warm_albedo=p["warm_albedo"],
        ice_albedo=p["ice_albedo"],
        co2_ppm=projected_co2_ppm,
    )
    temperature_k = temp_c + 273.15
    habitable_surface_pct = _habitability_percent(temp_c, projected_co2_ppm, p)
    ice_weight = _effective_ice_weight(temp_c)
    albedo_contrast = max(0.0, p["ice_albedo"] - p["warm_albedo"])

    flux_cold = np.clip((1.0 - p["stellar_flux_multiplier"]) / 0.18, 0.0, 1.0)
    snowball_score = float(
        np.clip(
            0.55 * ice_weight
            + 0.25 * np.clip((albedo_contrast - 0.15) / 0.45, 0.0, 1.0)
            + 0.20 * flux_cold,
            0.0,
            1.0,
        )
    )

    runaway_score = float(
        np.clip(
            0.40 * np.clip((temp_c - 28.0) / 12.0, 0.0, 1.0)
            + 0.25 * np.clip((projected_co2_ppm - 700.0) / 900.0, 0.0, 1.0)
            + 0.20 * np.clip((p["stellar_flux_multiplier"] - 1.02) / 0.18, 0.0, 1.0)
            + 0.15 * np.clip(avg_human_emissions / 6.0, 0.0, 1.0),
            0.0,
            1.0,
        )
    )

    mitigation_effect = p["mitigation_strength"] * np.clip(
        (PREVIEW_HORIZON_YEARS - p["mitigation_start_year"]) / PREVIEW_HORIZON_YEARS,
        0.0,
        1.0,
    )
    co2_pressure = max(0.0, log(max(projected_co2_ppm, 1.0) / CO2_BASELINE_PPM))
    hab_center = 0.5 * (p["habitable_temp_min_c"] + p["habitable_temp_max_c"])
    hab_half_width = max(1.0, 0.5 * (p["habitable_temp_max_c"] - p["habitable_temp_min_c"]))
    hab_distance = abs(temp_c - hab_center) / hab_half_width

    stability_score = (
        0.06 * abs(temp_c - REFERENCE_TEMP_C)
        + 0.90 * co2_pressure
        + 0.12 * avg_human_emissions
        + 0.80 * albedo_contrast
        + 0.50 * max(0.0, hab_distance - 1.0)
        + 0.70 * max(0.0, snowball_score - 0.60)
        + 0.70 * max(0.0, runaway_score - 0.60)
        - 0.90 * mitigation_effect
    )
    stability_score = float(max(0.0, stability_score))

    if stability_score < 1.00:
        system_state = "Stable"
    elif stability_score < 2.00:
        system_state = "Marginal"
    else:
        system_state = "Unstable"

    tipping_label = {"Stable": "Low", "Marginal": "Elevated", "Unstable": "High"}[system_state]
    tipping_score = max(snowball_score, runaway_score)
    if tipping_score > 0.72:
        tipping_label = "High"
    elif tipping_score > 0.45 and tipping_label == "Low":
        tipping_label = "Elevated"

    if system_state == "Unstable" and tipping_label == "Low":
        tipping_label = "High"

    return {
        "stellar_flux_w_m2": float(SOLAR_CONSTANT * p["stellar_flux_multiplier"]),
        "effective_albedo": float(effective_albedo),
        "projected_co2_ppm": float(projected_co2_ppm),
        "temperature_c": float(temp_c),
        "temperature_k": float(temperature_k),
        "habitable_surface_pct": float(np.clip(habitable_surface_pct, 0.0, 100.0)),
        "ice_fraction": float(ice_weight),
        "snowball_score": snowball_score,
        "runaway_score": runaway_score,
        "stability_index": stability_score,
        "stability_outlook": system_state,
        "system_state": system_state,
        "tipping_label": tipping_label,
        "net_emissions": float(avg_human_emissions + NATURAL_OUTGASSING - _weathering_sink(temp_c, projected_co2_ppm) - _biosphere_sink(temp_c, projected_co2_ppm)),
    }


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


def _build_export_payload(inputs, derived):
    return {
        "scenario": inputs,
        "predicted_state": {
            "global_temp_k": derived["temperature_k"],
            "projected_co2_year100_ppm": derived["projected_co2_ppm"],
            "habitable_surface_pct": derived["habitable_surface_pct"],
            "system_state": derived["system_state"],
            "tipping_risk": derived["tipping_label"],
        },
        "metadata": {
            "app": "Habitat Tipping Points",
            "screen": "Scenario Builder",
            "preview_type": "lightweight_estimate",
        },
    }


def _build_runtime_payload(inputs, derived):
    return {
        "scenario": inputs,
        "predicted_state": {
            "global_temp_k": derived["temperature_k"],
            "projected_co2_year100_ppm": derived["projected_co2_ppm"],
            "habitable_surface_pct": derived["habitable_surface_pct"],
            "system_state": derived["system_state"],
            "tipping_risk": derived["tipping_label"],
        },
        "initial_planet_parameters": {
            "radius": DEFAULT_PLANET_RADIUS_KM,
            "temperature_c": derived["temperature_c"],
            "co2_ppm": derived["projected_co2_ppm"],
            "albedo": derived["effective_albedo"],
            "stellar_energy_w_m2": derived["stellar_flux_w_m2"],
            "seasonal_change": "Yes" if inputs["enable_seasonality"] else "No",
        },
    }


def _build_debug_payload(inputs, derived):
    sanity_checks = _run_sanity_checks()
    return {
        "runtime_payload": _build_runtime_payload(inputs, derived),
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
    preset_cols = st.columns([1.8, 1.0, 1.0])
    preset_name = preset_cols[0].selectbox(
        "Preset Library",
        list(PRESETS.keys()),
        key="preset_name",
        label_visibility="collapsed",
    )
    if preset_cols[1].button("Load Preset", type="secondary", use_container_width=True):
        _apply_preset(preset_name)
        st.rerun()
    if preset_cols[2].button("Reset to Default", type="secondary", use_container_width=True):
        st.session_state["force_preset_name_once"] = "Earth-like Baseline"
        _apply_preset("Earth-like Baseline")
        st.rerun()

    current_inputs = _collect_inputs()
    st.session_state["builder_persisted_inputs"] = dict(current_inputs)
    st.session_state["builder_persisted_preset_name"] = str(st.session_state.get("preset_name", "Earth-like Baseline"))
    derived = _estimate_state(current_inputs)

    st.subheader("Current Predicted State")
    st.caption("Estimated from the current parameter configuration before full simulation.")
    kpi_cols = st.columns(5)
    temp_regime = _kpi_temperature_regime(derived["temperature_k"])
    co2_level = _kpi_status_co2(derived["projected_co2_ppm"])
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
            f"{derived['projected_co2_ppm']:.0f} ppm",
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
            min_value=0.70,
            max_value=1.30,
            step=0.01,
            key="stellar_flux_multiplier",
            help="Scales incoming stellar radiation from the host star.",
        )
        st.toggle("Enable Seasonality", key="enable_seasonality")

        _section_header("Climate")
        st.slider("Warm Albedo", min_value=0.10, max_value=0.60, step=0.01, key="warm_albedo")
        st.slider("Ice Albedo", min_value=0.30, max_value=0.90, step=0.01, key="ice_albedo")

        _section_header("Atmosphere & Carbon")
        st.number_input("Initial CO\u2082 (ppm)", min_value=100.0, max_value=3000.0, step=10.0, key="initial_co2_ppm")

        _section_header("Civilization")
        st.slider("Emissions Rate", min_value=0.00, max_value=8.00, step=0.05, key="emissions_rate")
        st.selectbox(
            "Emissions Growth Mode",
            ["Constant", "Growing", "Carefree", "Stabilization", "Aggressive Mitigation"],
            key="emissions_growth_mode",
        )
        st.number_input("Mitigation Start Year", min_value=0, max_value=500, step=5, key="mitigation_start_year")
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
            min_value=-80.0,
            max_value=80.0,
            step=1.0,
            key="habitable_temp_min_c",
        )
        st.number_input(
            "Habitable Temperature Max (\u00b0C)",
            min_value=-80.0,
            max_value=100.0,
            step=1.0,
            key="habitable_temp_max_c",
        )

    with right:
        with st.container(border=True):
            st.subheader("3D Planet Preview")
            fig = draw_planet(
                temp_c=derived["temperature_c"],
                co2_ppm=derived["projected_co2_ppm"],
                albedo=derived["effective_albedo"],
                stellar_flux=derived["stellar_flux_w_m2"],
                radius_km=DEFAULT_PLANET_RADIUS_KM,
                seed=int(st.session_state.get("texture_seed", 0)),
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
            export_payload = _build_export_payload(current_inputs, derived)
            runtime_payload = _build_runtime_payload(current_inputs, derived)
            debug_payload = _build_debug_payload(current_inputs, derived)

            run_now = st.button("Run Simulation", type="primary", use_container_width=True)

            action_cols = st.columns(3)
            save_pressed = action_cols[0].button("Save Scenario", type="secondary", use_container_width=True)
            action_cols[1].download_button(
                "Export JSON",
                data=json.dumps(export_payload, indent=2),
                file_name="scenario_builder_payload.json",
                mime="application/json",
                use_container_width=True,
                type="secondary",
            )
            if action_cols[2].button("Show Debug", type="secondary", use_container_width=True):
                st.session_state["show_debug"] = not st.session_state.get("show_debug", False)

            if run_now:
                submitted_seed = int(st.session_state.get("texture_seed", int(np.random.randint(0, 1_000_000_000))))
                st.session_state["submitted"] = True
                st.session_state["submitted_texture_seed"] = submitted_seed
                st.session_state["submitted_scenario_snapshot"] = dict(current_inputs)
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
