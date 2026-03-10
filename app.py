import json
from math import log

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Scenario Builder", layout="wide")

SOLAR_CONSTANT = 1361.0
SIGMA = 5.670374419e-8

PRESETS = {
    "Earth-like Baseline": {
        "planet_radius_km": 6371.0,
        "stellar_flux_multiplier": 1.00,
        "orbital_distance_au": 1.00,
        "seasonality_amplitude": 0.12,
        "obliquity_deg": 23.4,
        "seasonality_enabled": True,
        "warm_albedo": 0.30,
        "ice_albedo": 0.62,
        "ice_transition_temp_c": -10.0,
        "heat_transport_strength": 1.00,
        "climate_heat_capacity": 1.00,
        "initial_co2_ppm": 420.0,
        "volcanic_outgassing": 0.35,
        "weathering_strength": 0.55,
        "biosphere_uptake_strength": 0.50,
        "carbon_removal_sensitivity": 0.70,
        "emissions_rate": 2.5,
        "emissions_growth_mode": "Constant",
        "mitigation_start_year": 40,
        "mitigation_strength": 0.15,
        "carbon_capture_rate": 0.05,
        "habitable_temp_min_c": 0.0,
        "habitable_temp_max_c": 45.0,
    },
    "Carefree Civilization": {
        "planet_radius_km": 6371.0,
        "stellar_flux_multiplier": 1.03,
        "orbital_distance_au": 1.00,
        "seasonality_amplitude": 0.10,
        "obliquity_deg": 24.0,
        "seasonality_enabled": True,
        "warm_albedo": 0.28,
        "ice_albedo": 0.60,
        "ice_transition_temp_c": -12.0,
        "heat_transport_strength": 0.95,
        "climate_heat_capacity": 0.95,
        "initial_co2_ppm": 460.0,
        "volcanic_outgassing": 0.40,
        "weathering_strength": 0.40,
        "biosphere_uptake_strength": 0.35,
        "carbon_removal_sensitivity": 0.45,
        "emissions_rate": 4.8,
        "emissions_growth_mode": "Carefree",
        "mitigation_start_year": 120,
        "mitigation_strength": 0.02,
        "carbon_capture_rate": 0.00,
        "habitable_temp_min_c": 0.0,
        "habitable_temp_max_c": 45.0,
    },
    "Stabilization Policy": {
        "planet_radius_km": 6371.0,
        "stellar_flux_multiplier": 1.00,
        "orbital_distance_au": 1.00,
        "seasonality_amplitude": 0.12,
        "obliquity_deg": 23.4,
        "seasonality_enabled": True,
        "warm_albedo": 0.30,
        "ice_albedo": 0.62,
        "ice_transition_temp_c": -10.0,
        "heat_transport_strength": 1.05,
        "climate_heat_capacity": 1.00,
        "initial_co2_ppm": 420.0,
        "volcanic_outgassing": 0.35,
        "weathering_strength": 0.65,
        "biosphere_uptake_strength": 0.60,
        "carbon_removal_sensitivity": 0.85,
        "emissions_rate": 2.5,
        "emissions_growth_mode": "Stabilization",
        "mitigation_start_year": 10,
        "mitigation_strength": 0.70,
        "carbon_capture_rate": 0.40,
        "habitable_temp_min_c": 0.0,
        "habitable_temp_max_c": 45.0,
    },
    "Snowball-Prone World": {
        "planet_radius_km": 6200.0,
        "stellar_flux_multiplier": 0.86,
        "orbital_distance_au": 1.05,
        "seasonality_amplitude": 0.08,
        "obliquity_deg": 18.0,
        "seasonality_enabled": True,
        "warm_albedo": 0.34,
        "ice_albedo": 0.78,
        "ice_transition_temp_c": -4.0,
        "heat_transport_strength": 0.75,
        "climate_heat_capacity": 0.90,
        "initial_co2_ppm": 260.0,
        "volcanic_outgassing": 0.25,
        "weathering_strength": 0.60,
        "biosphere_uptake_strength": 0.42,
        "carbon_removal_sensitivity": 0.82,
        "emissions_rate": 1.0,
        "emissions_growth_mode": "Constant",
        "mitigation_start_year": 40,
        "mitigation_strength": 0.25,
        "carbon_capture_rate": 0.05,
        "habitable_temp_min_c": -5.0,
        "habitable_temp_max_c": 35.0,
    },
    "Runaway-Prone World": {
        "planet_radius_km": 6900.0,
        "stellar_flux_multiplier": 1.12,
        "orbital_distance_au": 0.95,
        "seasonality_amplitude": 0.16,
        "obliquity_deg": 27.0,
        "seasonality_enabled": True,
        "warm_albedo": 0.23,
        "ice_albedo": 0.52,
        "ice_transition_temp_c": -16.0,
        "heat_transport_strength": 1.15,
        "climate_heat_capacity": 1.10,
        "initial_co2_ppm": 700.0,
        "volcanic_outgassing": 0.45,
        "weathering_strength": 0.35,
        "biosphere_uptake_strength": 0.25,
        "carbon_removal_sensitivity": 0.35,
        "emissions_rate": 5.0,
        "emissions_growth_mode": "Growing",
        "mitigation_start_year": 120,
        "mitigation_strength": 0.08,
        "carbon_capture_rate": 0.02,
        "habitable_temp_min_c": 0.0,
        "habitable_temp_max_c": 50.0,
    },
}

HABITABILITY_PROFILES = {
    "Liquid Water": (0.0, 45.0),
    "Complex Life": (5.0, 32.0),
    "Broad Tolerance": (-10.0, 55.0),
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
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            bgcolor="rgb(10, 14, 24)",
            camera=dict(eye=dict(x=1.6 + 0.25 * radius_scale, y=1.3 + 0.2 * radius_scale, z=1.0)),
        ),
    )
    return fig


def _risk_label(score: float):
    if score < 0.25:
        return "Low"
    if score < 0.50:
        return "Moderate"
    if score < 0.75:
        return "Elevated"
    return "High"


def _risk_color(score: float):
    if score < 0.25:
        return "#2E8B57"
    if score < 0.50:
        return "#D2A106"
    if score < 0.75:
        return "#C56A1C"
    return "#B52A2A"


def _badge(label: str, score: float):
    text = _risk_label(score)
    color = _risk_color(score)
    st.markdown(
        (
            "<div style='padding:8px 10px;border-radius:8px;"
            "border:1px solid #2a2a2a;margin-bottom:8px;background:#101523;'>"
            f"<strong>{label}:</strong> <span style='color:{color};font-weight:600;'>{text}</span></div>"
        ),
        unsafe_allow_html=True,
    )


def _initialize_state():
    defaults = PRESETS["Earth-like Baseline"]
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("preset_name", "Earth-like Baseline")
    st.session_state.setdefault("habitability_profile", "Liquid Water")
    st.session_state.setdefault("show_debug", False)
    st.session_state.setdefault("texture_seed", int(np.random.randint(0, 1_000_000_000)))


def _apply_preset(name: str):
    preset = PRESETS.get(name)
    if not preset:
        return
    for key, value in preset.items():
        st.session_state[key] = value


def _collect_inputs():
    return {
        "planet_radius_km": float(st.session_state.planet_radius_km),
        "stellar_flux_multiplier": float(st.session_state.stellar_flux_multiplier),
        "orbital_distance_au": float(st.session_state.orbital_distance_au),
        "seasonality_amplitude": float(st.session_state.seasonality_amplitude),
        "obliquity_deg": float(st.session_state.obliquity_deg),
        "seasonality_enabled": bool(st.session_state.seasonality_enabled),
        "warm_albedo": float(st.session_state.warm_albedo),
        "ice_albedo": float(st.session_state.ice_albedo),
        "ice_transition_temp_c": float(st.session_state.ice_transition_temp_c),
        "heat_transport_strength": float(st.session_state.heat_transport_strength),
        "climate_heat_capacity": float(st.session_state.climate_heat_capacity),
        "initial_co2_ppm": float(st.session_state.initial_co2_ppm),
        "volcanic_outgassing": float(st.session_state.volcanic_outgassing),
        "weathering_strength": float(st.session_state.weathering_strength),
        "biosphere_uptake_strength": float(st.session_state.biosphere_uptake_strength),
        "carbon_removal_sensitivity": float(st.session_state.carbon_removal_sensitivity),
        "emissions_rate": float(st.session_state.emissions_rate),
        "emissions_growth_mode": str(st.session_state.emissions_growth_mode),
        "mitigation_start_year": int(st.session_state.mitigation_start_year),
        "mitigation_strength": float(st.session_state.mitigation_strength),
        "carbon_capture_rate": float(st.session_state.carbon_capture_rate),
        "habitable_temp_min_c": float(st.session_state.habitable_temp_min_c),
        "habitable_temp_max_c": float(st.session_state.habitable_temp_max_c),
    }


def _estimate_state(p):
    mode_multiplier = {
        "Constant": 1.00,
        "Growing": 1.25,
        "Carefree": 1.55,
        "Stabilization": 0.85,
    }
    emissions_mode = mode_multiplier.get(p["emissions_growth_mode"], 1.0)

    policy_pressure = (
        0.75 * p["mitigation_strength"]
        + 0.55 * p["carbon_capture_rate"]
        + 0.25 * p["carbon_removal_sensitivity"]
    )
    mitigation_timing = np.clip((120 - p["mitigation_start_year"]) / 120, 0.0, 1.0)
    net_emissions = (
        p["emissions_rate"] * emissions_mode * (1.0 - 0.6 * policy_pressure * mitigation_timing)
        + 1.8 * p["volcanic_outgassing"]
        - 1.5 * p["weathering_strength"]
        - 1.4 * p["biosphere_uptake_strength"]
    )

    projected_co2 = np.clip(p["initial_co2_ppm"] + 55.0 * net_emissions, 120.0, 2400.0)
    stellar_flux = SOLAR_CONSTANT * p["stellar_flux_multiplier"] / max(0.2, p["orbital_distance_au"] ** 2)

    temp_c = 10.0
    ice_fraction = 0.0
    effective_albedo = p["warm_albedo"]

    for _ in range(8):
        ice_fraction = 1.0 / (1.0 + np.exp((temp_c - p["ice_transition_temp_c"]) / 2.8))
        effective_albedo = np.clip(
            (1.0 - ice_fraction) * p["warm_albedo"] + ice_fraction * p["ice_albedo"], 0.05, 0.92
        )
        forcing = 5.35 * log(max(projected_co2, 1.0) / 280.0)
        raw_temp_k = ((stellar_flux * (1.0 - effective_albedo) + forcing) / (4.0 * SIGMA)) ** 0.25
        raw_temp_c = raw_temp_k - 273.15
        inertia = np.clip(0.55 + 0.18 * p["climate_heat_capacity"], 0.45, 0.9)
        temp_c = inertia * temp_c + (1.0 - inertia) * raw_temp_c

    latitudes = np.linspace(-90.0, 90.0, 241)
    gradient = 38.0 / np.clip(p["heat_transport_strength"], 0.4, 2.2)
    baseline_lat_temp = temp_c - gradient * np.abs(latitudes) / 90.0
    season_boost = 0.0
    if p["seasonality_enabled"]:
        season_boost = 16.0 * p["seasonality_amplitude"] * (1.0 + p["obliquity_deg"] / 90.0)
    effective_lat_temp = baseline_lat_temp - 0.35 * season_boost
    habitable_mask = (effective_lat_temp >= p["habitable_temp_min_c"]) & (
        effective_lat_temp <= p["habitable_temp_max_c"]
    )
    habitable_surface_pct = float(100.0 * habitable_mask.mean())

    snowball_score = float(
        np.clip(
            0.65 * ice_fraction
            + 0.25 * np.clip((p["ice_albedo"] - p["warm_albedo"] - 0.18) / 0.35, 0.0, 1.0)
            + 0.30 * np.clip((p["ice_transition_temp_c"] - temp_c) / 30.0, 0.0, 1.0),
            0.0,
            1.0,
        )
    )
    runaway_score = float(
        np.clip(
            0.45 * np.clip((temp_c - 28.0) / 34.0, 0.0, 1.0)
            + 0.35 * np.clip((projected_co2 - 650.0) / 1200.0, 0.0, 1.0)
            + 0.20 * np.clip((stellar_flux - SOLAR_CONSTANT) / 500.0, 0.0, 1.0),
            0.0,
            1.0,
        )
    )

    stability_index = (
        1.1 * p["weathering_strength"]
        + 1.0 * p["biosphere_uptake_strength"]
        + 0.9 * p["carbon_removal_sensitivity"]
        + 0.8 * p["mitigation_strength"]
        + 0.6 * p["carbon_capture_rate"]
        + 0.3 * p["heat_transport_strength"]
        - 0.35 * net_emissions
        - 0.25 * max(0.0, temp_c - 30.0) / 10.0
        - 0.2 * max(0.0, -10.0 - temp_c) / 10.0
    )

    if stability_index >= 2.3:
        system_state = "Stable"
    elif stability_index >= 1.7:
        system_state = "Marginally Stable"
    elif stability_index >= 1.1:
        system_state = "Uncertain"
    else:
        system_state = "Unstable"

    tipping_score = max(snowball_score, runaway_score)
    tipping_label = _risk_label(tipping_score)

    return {
        "stellar_flux_w_m2": float(stellar_flux),
        "effective_albedo": float(effective_albedo),
        "projected_co2_ppm": float(projected_co2),
        "temperature_c": float(temp_c),
        "temperature_k": float(temp_c + 273.15),
        "habitable_surface_pct": habitable_surface_pct,
        "ice_fraction": float(ice_fraction),
        "snowball_score": snowball_score,
        "runaway_score": runaway_score,
        "stability_index": float(stability_index),
        "system_state": system_state,
        "tipping_label": tipping_label,
        "net_emissions": float(net_emissions),
    }


def _build_payload(inputs, derived):
    return {
        "scenario_builder": {
            "star_orbit": {
                "stellar_flux_multiplier": inputs["stellar_flux_multiplier"],
                "orbital_distance_au": inputs["orbital_distance_au"],
                "seasonality_enabled": inputs["seasonality_enabled"],
                "seasonality_amplitude": inputs["seasonality_amplitude"],
                "obliquity_deg": inputs["obliquity_deg"],
            },
            "climate": {
                "warm_albedo": inputs["warm_albedo"],
                "ice_albedo": inputs["ice_albedo"],
                "ice_transition_temp_c": inputs["ice_transition_temp_c"],
                "heat_transport_strength": inputs["heat_transport_strength"],
                "climate_heat_capacity": inputs["climate_heat_capacity"],
            },
            "atmosphere_carbon": {
                "initial_co2_ppm": inputs["initial_co2_ppm"],
                "volcanic_outgassing": inputs["volcanic_outgassing"],
                "weathering_strength": inputs["weathering_strength"],
                "biosphere_uptake_strength": inputs["biosphere_uptake_strength"],
                "carbon_removal_sensitivity": inputs["carbon_removal_sensitivity"],
            },
            "civilization": {
                "emissions_rate": inputs["emissions_rate"],
                "emissions_growth_mode": inputs["emissions_growth_mode"],
                "mitigation_start_year": inputs["mitigation_start_year"],
                "mitigation_strength": inputs["mitigation_strength"],
                "carbon_capture_rate": inputs["carbon_capture_rate"],
            },
            "habitability_definition": {
                "habitable_temp_min_c": inputs["habitable_temp_min_c"],
                "habitable_temp_max_c": inputs["habitable_temp_max_c"],
            },
        },
        "initial_planet_parameters": {
            "radius": inputs["planet_radius_km"],
            "temperature_c": derived["temperature_c"],
            "co2_ppm": derived["projected_co2_ppm"],
            "albedo": derived["effective_albedo"],
            "stellar_energy_w_m2": derived["stellar_flux_w_m2"],
            "seasonal_change": "Yes" if inputs["seasonality_enabled"] else "No",
        },
        "kpi": {
            "global_temp_k": derived["temperature_k"],
            "co2_ppm": derived["projected_co2_ppm"],
            "habitable_surface_pct": derived["habitable_surface_pct"],
            "system_state": derived["system_state"],
            "tipping_risk": derived["tipping_label"],
        },
    }


_initialize_state()

st.title("Scenario Builder")
st.caption("Define a planetary climate-biosphere-civilization scenario and launch the digital twin.")

nav_cols = st.columns(4)
nav_cols[0].button("Scenario Builder", disabled=True, use_container_width=True)
if nav_cols[1].button("Climate Twin", use_container_width=True):
    st.switch_page("pages/01_Climate_Twin.py")
if nav_cols[2].button("Refugia Map", use_container_width=True):
    st.switch_page("pages/02_Refugia_Map.py")
if nav_cols[3].button("Stability & Policy", use_container_width=True):
    st.switch_page("pages/03_Stability_and_Policy.py")

preset_cols = st.columns([1.8, 1.0, 1.0])
preset_name = preset_cols[0].selectbox("Preset library", list(PRESETS.keys()), key="preset_name")
if preset_cols[1].button("Load Preset", type="secondary", use_container_width=True):
    _apply_preset(preset_name)
    st.rerun()
if preset_cols[2].button("Reset to Default", use_container_width=True):
    _apply_preset("Earth-like Baseline")
    st.rerun()

current_inputs = _collect_inputs()
derived = _estimate_state(current_inputs)

kpi_cols = st.columns(5)
kpi_cols[0].metric("Global Temp", f"{derived['temperature_k']:.1f} K")
kpi_cols[1].metric("CO2", f"{derived['projected_co2_ppm']:.0f} ppm")
kpi_cols[2].metric("Habitable Surface", f"{derived['habitable_surface_pct']:.0f}%")
kpi_cols[3].metric("System State", derived["system_state"])
kpi_cols[4].metric("Tipping Risk", derived["tipping_label"])

left, right = st.columns([1.15, 1.0])

with left:
    st.subheader("Star & Orbit")
    st.number_input(
        "Stellar Flux Multiplier",
        min_value=0.70,
        max_value=1.30,
        step=0.01,
        key="stellar_flux_multiplier",
        help="Controls incoming stellar energy. Small changes may trigger major climate regime shifts.",
    )
    st.number_input("Orbital Distance (AU)", min_value=0.40, max_value=2.00, step=0.01, key="orbital_distance_au")
    st.toggle("Enable Seasonality", key="seasonality_enabled")
    st.slider("Seasonality Amplitude", min_value=0.00, max_value=0.30, step=0.01, key="seasonality_amplitude")
    with st.expander("Advanced orbit settings"):
        st.slider("Obliquity / Seasonal Tilt (deg)", min_value=0.0, max_value=45.0, step=0.5, key="obliquity_deg")
        st.number_input("Planet Radius (km)", min_value=1500.0, max_value=20000.0, step=50.0, key="planet_radius_km")

    st.subheader("Climate")
    st.slider("Warm Albedo", min_value=0.10, max_value=0.60, step=0.01, key="warm_albedo")
    st.slider("Ice Albedo", min_value=0.30, max_value=0.90, step=0.01, key="ice_albedo")
    st.slider(
        "Ice Transition Temperature (degC)",
        min_value=-40.0,
        max_value=10.0,
        step=0.5,
        key="ice_transition_temp_c",
    )
    st.slider("Heat Transport Strength", min_value=0.40, max_value=2.20, step=0.05, key="heat_transport_strength")
    st.slider("Climate Heat Capacity", min_value=0.40, max_value=2.50, step=0.05, key="climate_heat_capacity")

    st.subheader("Atmosphere & Carbon")
    st.number_input("Initial CO2 (ppm)", min_value=100.0, max_value=3000.0, step=10.0, key="initial_co2_ppm")
    st.slider("Volcanic Outgassing", min_value=0.00, max_value=1.50, step=0.01, key="volcanic_outgassing")
    st.slider("Weathering Strength", min_value=0.00, max_value=1.50, step=0.01, key="weathering_strength")
    st.slider("Biosphere Uptake Strength", min_value=0.00, max_value=1.50, step=0.01, key="biosphere_uptake_strength")
    st.slider(
        "Carbon Removal Sensitivity", min_value=0.00, max_value=1.50, step=0.01, key="carbon_removal_sensitivity"
    )

    st.subheader("Civilization")
    st.slider("Emissions Rate", min_value=0.00, max_value=8.00, step=0.05, key="emissions_rate")
    st.selectbox(
        "Emissions Growth Mode",
        ["Constant", "Growing", "Carefree", "Stabilization"],
        key="emissions_growth_mode",
    )
    st.number_input("Mitigation Start Year", min_value=0, max_value=500, step=5, key="mitigation_start_year")
    st.slider("Mitigation Strength", min_value=0.00, max_value=1.00, step=0.01, key="mitigation_strength")
    st.slider("Carbon Capture Rate", min_value=0.00, max_value=1.00, step=0.01, key="carbon_capture_rate")

    st.subheader("Habitability Definition")
    habit_profile = st.selectbox("Profile", list(HABITABILITY_PROFILES.keys()), key="habitability_profile")
    if st.button("Load Habitability Profile", use_container_width=False):
        tmin, tmax = HABITABILITY_PROFILES[habit_profile]
        st.session_state["habitable_temp_min_c"] = tmin
        st.session_state["habitable_temp_max_c"] = tmax
        st.rerun()
    st.number_input("Habitable Temperature Min (degC)", min_value=-80.0, max_value=80.0, step=1.0, key="habitable_temp_min_c")
    st.number_input("Habitable Temperature Max (degC)", min_value=-80.0, max_value=100.0, step=1.0, key="habitable_temp_max_c")

with right:
    st.subheader("3D Planet Preview")
    fig = draw_planet(
        temp_c=derived["temperature_c"],
        co2_ppm=derived["projected_co2_ppm"],
        albedo=derived["effective_albedo"],
        stellar_flux=derived["stellar_flux_w_m2"],
        radius_km=current_inputs["planet_radius_km"],
        seed=int(st.session_state.get("texture_seed", 0)),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scenario Summary")
    st.markdown(
        "\n".join(
            [
                f"**Star flux:** {derived['stellar_flux_w_m2'] / SOLAR_CONSTANT:.2f} S_earth",
                f"**Initial CO2:** {current_inputs['initial_co2_ppm']:.0f} ppm",
                f"**Projected CO2:** {derived['projected_co2_ppm']:.0f} ppm",
                f"**Emissions mode:** {current_inputs['emissions_growth_mode']}",
                f"**Albedo regime:** warm {current_inputs['warm_albedo']:.2f} / ice {current_inputs['ice_albedo']:.2f}",
                f"**Seasonality:** {'On' if current_inputs['seasonality_enabled'] else 'Off'}",
                f"**Policy strength:** {current_inputs['mitigation_strength']:.2f}",
            ]
        )
    )

    st.subheader("Quick Risk")
    _badge("Snowball Risk", derived["snowball_score"])
    _badge("Runaway Risk", derived["runaway_score"])
    stability_score = float(np.clip((2.6 - derived["stability_index"]) / 2.6, 0.0, 1.0))
    _badge("Stability Outlook", stability_score)

    run_now = st.button("Run Simulation", type="primary", use_container_width=True)

    action_cols = st.columns(3)
    save_pressed = action_cols[0].button("Save Scenario", type="secondary", use_container_width=True)
    payload_for_actions = _build_payload(current_inputs, derived)
    action_cols[1].download_button(
        "Export JSON",
        data=json.dumps(payload_for_actions, indent=2),
        file_name="scenario_builder_payload.json",
        mime="application/json",
        use_container_width=True,
    )
    if action_cols[2].button("Show Debug", use_container_width=True):
        st.session_state["show_debug"] = not st.session_state.get("show_debug", False)

    if run_now:
        st.session_state["texture_seed"] = int(np.random.randint(0, 1_000_000_000))
        st.session_state["submitted"] = True
        st.session_state["params"] = payload_for_actions
        st.success("Simulation inputs captured.")

    if save_pressed:
        saved = st.session_state.setdefault("saved_scenarios", [])
        saved.append(payload_for_actions)
        st.info(f"Scenario saved ({len(saved)} total).")

    if st.session_state.get("show_debug", False):
        with st.expander("Debug / Advanced", expanded=False):
            st.json(payload_for_actions)

if current_inputs["habitable_temp_max_c"] <= current_inputs["habitable_temp_min_c"]:
    st.warning("Habitability bounds are invalid: max temperature must be greater than min temperature.")