import time
from math import log

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Climate Twin", layout="wide")

SIM_YEARS = 500
SIM_DT_YEARS = 1.0
TEMPERATURE_RESPONSE_YEARS = 12.0

SOLAR_CONSTANT = 1361.0
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

HABITABILITY_PROFILES = {
    "Liquid Water": (0.0, 45.0),
    "Conservative Biosphere": (5.0, 32.0),
    "Broad Microbial Tolerance": (-10.0, 55.0),
}

EMISSIONS_MODES = ("Constant", "Growing", "Carefree", "Stabilization", "Aggressive Mitigation")
YEAR_KEY = "climate_twin_year"
PLAY_KEY = "climate_twin_playing"
HABITABILITY_LAT_BANDS = 181
HABITABILITY_SOFT_MARGIN_C = 1.6
TEMP_HOT_THRESHOLD_C = 32.0
TEMP_COLD_THRESHOLD_C = -5.0
CO2_DANGER_THRESHOLD_PPM = 1000.0
HABITABILITY_THRESHOLDS = (75, 50, 25)


def _status_color(label: str):
    normalized = str(label).strip().lower()
    if normalized in {"stable", "low", "high habitability"}:
        return "#2E8B57"
    if normalized in {"marginal", "elevated", "moderate"}:
        return "#D2A106"
    if normalized in {"cold", "stable cooling"}:
        return "#3B82C4"
    return "#B52A2A"


def _kpi_card(label: str, value: str, status: str):
    color = _status_color(status)
    st.markdown(
        (
            "<div style='padding:10px 12px;border-radius:10px;border:1px solid #273249;"
            "background:#0f1626;margin-bottom:8px;'>"
            f"<div style='font-size:11px;letter-spacing:0.04em;text-transform:uppercase;color:#96a3bd;'>{label}</div>"
            f"<div style='font-size:26px;font-weight:700;line-height:1.2;color:#f4f8ff;margin-top:4px;'>{value}</div>"
            f"<div style='margin-top:6px;font-size:12px;font-weight:600;color:{color};'>{status}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


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
    params["habitability_profile"] = (
        str(params["habitability_profile"])
        if str(params["habitability_profile"]) in HABITABILITY_PROFILES
        else "Liquid Water"
    )
    params["habitable_temp_min_c"] = float(params["habitable_temp_min_c"])
    params["habitable_temp_max_c"] = float(params["habitable_temp_max_c"])
    if params["habitable_temp_max_c"] <= params["habitable_temp_min_c"]:
        params["habitable_temp_max_c"] = params["habitable_temp_min_c"] + 1.0
    return params


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
    return float(temp_c), float(_effective_albedo(temp_c, warm_albedo, ice_albedo))


def _human_emissions_rate(mode: str, emissions_rate: float, year: int, mitigation_start_year: int, mitigation_strength: float):
    if mode == "Growing":
        base = emissions_rate * (1.0 + 0.010 * year)
    elif mode == "Carefree":
        # Carefree keeps growing, but long-run growth softens to avoid unrealistic explosions.
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


def _soft_step(x):
    safe_x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-safe_x))


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


def _habitability_percent(temp_global_c: float, projected_co2_ppm: float, p):
    lat_deg = np.linspace(-89.5, 89.5, HABITABILITY_LAT_BANDS)
    lat_rad = np.deg2rad(lat_deg)

    hot_anomaly = max(0.0, temp_global_c - 22.0)
    cold_anomaly = max(0.0, 8.0 - temp_global_c)
    albedo_contrast = max(0.0, p["ice_albedo"] - p["warm_albedo"])
    seasonality = DEFAULT_SEASONALITY_AMPLITUDE if p["enable_seasonality"] else 0.0

    equatorial_heat_excess = 4.8 + 1.05 * hot_anomaly - 0.08 * cold_anomaly + 6.0 * seasonality
    polar_cooling_strength = (
        20.0 + 0.55 * cold_anomaly + 0.65 * hot_anomaly + 6.0 * max(0.0, albedo_contrast - 0.20) + 12.0 * seasonality
    )
    equatorial_heat_excess = float(np.clip(equatorial_heat_excess, 2.5, 18.0))
    polar_cooling_strength = float(np.clip(polar_cooling_strength, 12.0, 42.0))

    lat_temp = (
        temp_global_c
        + equatorial_heat_excess * (np.cos(lat_rad) ** 2)
        - polar_cooling_strength * (np.sin(lat_rad) ** 2)
    )
    weights = np.cos(lat_rad)
    thermal_spread = equatorial_heat_excess + polar_cooling_strength
    soft_margin_c = (
        HABITABILITY_SOFT_MARGIN_C
        + 0.05 * np.clip(thermal_spread - 30.0, 0.0, 24.0)
        + (0.45 if p["enable_seasonality"] else 0.0)
    )
    lower_ok = _soft_step((lat_temp - p["habitable_temp_min_c"]) / soft_margin_c)
    upper_ok = _soft_step((p["habitable_temp_max_c"] - lat_temp) / soft_margin_c)
    local_habitability = lower_ok * upper_ok
    habitable_frac = 100.0 * float(np.sum(weights * local_habitability) / np.sum(weights))

    overheat = np.clip((temp_global_c - 30.0) / 14.0, 0.0, 1.0)
    cold_stress = np.clip((2.0 - temp_global_c) / 16.0, 0.0, 1.0)
    greenhouse_stress = np.clip((projected_co2_ppm - 650.0) / 1000.0, 0.0, 1.0)
    spread_stress = np.clip((thermal_spread - 48.0) / 26.0, 0.0, 1.0)
    comfort_stress = np.clip((abs(temp_global_c - REFERENCE_TEMP_C) - 2.0) / 30.0, 0.0, 1.0)
    stress = np.clip(
        0.23 * overheat + 0.14 * cold_stress + 0.18 * greenhouse_stress + 0.10 * spread_stress + 0.18 * comfort_stress,
        0.0,
        0.50,
    )
    habitable_frac *= 1.0 - stress
    return float(np.clip(habitable_frac, 0.0, 100.0))


def _regime_label(
    temp_c: float,
    co2_ppm: float,
    habitable_pct: float,
    params,
    dtemp: float,
    dco2: float,
    dhabitable: float,
    year: int,
):
    hot_stress = max(0.0, (temp_c - 18.0) / 9.0)
    cold_stress = max(0.0, (8.0 - temp_c) / 10.0)
    co2_pressure = max(0.0, log(max(co2_ppm, 1.0) / CO2_BASELINE_PPM))
    high_co2_penalty = max(0.0, (co2_ppm - 700.0) / 500.0)
    habitability_stress = max(0.0, (75.0 - habitable_pct) / 28.0)

    base_score = 0.85 * co2_pressure + 1.00 * hot_stress + 0.45 * cold_stress + 0.48 * habitability_stress + 0.40 * high_co2_penalty

    emissions_pressure = {
        "Aggressive Mitigation": -0.35,
        "Stabilization": -0.18,
        "Constant": 0.08,
        "Growing": 0.28,
        "Carefree": 0.38,
    }.get(params["emissions_growth_mode"], 0.10)

    years_until_mitigation = max(0, int(params["mitigation_start_year"]) - int(year))
    delay_pressure = 0.22 * np.clip(years_until_mitigation / 120.0, 0.0, 1.0)
    weakness_pressure = 0.14 * (1.0 - float(params["mitigation_strength"])) * (1.0 if years_until_mitigation > 0 else 0.4)
    mitigation_pressure = delay_pressure + weakness_pressure

    trend_pressure = (
        3.2 * max(0.0, dtemp)
        + 0.040 * max(0.0, dco2)
        + 0.045 * max(0.0, -dhabitable)
    )

    cold_lock_in = 0.12 * np.clip((5.0 - temp_c) / 15.0, 0.0, 1.0) * np.clip(
        (params["ice_albedo"] - params["warm_albedo"] - 0.20) / 0.40, 0.0, 1.0
    )
    score = base_score + emissions_pressure + mitigation_pressure + trend_pressure + cold_lock_in

    if score < 1.05:
        return "Stable"
    if score < 2.25:
        return "Marginal"
    return "Unstable"


def _event_color(kind: str):
    if kind == "mitigation":
        return "#4F9DFF"
    if kind == "habitability":
        return "#D2A106"
    if kind == "danger":
        return "#B52A2A"
    return "#8899B8"


def _threshold_state(temp_c: float, co2_ppm: float, habitable_pct: float):
    flags = {
        "temp_hot": temp_c >= TEMP_HOT_THRESHOLD_C,
        "temp_cold": temp_c <= TEMP_COLD_THRESHOLD_C,
        "co2_danger": co2_ppm >= CO2_DANGER_THRESHOLD_PPM,
    }
    for threshold in HABITABILITY_THRESHOLDS:
        flags[f"hab_{threshold}"] = habitable_pct < float(threshold)
    return flags


def _event_specifications():
    specs = [
        {
            "id": "temp_hot",
            "kind": "danger",
            "event_label": f"Global temperature exceeds {TEMP_HOT_THRESHOLD_C:.0f}°C",
            "initial_label": f"Initial condition: temperature already above {TEMP_HOT_THRESHOLD_C:.0f}°C",
        },
        {
            "id": "temp_cold",
            "kind": "danger",
            "event_label": f"Global temperature falls below {TEMP_COLD_THRESHOLD_C:.0f}°C",
            "initial_label": f"Initial condition: temperature already below {TEMP_COLD_THRESHOLD_C:.0f}°C",
        },
        {
            "id": "co2_danger",
            "kind": "danger",
            "event_label": f"CO\u2082 exceeds danger threshold ({CO2_DANGER_THRESHOLD_PPM:.0f} ppm)",
            "initial_label": f"Initial condition: CO\u2082 already above {CO2_DANGER_THRESHOLD_PPM:.0f} ppm",
        },
    ]
    for threshold in HABITABILITY_THRESHOLDS:
        specs.append(
            {
                "id": f"hab_{threshold}",
                "kind": "habitability",
                "event_label": f"Habitable surface drops below {threshold}%",
                "initial_label": f"Initial condition: habitability already below {threshold}%",
            }
        )
    return specs


@st.cache_data(show_spinner=False)
def simulate_time_series(params):
    time_years = np.arange(0, SIM_YEARS + 1, dtype=int)
    n = len(time_years)

    temp_c_series = np.zeros(n, dtype=float)
    temp_k_series = np.zeros(n, dtype=float)
    co2_series = np.zeros(n, dtype=float)
    habitable_series = np.zeros(n, dtype=float)
    regime_series = []
    events = []
    initial_conditions = []

    co2_ppm = max(float(params["initial_co2_ppm"]), 1.0)
    temp_c, _ = _equilibrium_temperature_c(
        params["stellar_flux_multiplier"], params["warm_albedo"], params["ice_albedo"], co2_ppm
    )
    event_specs = _event_specifications()
    mitigation_event_added = False
    prev_flags = None

    for i, year in enumerate(time_years):
        temp_eq_c, _ = _equilibrium_temperature_c(
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
        habitable_pct = _habitability_percent(temp_c, co2_ppm, params)
        if i > 0:
            dtemp = temp_c - float(temp_c_series[i - 1])
            dco2 = co2_ppm - float(co2_series[i - 1])
            dhabitable = habitable_pct - float(habitable_series[i - 1])
        else:
            dtemp = 0.0
            dco2 = 0.0
            dhabitable = 0.0
        regime = _regime_label(temp_c, co2_ppm, habitable_pct, params, dtemp, dco2, dhabitable, int(year))

        temp_c_series[i] = temp_c
        temp_k_series[i] = temp_c + 273.15
        co2_series[i] = co2_ppm
        habitable_series[i] = habitable_pct
        regime_series.append(regime)
        current_flags = _threshold_state(temp_c, co2_ppm, habitable_pct)

        if i == 0:
            for spec in event_specs:
                if current_flags[spec["id"]]:
                    initial_conditions.append({"label": spec["initial_label"], "kind": spec["kind"]})
            prev_flags = current_flags.copy()
        else:
            for spec in event_specs:
                is_now_violated = bool(current_flags[spec["id"]])
                was_violated = bool(prev_flags[spec["id"]]) if prev_flags is not None else False
                if is_now_violated and not was_violated:
                    events.append({"year": int(year), "label": spec["event_label"], "kind": spec["kind"]})
            prev_flags = current_flags.copy()

        if (not mitigation_event_added) and year == params["mitigation_start_year"] and year <= SIM_YEARS:
            events.append({"year": int(year), "label": "Mitigation begins", "kind": "mitigation"})
            mitigation_event_added = True

        if i == n - 1:
            continue

        e_human = _human_emissions_rate(
            params["emissions_growth_mode"],
            params["emissions_rate"],
            year,
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

    events = sorted(events, key=lambda item: item["year"])
    return {
        "time_years": time_years,
        "global_temperature_c": temp_c_series,
        "global_temperature_k": temp_k_series,
        "co2_ppm": co2_series,
        "habitable_surface_percent": np.clip(habitable_series, 0.0, 100.0),
        "regime": regime_series,
        "initial_conditions": initial_conditions,
        "events": events,
    }


def _build_chart(x, y, title: str, yaxis_title: str, color: str, current_year: int, events):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=color, width=2.4),
            hovertemplate="Year %{x}<br>%{y:.2f}<extra></extra>",
        )
    )

    for event in events:
        fig.add_vline(
            x=event["year"],
            line=dict(color=_event_color(event["kind"]), width=1.2, dash="dot"),
            opacity=0.55,
        )

    fig.add_vline(
        x=current_year,
        line=dict(color="#F4F8FF", width=2.0, dash="dash"),
        opacity=0.9,
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        showarrow=False,
        align="left",
        text=(
            "Markers: "
            "<span style='color:#F4F8FF;'>current year</span>, "
            "<span style='color:#4F9DFF;'>mitigation</span>, "
            "<span style='color:#D2A106;'>habitability</span>/"
            "<span style='color:#B52A2A;'>danger</span>"
        ),
        bordercolor="rgba(120,140,180,0.22)",
        borderwidth=1,
        borderpad=3,
        bgcolor="rgba(15,22,38,0.70)",
        font=dict(size=10, color="#9fb0cf"),
    )
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=42, b=8),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f1626",
        font=dict(color="#d9e3f5"),
    )
    fig.update_xaxes(title="Year", gridcolor="rgba(110,130,170,0.20)", zeroline=False)
    fig.update_yaxes(title=yaxis_title, gridcolor="rgba(110,130,170,0.20)", zeroline=False)
    return fig


def _explanation_lines(series, idx: int, params):
    lines = []
    temp_c = float(series["global_temperature_c"][idx])
    co2_ppm = float(series["co2_ppm"][idx])
    habitable = float(series["habitable_surface_percent"][idx])
    year = int(series["time_years"][idx])
    regime = str(series["regime"][idx])
    initial_conditions = series.get("initial_conditions", [])

    if idx > 0:
        dco2 = co2_ppm - float(series["co2_ppm"][idx - 1])
        dtemp = temp_c - float(series["global_temperature_c"][idx - 1])
        dhab = habitable - float(series["habitable_surface_percent"][idx - 1])
    else:
        dco2 = 0.0
        dtemp = 0.0
        dhab = 0.0

    if idx == 0 and initial_conditions:
        lines.append(initial_conditions[0]["label"])

    if year == 0 and regime == "Unstable":
        lines.append("The scenario starts in an already dangerous regime, not a future transition.")
    elif year == 0 and regime == "Marginal":
        lines.append("The world starts near climate limits and is sensitive to further forcing.")

    if dco2 > 0.9:
        lines.append("Atmospheric CO\u2082 is increasing, strengthening greenhouse forcing.")
    elif dco2 < -0.1:
        lines.append("Atmospheric CO\u2082 is declining as sinks and mitigation counter emissions.")
    else:
        lines.append("CO\u2082 growth is relatively slow at this stage of the scenario.")

    if dtemp > 0.03:
        lines.append("Surface temperature is trending upward and shifting climate conditions.")
    elif dtemp < -0.03:
        lines.append("Surface temperature is cooling toward a less stressed climate state.")
    else:
        lines.append("Global temperature is changing slowly year-to-year.")

    if dhab < -0.15:
        lines.append("Habitable regions are shrinking as more latitudes leave the chosen range.")
    elif dhab > 0.15:
        lines.append("Habitable regions are expanding as temperatures align with the selected range.")
    else:
        lines.append("Habitable surface is broadly stable around the current climate state.")

    if temp_c < 4.0:
        lines.append("Persistent cold and ice feedbacks are keeping this world snowball-sensitive.")

    if year >= params["mitigation_start_year"]:
        lines.append("Mitigation is active and reducing anthropogenic pressure on the system.")
    else:
        years_left = params["mitigation_start_year"] - year
        lines.append(f"Mitigation has not started yet ({years_left} years until activation).")

    return lines[:4]


def _safe_index(year: int, n: int):
    return max(0, min(int(year), n - 1))


def render_climate_twin_page():
    st.markdown("<style>[data-testid='stHeaderActionElements']{display:none;}</style>", unsafe_allow_html=True)
    st.title("Climate Twin")
    st.caption("Track temperature, atmospheric CO\u2082, and habitable surface through time.")

    params = _load_current_scenario()
    series = simulate_time_series(params)
    years = series["time_years"]
    events = series["events"]
    initial_conditions = series.get("initial_conditions", [])

    st.session_state.setdefault(YEAR_KEY, 0)
    st.session_state.setdefault(PLAY_KEY, False)
    st.session_state[YEAR_KEY] = int(np.clip(st.session_state[YEAR_KEY], 0, SIM_YEARS))
    idx = _safe_index(st.session_state[YEAR_KEY], len(years))

    current_temp_c = float(series["global_temperature_c"][idx])
    current_temp_k = float(series["global_temperature_k"][idx])
    current_co2 = float(series["co2_ppm"][idx])
    current_habitable = float(series["habitable_surface_percent"][idx])
    current_regime = str(series["regime"][idx])

    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        _kpi_card("Current Year", f"{int(years[idx])}", "Timeline")
    with kpi_cols[1]:
        thermal = "Cold" if current_temp_k < 283.0 else ("Mild" if current_temp_k <= 295.0 else "Hot")
        _kpi_card("Global Temperature", f"{current_temp_c:.1f} \u00b0C", thermal)
    with kpi_cols[2]:
        co2_level = "Low" if current_co2 / CO2_BASELINE_PPM < 1.5 else ("Elevated" if current_co2 / CO2_BASELINE_PPM < 2.5 else "High")
        _kpi_card("Atmospheric CO\u2082", f"{current_co2:.0f} ppm", co2_level)
    with kpi_cols[3]:
        hab_level = "Low" if current_habitable < 40.0 else ("Moderate" if current_habitable < 75.0 else "High Habitability")
        _kpi_card("Habitable Surface", f"{current_habitable:.1f}%", hab_level)
    with kpi_cols[4]:
        _kpi_card("Current Regime", current_regime, current_regime)

    left_col, right_col = st.columns([1.8, 1.0])

    with left_col:
        temp_fig = _build_chart(
            years,
            series["global_temperature_c"],
            "Global Temperature vs Time",
            "Temperature (\u00b0C)",
            "#FF8C42",
            int(years[idx]),
            events,
        )
        st.plotly_chart(temp_fig, use_container_width=True)

        co2_fig = _build_chart(
            years,
            series["co2_ppm"],
            "Atmospheric CO\u2082 vs Time",
            "CO\u2082 (ppm)",
            "#66D9EF",
            int(years[idx]),
            events,
        )
        st.plotly_chart(co2_fig, use_container_width=True)

        hab_fig = _build_chart(
            years,
            series["habitable_surface_percent"],
            "Habitable Surface vs Time",
            "Habitable Surface (%)",
            "#7ED957",
            int(years[idx]),
            events,
        )
        st.plotly_chart(hab_fig, use_container_width=True)

    with right_col:
        with st.container(border=True):
            st.subheader("Current Snapshot")
            st.markdown(
                "\n".join(
                    [
                        f"- Year: {int(years[idx])}",
                        f"- Global Temp: {current_temp_c:.2f} \u00b0C ({current_temp_k:.2f} K)",
                        f"- Atmospheric CO\u2082: {current_co2:.1f} ppm",
                        f"- Habitable Surface: {current_habitable:.1f}%",
                        f"- Current Regime: {current_regime}",
                    ]
                )
            )

        with st.container(border=True):
            st.subheader("What Is Happening?")
            for line in _explanation_lines(series, idx, params):
                st.markdown(f"- {line}")

        with st.container(border=True):
            st.subheader("Key Events")
            if initial_conditions:
                st.caption("Initial Conditions")
                for warning in initial_conditions:
                    color = _event_color(warning["kind"])
                    st.markdown(
                        (
                            f"<div style='padding:4px 0;'>"
                            f"<span style='color:{color};font-weight:700;'>Year 0</span>"
                            f"<span style='color:#d9e3f5;'> \u2014 {warning['label']}</span>"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
                st.markdown("<div style='height:0.2rem;'></div>", unsafe_allow_html=True)

            if not events:
                st.caption("No major timeline events detected in this scenario.")
            else:
                for i, event in enumerate(events):
                    row = st.columns([1, 5])
                    if row[0].button("Go", key=f"event_jump_{i}", use_container_width=True):
                        st.session_state[YEAR_KEY] = int(event["year"])
                        st.session_state[PLAY_KEY] = False
                        st.rerun()
                    color = _event_color(event["kind"])
                    row[1].markdown(
                        (
                            f"<div style='padding:6px 0;'>"
                            f"<span style='color:{color};font-weight:700;'>Year {event['year']}</span>"
                            f"<span style='color:#d9e3f5;'> \u2014 {event['label']}</span>"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    controls = st.columns([1.1, 1.1, 7.0, 1.2, 1.2])
    if controls[0].button("Play", use_container_width=True):
        st.session_state[PLAY_KEY] = True
    if controls[1].button("Pause", use_container_width=True):
        st.session_state[PLAY_KEY] = False

    controls[2].slider("Year", min_value=0, max_value=SIM_YEARS, key=YEAR_KEY)

    mitigation_year = int(params["mitigation_start_year"])
    if controls[3].button("Mitigation", use_container_width=True):
        st.session_state[YEAR_KEY] = int(np.clip(mitigation_year, 0, SIM_YEARS))
        st.session_state[PLAY_KEY] = False
        st.rerun()

    if controls[4].button("Next Event", use_container_width=True):
        current_year = int(st.session_state[YEAR_KEY])
        future = [event["year"] for event in events if int(event["year"]) > current_year]
        if future:
            st.session_state[YEAR_KEY] = int(future[0])
            st.session_state[PLAY_KEY] = False
            st.rerun()

    if st.session_state.get(PLAY_KEY, False):
        if st.session_state[YEAR_KEY] >= SIM_YEARS:
            st.session_state[PLAY_KEY] = False
        else:
            time.sleep(0.08)
            st.session_state[YEAR_KEY] = int(st.session_state[YEAR_KEY] + 1)
            st.rerun()


render_climate_twin_page()
