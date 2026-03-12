import json
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from htp.model.constants import CO2_BASELINE_PPM, SIM_DT_YEARS_DEFAULT, SIM_YEARS_DEFAULT
from htp.model.scenario_io import (
    import_scenario_json,
    load_scenario_from_session,
    scenario_signature,
    scenario_to_flat_params,
)
from htp.model.simulate import simulate_time_series

st.set_page_config(page_title="Climate Twin", layout="wide")

YEAR_KEY = "climate_twin_year"
PLAY_KEY = "climate_twin_playing"
SERIES_CACHE_KEY = "climate_twin_series_cache"


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


def _event_color(kind: str):
    if kind == "mitigation":
        return "#4F9DFF"
    if kind == "habitability":
        return "#D2A106"
    if kind == "danger":
        return "#B52A2A"
    return "#8899B8"


@st.cache_data(show_spinner=False)
def _simulate_cached(signature: str, scenario_json: str, years: int, dt_years: float):
    scenario = import_scenario_json(scenario_json)
    result = simulate_time_series(scenario, years=years, dt_years=dt_years)
    return {
        "time_years": np.asarray(result["time_years"], dtype=float).tolist(),
        "global_temperature_c": np.asarray(result["global_temperature_c"], dtype=float).tolist(),
        "global_temperature_k": np.asarray(result["global_temperature_k"], dtype=float).tolist(),
        "co2_ppm": np.asarray(result["co2_ppm"], dtype=float).tolist(),
        "habitable_surface_percent": np.asarray(result["habitable_surface_percent"], dtype=float).tolist(),
        "regime": list(result["regime"]),
        "effective_albedo": np.asarray(result.get("effective_albedo", []), dtype=float).tolist(),
        "initial_conditions": list(result.get("initial_conditions", [])),
        "events": list(result.get("events", [])),
        "meta": dict(result.get("meta", {})),
    }


def _series_to_numpy(series: dict):
    return {
        "time_years": np.asarray(series["time_years"], dtype=float),
        "global_temperature_c": np.asarray(series["global_temperature_c"], dtype=float),
        "global_temperature_k": np.asarray(series["global_temperature_k"], dtype=float),
        "co2_ppm": np.asarray(series["co2_ppm"], dtype=float),
        "habitable_surface_percent": np.asarray(series["habitable_surface_percent"], dtype=float),
        "regime": list(series["regime"]),
        "effective_albedo": np.asarray(series.get("effective_albedo", []), dtype=float),
        "initial_conditions": list(series.get("initial_conditions", [])),
        "events": list(series.get("events", [])),
        "meta": dict(series.get("meta", {})),
    }


def _store_series_cache(signature: str, series: dict):
    st.session_state[SERIES_CACHE_KEY] = {
        "scenario_signature": signature,
        "time_years": np.asarray(series["time_years"], dtype=float).tolist(),
        "global_temperature_c": np.asarray(series["global_temperature_c"], dtype=float).tolist(),
        "co2_ppm": np.asarray(series["co2_ppm"], dtype=float).tolist(),
        "habitable_surface_percent": np.asarray(series["habitable_surface_percent"], dtype=float).tolist(),
        "meta": dict(series.get("meta", {})),
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


def _safe_index(year: int, n: int):
    return max(0, min(int(year), n - 1))


def _explanation_lines(series, idx: int, params):
    lines = []
    temp_c = float(series["global_temperature_c"][idx])
    co2_ppm = float(series["co2_ppm"][idx])
    habitable = float(series["habitable_surface_percent"][idx])
    year = int(round(series["time_years"][idx]))
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
        lines.append("Atmospheric CO₂ is increasing, strengthening greenhouse forcing.")
    elif dco2 < -0.1:
        lines.append("Atmospheric CO₂ is declining as sinks and mitigation counter emissions.")
    else:
        lines.append("CO₂ growth is relatively slow at this stage of the scenario.")

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

    if year >= int(params["mitigation_start_year"]):
        lines.append("Mitigation is active and reducing anthropogenic pressure on the system.")
    else:
        years_left = int(params["mitigation_start_year"]) - year
        lines.append(f"Mitigation has not started yet ({years_left} years until activation).")

    return lines[:4]


def render_climate_twin_page():
    st.markdown("<style>[data-testid='stHeaderActionElements']{display:none;}</style>", unsafe_allow_html=True)
    st.title("Climate Twin")
    st.caption("Track temperature, atmospheric CO₂, and habitable surface through time.")

    scenario = load_scenario_from_session(st.session_state)
    params = scenario_to_flat_params(scenario)
    signature = scenario_signature(scenario)
    scenario_json = json.dumps(scenario.model_dump(mode="json"), sort_keys=True)
    series = _series_to_numpy(
        _simulate_cached(signature, scenario_json, years=SIM_YEARS_DEFAULT, dt_years=SIM_DT_YEARS_DEFAULT)
    )
    _store_series_cache(signature, series)

    years = series["time_years"]
    events = series["events"]
    initial_conditions = series.get("initial_conditions", [])

    st.session_state.setdefault(YEAR_KEY, 0)
    st.session_state.setdefault(PLAY_KEY, False)
    st.session_state[YEAR_KEY] = int(np.clip(st.session_state[YEAR_KEY], 0, SIM_YEARS_DEFAULT))
    idx = _safe_index(st.session_state[YEAR_KEY], len(years))

    current_temp_c = float(series["global_temperature_c"][idx])
    current_temp_k = float(series["global_temperature_k"][idx])
    current_co2 = float(series["co2_ppm"][idx])
    current_habitable = float(series["habitable_surface_percent"][idx])
    current_regime = str(series["regime"][idx])

    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        _kpi_card("Current Year", f"{int(round(years[idx]))}", "Timeline")
    with kpi_cols[1]:
        thermal = "Cold" if current_temp_k < 283.0 else ("Mild" if current_temp_k <= 295.0 else "Hot")
        _kpi_card("Global Temperature", f"{current_temp_c:.1f} °C", thermal)
    with kpi_cols[2]:
        ratio = current_co2 / CO2_BASELINE_PPM
        co2_level = "Low" if ratio < 1.5 else ("Elevated" if ratio < 2.5 else "High")
        _kpi_card("Atmospheric CO₂", f"{current_co2:.0f} ppm", co2_level)
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
            "Temperature (°C)",
            "#FF8C42",
            int(round(years[idx])),
            events,
        )
        st.plotly_chart(temp_fig, use_container_width=True, key="climate_twin_temp_chart")

        co2_fig = _build_chart(
            years,
            series["co2_ppm"],
            "Atmospheric CO₂ vs Time",
            "CO₂ (ppm)",
            "#66D9EF",
            int(round(years[idx])),
            events,
        )
        st.plotly_chart(co2_fig, use_container_width=True, key="climate_twin_co2_chart")

        hab_fig = _build_chart(
            years,
            series["habitable_surface_percent"],
            "Habitable Surface vs Time",
            "Habitable Surface (%)",
            "#7ED957",
            int(round(years[idx])),
            events,
        )
        st.plotly_chart(hab_fig, use_container_width=True, key="climate_twin_hab_chart")

    with right_col:
        with st.container(border=True):
            st.subheader("Current Snapshot")
            st.markdown(
                "\n".join(
                    [
                        f"- Year: {int(round(years[idx]))}",
                        f"- Global Temp: {current_temp_c:.2f} °C ({current_temp_k:.2f} K)",
                        f"- Atmospheric CO₂: {current_co2:.1f} ppm",
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
                            f"<span style='color:#d9e3f5;'> — {warning['label']}</span>"
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
                            f"<span style='color:#d9e3f5;'> — {event['label']}</span>"
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

    selected_year = controls[2].slider(
        "Year",
        min_value=0,
        max_value=SIM_YEARS_DEFAULT,
        value=int(st.session_state[YEAR_KEY]),
        step=10,
    )
    if int(selected_year) != int(st.session_state[YEAR_KEY]):
        st.session_state[YEAR_KEY] = int(selected_year)
        st.session_state[PLAY_KEY] = False
        st.rerun()

    mitigation_year = int(params["mitigation_start_year"])
    if controls[3].button("Mitigation", use_container_width=True):
        st.session_state[YEAR_KEY] = int(np.clip(mitigation_year, 0, SIM_YEARS_DEFAULT))
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
        if st.session_state[YEAR_KEY] >= SIM_YEARS_DEFAULT:
            st.session_state[PLAY_KEY] = False
        else:
            time.sleep(0.07)
            st.session_state[YEAR_KEY] = int(min(SIM_YEARS_DEFAULT, st.session_state[YEAR_KEY] + 10))
            st.rerun()


render_climate_twin_page()

