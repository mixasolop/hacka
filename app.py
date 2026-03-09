import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Habitat Tipping Points", layout="wide")


def _water_texture(x, y, z, seed: int):
    rng = np.random.default_rng(seed)
    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z, -1, 1))

    tex = np.zeros_like(x, dtype=float)
    for _ in range(8):
        k1 = rng.integers(1, 6)
        k2 = rng.integers(1, 6)
        phase1 = rng.uniform(0, 2 * np.pi)
        phase2 = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.08, 0.25)
        tex += amp * np.sin(k1 * lon + phase1) * np.cos(k2 * lat + phase2)

    tex += 0.15 * np.sin(14 * lon + rng.uniform(0, 2 * np.pi))
    tex += 0.08 * np.cos(18 * lat + rng.uniform(0, 2 * np.pi))
    tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-9)
    return tex


def draw_planet(
    temp_c: float,
    co2_ppm: float,
    albedo: float,
    stellar_energy: float,
    radius_km: float,
    seed: int,
):
    heat = max(0.0, min(1.0, (temp_c + 30.0) / 80.0))
    co2_factor = max(0.0, min(1.0, co2_ppm / 1200.0))
    light = max(0.2, min(1.2, stellar_energy / 1361.0))

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    radius_scale = np.clip(radius_km / 6371.0, 0.35, 2.2)
    x = radius_scale * np.outer(np.cos(u), np.sin(v))
    y = radius_scale * np.outer(np.sin(u), np.sin(v))
    z = radius_scale * np.outer(np.ones_like(u), np.cos(v))

    base_r = 0.15 + 0.75 * heat
    base_g = 0.55 - 0.25 * heat + 0.10 * (1.0 - co2_factor)
    base_b = 0.85 - 0.65 * heat
    brightness = (0.75 + 0.35 * albedo) * light
    color = np.clip(np.array([base_r, base_g, base_b]) * brightness, 0, 1)

    texture = _water_texture(x, y, z, seed=seed)

    c0 = np.clip(color * np.array([0.25, 0.45, 0.95]), 0, 1)
    c1 = np.clip(color * np.array([0.35, 0.75, 1.15]), 0, 1)
    c2 = np.clip(color * np.array([0.55, 1.05, 1.20]), 0, 1)
    c3 = np.clip(color * np.array([0.75, 1.25, 1.25]), 0, 1)
    colorscale = [
        [0.00, f"rgb({int(255*c0[0])}, {int(255*c0[1])}, {int(255*c0[2])})"],
        [0.45, f"rgb({int(255*c1[0])}, {int(255*c1[1])}, {int(255*c1[2])})"],
        [0.78, f"rgb({int(255*c2[0])}, {int(255*c2[1])}, {int(255*c2[2])})"],
        [1.00, f"rgb({int(255*c3[0])}, {int(255*c3[1])}, {int(255*c3[2])})"],
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


st.title("Habitat Tipping Points")
st.caption("Enter parameters and click Submit to draw the 3D planet on the right.")

left, right = st.columns([1.1, 1.0])

with left:
    with st.form("input_form"):
        st.subheader("Human Parameters")
        emission_rate = st.number_input("Emission rate", min_value=0.0, value=2.5, step=0.1)
        mitigation_rate = st.number_input(
            "Emission mitigation rate (0.0 - 100.0)",
            min_value=0.0,
            max_value=100.0,
            value=1.2,
            step=0.1,
        )

        st.subheader("Initial Planet Parameters")
        radius = st.number_input("Planet radius (km)", value=6371.0, step=100.0)
        factors = st.text_input("Factors", placeholder="e.g. volcanoes, oceans, forests")
        temperature_c = st.number_input("Temperature (C)", value=15.0, step=0.1)
        co2_ppm = st.number_input("CO2 level (ppm)", value=420.0, step=1.0)
        albedo = st.number_input("Albedo (0.00 - 1.00)", value=0.30, step=0.01)
        stellar_energy = st.number_input(
            "Stellar energy (W/m^2)", min_value=0.0, value=1361.0, step=1.0
        )
        seasonal_change = st.selectbox("Seasonal change", ["Yes", "No"], index=0)

        submit = st.form_submit_button("Submit")

    if submit:
        st.session_state["submitted"] = True
        st.session_state["texture_seed"] = int(np.random.randint(0, 1_000_000_000))
        validation_warnings = []
        if radius <= 0:
            validation_warnings.append("Radius must be > 0 km. Using 6371 km for rendering.")
        if co2_ppm < 0:
            validation_warnings.append("CO2 should be >= 0 ppm.")
        if not 0 <= albedo <= 1:
            validation_warnings.append("Albedo should be between 0 and 1.")

        st.session_state["warnings"] = validation_warnings
        st.session_state["params"] = {
            "human_parameters": {
                "emission_rate": emission_rate,
                "emission_mitigation_rate": mitigation_rate,
            },
            "initial_planet_parameters": {
                "radius" : radius,
                "factors": factors,
                "temperature_c": temperature_c,
                "co2_ppm": co2_ppm,
                "albedo": albedo,
                "stellar_energy_w_m2": stellar_energy,
                "seasonal_change": seasonal_change,
            },
        }
        st.success("Input captured.")

with right:
    st.subheader("3D Planet")
    if st.session_state.get("submitted"):
        for warning in st.session_state.get("warnings", []):
            st.warning(warning)
        if st.button("Regenerate texture"):
            st.session_state["texture_seed"] = int(np.random.randint(0, 1_000_000_000))
        p = st.session_state["params"]["initial_planet_parameters"]
        radius_for_render = float(p["radius"]) if float(p["radius"]) > 0 else 6371.0
        fig = draw_planet(
            temp_c=float(p["temperature_c"]),
            co2_ppm=float(p["co2_ppm"]),
            albedo=float(p["albedo"]),
            stellar_energy=float(p["stellar_energy_w_m2"]),
            radius_km=radius_for_render,
            seed=int(st.session_state.get("texture_seed", 0)),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.json(st.session_state["params"])
    else:
        st.info("Fill the form and click Submit.")
