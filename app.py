import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Habitat Tipping Points", layout="wide")


def _planet_surface(lon, lat, seed: int, temp_c: float):
    rng = np.random.default_rng(seed)
    # Fractal-style continental noise.
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

    # Extra mountain ridges only on land.
    ridges = np.abs(np.sin(9 * lon + rng.uniform(0, 2 * np.pi)) * np.cos(11 * lat))
    ridges *= np.abs(np.sin(15 * lon + rng.uniform(0, 2 * np.pi)))
    ridges = (ridges - ridges.min()) / (ridges.max() - ridges.min() + 1e-9)
    mountain = np.clip(0.65 * land_height + 0.55 * ridges, 0, 1)

    # Cold poles produce ice caps; warmer planets shrink them.
    abs_lat = np.abs(lat) / (np.pi / 2)  # 0 at equator, 1 at poles
    ice_threshold = np.clip(0.72 + 0.004 * temp_c, 0.56, 0.90)
    ice = abs_lat > ice_threshold

    # Surface color index [0..1] mapped by colorscale below.
    surf = np.where(land, 0.50 + 0.32 * land_height, 0.06 + 0.26 * (1 - ocean_depth))
    surf = np.where(land & (mountain > 0.72), 0.84 + 0.10 * mountain, surf)
    surf = np.where(ice, 0.97, surf)
    surf = np.clip(surf, 0, 1)

    # Geometric displacement for nicer relief.
    relief = np.where(land, 0.02 + 0.05 * mountain, -0.004 * ocean_depth)
    relief = np.where(ice, relief + 0.012, relief)
    return surf, relief


def draw_planet(
    temp_c: float,
    co2_ppm: float,
    albedo: float,
    stellar_energy: float,
    radius_km: float,
    seed: int,
):
    
    #to change just a part of the planet, we do
    # 1) Build a region mask (example: equatorial belt on land)
    # region_mask = land & (np.abs(lat) < 0.35)

    # 2) Change only that part in the surface index
    # surf = np.where(region_mask, 0.72, surf)  # 0.72 maps to a specific color band
    
    
    heat = max(0.0, min(1.0, (temp_c + 30.0) / 80.0))
    co2_factor = max(0.0, min(1.0, co2_ppm / 1200.0))
    light = max(0.2, min(1.2, stellar_energy / 1361.0))

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    radius_scale = np.clip(radius_km / 6371.0, 0.35, 2.2)
    x0 = np.outer(np.cos(u), np.sin(v))
    y0 = np.outer(np.sin(u), np.sin(v))
    z0 = np.outer(np.ones_like(u), np.cos(v))
    lon = np.arctan2(y0, x0)
    lat = np.arcsin(np.clip(z0, -1, 1))

    base_r = 0.15 + 0.75 * heat
    base_g = 0.55 - 0.25 * heat + 0.10 * (1.0 - co2_factor)
    base_b = 0.85 - 0.65 * heat
    brightness = (0.75 + 0.35 * albedo) * light
    color = np.clip(np.array([base_r, base_g, base_b]) * brightness, 0, 1)

    texture, relief = _planet_surface(lon, lat, seed=seed, temp_c=temp_c)
    rfield = radius_scale * (1.0 + relief)
    x = rfield * x0
    y = rfield * y0
    z = rfield * z0

    c0 = np.clip(color * np.array([0.15, 0.30, 0.90]), 0, 1)  # deep ocean
    c1 = np.clip(color * np.array([0.30, 0.65, 1.05]), 0, 1)  # shallow ocean
    c2 = np.clip(color * np.array([0.92, 0.84, 0.56]), 0, 1)  # coast/beach
    c3 = np.clip(color * np.array([0.42, 0.88, 0.42]), 0, 1)  # lowland
    c4 = np.clip(color * np.array([0.23, 0.60, 0.28]), 0, 1)  # highland
    c5 = np.clip(color * np.array([0.60, 0.58, 0.55]), 0, 1)  # mountain 
    c6 = np.array([0.96, 0.97, 1.0])  # ice/snow
    colorscale = [
        [0.00, f"rgb({int(255*c0[0])}, {int(255*c0[1])}, {int(255*c0[2])})"],
        [0.28, f"rgb({int(255*c1[0])}, {int(255*c1[1])}, {int(255*c1[2])})"],
        [0.47, f"rgb({int(255*c2[0])}, {int(255*c2[1])}, {int(255*c2[2])})"],
        [0.63, f"rgb({int(255*c3[0])}, {int(255*c3[1])}, {int(255*c3[2])})"],
        [0.80, f"rgb({int(255*c4[0])}, {int(255*c4[1])}, {int(255*c4[2])})"],
        [0.92, f"rgb({int(255*c5[0])}, {int(255*c5[1])}, {int(255*c5[2])})"],
        [1.00, f"rgb({int(255*c6[0])}, {int(255*c6[1])}, {int(255*c6[2])})"],
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

nav1, nav2 = st.columns(2)
if nav1.button("Screen 2"):
    st.switch_page("pages/screen_2.py")
if nav2.button("Screen 3"):
    st.switch_page("pages/screen_3.py")

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
        invalid = (
            radius <= 0
            or co2_ppm < 0
            or albedo < 0
            or albedo > 1
            or temperature_c < -120
            or temperature_c > 120
            or stellar_energy < 0
        )

        if invalid:
            st.session_state["submitted"] = False
            st.session_state.pop("params", None)
            st.error("Invalid Input")
        else:
            st.session_state["submitted"] = True
            st.session_state["texture_seed"] = int(np.random.randint(0, 1_000_000_000))
            st.session_state["params"] = {
                "human_parameters": {
                    "emission_rate": emission_rate,
                    "emission_mitigation_rate": mitigation_rate,
                },
                "initial_planet_parameters": {
                    "radius": radius,
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
        p = st.session_state["params"]["initial_planet_parameters"]
        fig = draw_planet(
            temp_c=float(p["temperature_c"]),
            co2_ppm=float(p["co2_ppm"]),
            albedo=float(p["albedo"]),
            stellar_energy=float(p["stellar_energy_w_m2"]),
            radius_km=float(p["radius"]),
            seed=int(st.session_state.get("texture_seed", 0)),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.json(st.session_state["params"])
    else:
        st.info("Fill the form and click Submit.")
