import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Screen 2 - Planet + Sun", layout="wide")
st.title("Screen 2: Planet + Sun")


def make_sphere(center: tuple[float, float, float], radius: float, samples: int = 90):
    u = np.linspace(0, 2 * np.pi, samples)
    v = np.linspace(0, np.pi, samples)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    return x, y, z


params = st.session_state.get("params")
if not params or "initial_planet_parameters" not in params:
    st.info("No parameters found. Go back to the main page and submit the form first.")
    st.stop()

p = params["initial_planet_parameters"]

radius_km = float(p.get("radius", 6371.0))
temp_c = float(p.get("temperature_c", 15.0))
co2_ppm = float(p.get("co2_ppm", 420.0))
albedo = float(p.get("albedo", 0.3))
stellar_energy = float(p.get("stellar_energy_w_m2", 1361.0))

if radius_km <= 0 or stellar_energy <= 0 or co2_ppm < 0 or not (0 <= albedo <= 1):
    st.error("Invalid Input")
    st.stop()

# Inverse-square law against Earth reference at 1 AU.
distance_au = np.sqrt(1361.0 / stellar_energy)

# Visual compression so both bodies fit in one 3D scene.
planet_r = np.clip(radius_km / 6371.0, 0.35, 2.2)
sun_r = 1.0 + np.clip(temp_c / 200.0, -0.2, 0.4) + (1.0 - albedo) * 0.15
scene_distance = 7.0 * distance_au

px, py, pz = make_sphere((0.0, 0.0, 0.0), planet_r)
sx, sy, sz = make_sphere((scene_distance, 0.0, 0.0), sun_r)

# Planet coloring responds to climate params.
heat = max(0.0, min(1.0, (temp_c + 30.0) / 80.0))
co2_factor = max(0.0, min(1.0, co2_ppm / 1200.0))
planet_colorscale = [
    [0.0, "rgb(10,35,120)"],
    [0.35, "rgb(45,110,205)"],
    [0.48, "rgb(190,170,110)"],
    [0.64, f"rgb({int(50 + 80 * heat)},{int(120 - 40 * heat)},{int(45 + 30 * (1 - co2_factor))})"],
    [0.82, "rgb(100,92,82)"],
    [1.0, "rgb(245,247,255)"],
]

fig = go.Figure()
fig.add_trace(
    go.Surface(
        x=px,
        y=py,
        z=pz,
        surfacecolor=pz,
        colorscale=planet_colorscale,
        showscale=False,
        lighting=dict(ambient=0.45, diffuse=0.85, specular=0.2, roughness=0.9),
    )
)
fig.add_trace(
    go.Surface(
        x=sx,
        y=sy,
        z=sz,
        surfacecolor=sz,
        colorscale=[[0, "#ffb347"], [0.5, "#ffd05a"], [1, "#fff4bf"]],
        showscale=False,
        lighting=dict(ambient=0.95, diffuse=1.0, specular=0.55, roughness=0.25),
        lightposition=dict(x=220, y=120, z=200),
    )
)

fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="rgb(6, 9, 20)",
        aspectmode="data",
        camera=dict(eye=dict(x=1.7, y=1.25, z=0.9)),
    ),
)

st.plotly_chart(fig, use_container_width=True)
st.caption(f"Derived star distance: {distance_au:.2f} AU (from stellar energy {stellar_energy:.2f} W/m^2)")
