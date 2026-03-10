import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Screen 2 - Orbit View", layout="wide")
st.title("Screen 2: Scientific Orbit View")


def make_sphere(radius: float, samples: int = 52):
    u = np.linspace(0.0, 2.0 * np.pi, samples)
    v = np.linspace(0.0, np.pi, samples)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def rotate_x(x, y, z, angle_rad: float):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    y2 = c * y - s * z
    z2 = s * y + c * z
    return x, y2, z2


def rotate_z(x, y, z, angle_rad: float):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    x2 = c * x - s * y
    y2 = s * x + c * y
    return x2, y2, z


def planet_texture(x, y, z):
    r = np.maximum(np.sqrt(x * x + y * y + z * z), 1e-9)
    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z / r, -1.0, 1.0))
    tex = (
        0.50
        + 0.18 * np.sin(3.2 * lon) * np.cos(1.9 * lat)
        + 0.14 * np.sin(7.0 * lon + 1.1)
        + 0.09 * np.cos(5.0 * lat - 0.3)
    )
    tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-9)
    return tex


default_initial = {
    "radius": 6371.0,
    "temperature_c": 15.0,
    "co2_ppm": 420.0,
    "albedo": 0.30,
    "stellar_energy_w_m2": 1361.0,
    "seasonal_change": "Yes",
}

params = st.session_state.get("params", {})
p = params.get("initial_planet_parameters", default_initial) if isinstance(params, dict) else default_initial

radius_km = float(p.get("radius", 6371.0))
temp_c = float(p.get("temperature_c", 15.0))
co2_ppm = float(p.get("co2_ppm", 420.0))
albedo = float(p.get("albedo", 0.3))
stellar_energy = float(p.get("stellar_energy_w_m2", 1361.0))
seasonal_raw = str(p.get("seasonal_change", "No")).strip().lower()
seasonal_change = seasonal_raw in {"yes", "y", "true", "1", "tak", "так"}

if radius_km <= 0 or stellar_energy <= 0 or co2_ppm < 0 or not (0.0 <= albedo <= 1.0):
    st.warning("Invalid input detected. Using fallback values.")
    radius_km = 6371.0
    temp_c = 15.0
    co2_ppm = 420.0
    albedo = 0.30
    stellar_energy = 1361.0
    seasonal_change = False

ctrl_a, ctrl_b, ctrl_c = st.columns(3)
speed_mult = ctrl_a.slider("Simulation speed (x)", min_value=0.1, max_value=1000.0, value=100.0, step=0.1)
fps = ctrl_b.slider("Animation FPS", min_value=10, max_value=60, value=24, step=1)
zoom_target = ctrl_c.selectbox("Zoom target", ["Global", "Planet", "Sun"], index=0)

# Physical relations:
# F = F_earth / d^2  -> d(AU) = sqrt(F_earth / F)
distance_au = np.sqrt(1361.0 / stellar_energy)
orbital_period_days = 365.25 * distance_au ** 1.5  # Kepler for 1 solar mass approximation

if seasonal_change:
    # Plausible spin slowdown for larger planets and tighter orbits.
    base_spin_h = 24.0 * (radius_km / 6371.0) ** 0.65
    tidal_slowdown = 1.0 + 1.8 * np.clip((1.0 / max(distance_au, 0.2)) - 1.0, 0.0, 4.0)
    thermal_factor = 1.0 + 0.15 * np.clip((temp_c - 15.0) / 60.0, -0.5, 1.5)
    spin_period_hours = float(np.clip(base_spin_h * tidal_slowdown * thermal_factor, 8.0, 500.0))
else:
    spin_period_hours = None

# Scene geometry (display-normalized, preserving distance ratio)
planet_r = np.clip(radius_km / 6371.0, 0.35, 2.4)
sun_r = 1.0 + np.clip(temp_c / 200.0, -0.2, 0.4) + (1.0 - albedo) * 0.15
orbit_r = max(7.0 * distance_au, sun_r + planet_r + 1.0)

px0, py0, pz0 = make_sphere(planet_r, samples=52)
sx0, sy0, sz0 = make_sphere(sun_r, samples=52)
planet_surface = planet_texture(px0, py0, pz0)

frame_count = 150
frame_ms = int(round(1000.0 / fps))
sim_days_per_frame = speed_mult / fps
tilt = np.deg2rad(23.44) if seasonal_change else 0.0


def planet_state(frame_idx: int):
    orbit_angle = 2.0 * np.pi * (frame_idx * sim_days_per_frame) / orbital_period_days
    spin_angle = 0.0
    if spin_period_hours is not None:
        spin_angle = 2.0 * np.pi * (frame_idx * sim_days_per_frame * 24.0) / spin_period_hours

    x, y, z = rotate_x(px0, py0, pz0, tilt)
    if seasonal_change:
        x, y, z = rotate_z(x, y, z, spin_angle)

    cx = orbit_r * np.cos(orbit_angle)
    cy = orbit_r * np.sin(orbit_angle)
    return x + cx, y + cy, z


orbit_theta = np.linspace(0.0, 2.0 * np.pi, 360)
x_init, y_init, z_init = planet_state(0)

planet_colorscale = [
    [0.00, "rgb(8,34,120)"],
    [0.32, "rgb(42,112,210)"],
    [0.46, "rgb(194,175,116)"],
    [0.66, "rgb(70,140,70)"],
    [0.84, "rgb(120,108,96)"],
    [1.00, "rgb(248,250,255)"],
]

fig = go.Figure()
fig.add_trace(
    go.Surface(
        x=x_init,
        y=y_init,
        z=z_init,
        surfacecolor=planet_surface,
        colorscale=planet_colorscale,
        showscale=False,
        lighting=dict(ambient=0.45, diffuse=0.85, specular=0.2, roughness=0.9),
        name="Planet",
    )
)
fig.add_trace(
    go.Surface(
        x=sx0,
        y=sy0,
        z=sz0,
        surfacecolor=sz0,
        colorscale=[[0, "#ffb347"], [0.5, "#ffd05a"], [1, "#fff4bf"]],
        showscale=False,
        lighting=dict(ambient=0.95, diffuse=1.0, specular=0.55, roughness=0.25),
        lightposition=dict(x=220, y=120, z=200),
        name="Star",
    )
)
fig.add_trace(
    go.Scatter3d(
        x=orbit_r * np.cos(orbit_theta),
        y=orbit_r * np.sin(orbit_theta),
        z=np.zeros_like(orbit_theta),
        mode="lines",
        line=dict(color="rgba(180,180,220,0.35)", width=3),
        hoverinfo="skip",
        showlegend=False,
    )
)

frames = []
for i in range(frame_count):
    xf, yf, zf = planet_state(i)
    frames.append(
        go.Frame(
            name=str(i),
            data=[go.Surface(x=xf, y=yf, z=zf, surfacecolor=planet_surface)],
            traces=[0],
        )
    )
fig.frames = frames

if zoom_target == "Planet":
    span = max(planet_r * 3.6, 0.5)
    x_range = [-orbit_r - span, orbit_r + span]
    y_range = [-orbit_r - span, orbit_r + span]
    z_range = [-span, span]
elif zoom_target == "Sun":
    span = max(sun_r * 4.0, 0.5)
    x_range = [-span, span]
    y_range = [-span, span]
    z_range = [-span, span]
else:
    ext = orbit_r + max(planet_r, sun_r) + 1.2
    x_range = [-ext, ext]
    y_range = [-ext, ext]
    z_range = [-ext * 0.45, ext * 0.45]

# Long play sequence to behave as "continuous" spin without JS hacks.
loop_count = max(1, int((45 * 60 * 1000) / (frame_count * frame_ms)))  # ~45 minutes
play_sequence = [str(i) for _ in range(loop_count) for i in range(frame_count)]

fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    uirevision="orbit-view",
    scene=dict(
        xaxis=dict(visible=False, range=x_range),
        yaxis=dict(visible=False, range=y_range),
        zaxis=dict(visible=False, range=z_range),
        bgcolor="rgb(6, 9, 20)",
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.35),
        camera=dict(eye=dict(x=1.55, y=1.25, z=0.65)),
        uirevision="orbit-view",
    ),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        play_sequence,
                        {
                            "frame": {"duration": frame_ms, "redraw": True},
                            "transition": {"duration": 0},
                            "fromcurrent": True,
                            "mode": "immediate",
                        },
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "transition": {"duration": 0},
                            "mode": "immediate",
                        },
                    ],
                ),
            ],
        )
    ],
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "scrollZoom": True,
        "displaylogo": False,
    },
    key="orbit_chart",
)

spin_text = "Disabled (seasonal_change = No)"
if spin_period_hours is not None:
    spin_text = f"{spin_period_hours:.1f} h/rotation"

st.caption(
    f"Distance: {distance_au:.3f} AU | Orbital period: {orbital_period_days:.1f} days | "
    f"Self-spin: {spin_text} | Speed: {speed_mult:.1f}x"
)
st.json({"initial_planet_parameters": p})
