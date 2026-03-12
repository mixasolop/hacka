"""Reusable Plotly spinning-sphere renderer for Streamlit."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit.components.v1 as components


def render_spinning_surface(
    fig: go.Figure,
    *,
    component_key: str,
    height_px: int,
    speed_deg_per_sec: float = 10.2,
    include_plotlyjs: str = "inline",
) -> None:
    div_id = f"{component_key}_plot"
    post_script = f"""
const plot = document.getElementById('{div_id}');
if (plot) {{
  const speedDegPerSec = {float(speed_deg_per_sec):.3f};
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
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        default_width="100%",
        default_height=f"{int(height_px)}px",
        div_id=div_id,
        post_script=post_script,
        config={"displayModeBar": True, "responsive": True},
    )
    components.html(html, height=int(height_px) + 12, scrolling=False)
