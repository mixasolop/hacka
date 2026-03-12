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
  const interactionPauseMs = 420;
  const returnDurationMs = 680;
  const toRad = Math.PI / 180.0;
  const normalizeDeg = (deg) => ((deg % 360.0) + 360.0) % 360.0;
  const initialEye = plot.layout?.scene?.camera?.eye ?? {{x: 1.8, y: 1.4, z: 1.0}};
  const initialUp = plot.layout?.scene?.camera?.up ?? {{x: 0.0, y: 0.0, z: 1.0}};
  const baseRadius = Math.max(0.5, Math.hypot(initialEye.x ?? 1.8, initialEye.y ?? 1.4));
  const baseZ = Number.isFinite(initialEye.z) ? initialEye.z : 1.0;
  let displayRadius = baseRadius;
  let displayZ = baseZ;
  let autoAngleDeg = normalizeDeg(Math.atan2(initialEye.y ?? 1.4, initialEye.x ?? 1.8) / toRad);
  let displayAngleDeg = autoAngleDeg;
  let returnStartTs = null;
  let returnFromRadius = displayRadius;
  let returnFromZ = displayZ;
  let wasPaused = false;
  let lastTs = performance.now();
  let userPauseUntil = 0;
  let internalUpdate = false;
  let pending = false;
  const syncFromCamera = () => {{
    const eye = plot.layout?.scene?.camera?.eye;
    if (!eye) return;
    const x = Number(eye.x);
    const y = Number(eye.y);
    const z = Number(eye.z);
    if (Number.isFinite(x) && Number.isFinite(y)) {{
      displayRadius = Math.max(0.5, Math.hypot(x, y));
      displayAngleDeg = normalizeDeg(Math.atan2(y, x) / toRad);
    }}
    if (Number.isFinite(z)) {{
      displayZ = z;
    }}
  }};
  const pauseFromInteraction = () => {{
    syncFromCamera();
    userPauseUntil = performance.now() + interactionPauseMs;
    returnStartTs = null;
  }};
  plot.addEventListener('pointerdown', pauseFromInteraction, {{ passive: true }});
  plot.addEventListener('wheel', pauseFromInteraction, {{ passive: true }});
  plot.addEventListener('touchstart', pauseFromInteraction, {{ passive: true }});
  plot.on('plotly_click', pauseFromInteraction);
  plot.on('plotly_doubleclick', pauseFromInteraction);
  plot.on('plotly_relayouting', () => {{
    if (!internalUpdate) {{
      pauseFromInteraction();
    }}
  }});
  plot.on('plotly_relayout', (ev) => {{
    if (internalUpdate) return;
    if (!ev) return;
    if (Object.prototype.hasOwnProperty.call(ev, 'scene.camera') ||
        Object.prototype.hasOwnProperty.call(ev, 'scene.camera.eye') ||
        Object.prototype.hasOwnProperty.call(ev, 'scene.camera.eye.x') ||
        Object.prototype.hasOwnProperty.call(ev, 'scene.camera.eye.y') ||
        Object.prototype.hasOwnProperty.call(ev, 'scene.camera.eye.z') ||
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
    const angleRad = displayAngleDeg * toRad;
    const result = Plotly.relayout(plot, {{
      'scene.camera.eye': {{x: displayRadius * Math.cos(angleRad), y: displayRadius * Math.sin(angleRad), z: displayZ}},
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
    const paused = ts < userPauseUntil;
    if (paused) {{
      wasPaused = true;
      syncFromCamera();
    }} else {{
      if (wasPaused) {{
        wasPaused = false;
        // Resume from the current release angle instead of "catching up"
        // to an old timeline angle to avoid visible jumps.
        autoAngleDeg = displayAngleDeg;
        returnStartTs = ts;
        returnFromRadius = displayRadius;
        returnFromZ = displayZ;
      }}

      autoAngleDeg = normalizeDeg(autoAngleDeg + speedDegPerSec * dt);
      displayAngleDeg = autoAngleDeg;

      if (returnStartTs !== null) {{
        const progress = Math.min(1.0, (ts - returnStartTs) / returnDurationMs);
        const eased = 1.0 - Math.pow(1.0 - progress, 2.0);
        displayRadius = returnFromRadius + (baseRadius - returnFromRadius) * eased;
        displayZ = returnFromZ + (baseZ - returnFromZ) * eased;
        if (progress >= 1.0) {{
          returnStartTs = null;
          displayRadius = baseRadius;
          displayZ = baseZ;
        }}
      }} else {{
        displayRadius = baseRadius;
        displayZ = baseZ;
      }}

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
