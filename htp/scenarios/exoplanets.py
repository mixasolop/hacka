"""NASA Exoplanet Archive helpers used by the Scenario Builder."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

import numpy as np

from htp.model.safety import safe_float
from .presets import EXOPLANET_PRESET_PREFIX

EXOPLANET_ARCHIVE_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
EXOPLANET_ARCHIVE_QUERY = (
    "select pl_name,hostname,pl_insol,pl_eqt,pl_orbeccen "
    "from ps where default_flag=1 and pl_name is not null and pl_insol is not null"
)
EXOPLANET_SAMPLE_SIZE = 100


def fetch_exoplanet_rows(*, timeout: float = 20.0) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({"query": EXOPLANET_ARCHIVE_QUERY, "format": "json"})
    url = f"{EXOPLANET_ARCHIVE_TAP_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=float(timeout)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    if not isinstance(payload, list):
        return []

    rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        name = str(row.get("pl_name", "")).strip()
        if not name:
            continue
        rows.append(
            {
                "pl_name": name,
                "hostname": str(row.get("hostname", "")).strip(),
                "pl_insol": safe_float(row.get("pl_insol"), float("nan")),
                "pl_eqt": safe_float(row.get("pl_eqt"), float("nan")),
                "pl_orbeccen": safe_float(row.get("pl_orbeccen"), float("nan")),
            }
        )
    return rows


def sample_exoplanet_rows(
    rows: list[dict[str, Any]],
    *,
    sample_size: int = EXOPLANET_SAMPLE_SIZE,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    rng_obj = rng if rng is not None else np.random.default_rng()
    take = min(int(sample_size), len(rows))
    idx = rng_obj.choice(len(rows), size=take, replace=False)
    sample = [rows[int(i)] for i in idx]
    sample.sort(key=lambda item: str(item.get("pl_name", "")).lower())
    return sample


def format_exoplanet_option(row: dict[str, Any], *, preset_prefix: str = EXOPLANET_PRESET_PREFIX) -> str:
    name = str(row.get("pl_name", "Unknown")).strip() or "Unknown"
    host = str(row.get("hostname", "")).strip()
    insol = safe_float(row.get("pl_insol"), float("nan"))
    eqt_k = safe_float(row.get("pl_eqt"), float("nan"))
    ecc = safe_float(row.get("pl_orbeccen"), float("nan"))
    host_suffix = f" @ {host}" if host else ""
    insol_text = f"{insol:.2f} S_earth" if np.isfinite(insol) else "S_earth n/a"
    eqt_text = f"{eqt_k:.0f} K" if np.isfinite(eqt_k) else "Teq n/a"
    ecc_text = f"e={ecc:.2f}" if np.isfinite(ecc) else "e=n/a"
    return f"{preset_prefix}{name}{host_suffix} [{insol_text}, {eqt_text}, {ecc_text}]"


def build_exoplanet_option_map(
    sample: list[dict[str, Any]],
    *,
    preset_prefix: str = EXOPLANET_PRESET_PREFIX,
) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(sample):
        if not isinstance(row, dict):
            continue
        label = format_exoplanet_option(row, preset_prefix=preset_prefix)
        if label in mapping:
            label = f"{label} #{idx + 1}"
        mapping[label] = row
    return mapping
