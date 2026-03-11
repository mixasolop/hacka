import csv
import io
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

NOAA_MAUNA_LOA_CO2_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
NASA_GISTEMP_GLOBAL_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"


def _download_text(url: str, timeout_seconds: float = 8.0):
    with urlopen(url, timeout=timeout_seconds) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _to_float(token: str):
    text = str(token).strip()
    if not text or text in {"***", "****", "*****"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_noaa_latest_co2(csv_text: str):
    latest = None
    reader = csv.reader(io.StringIO(csv_text))
    for row in reader:
        if not row:
            continue
        first = str(row[0]).strip()
        if first.startswith("#"):
            continue
        if len(row) < 5:
            continue
        try:
            year = int(str(row[0]).strip())
            month = int(str(row[1]).strip())
        except ValueError:
            continue
        co2_ppm = _to_float(row[4])
        if co2_ppm is None or co2_ppm < 0.0:
            continue
        stamp = year * 100 + month
        if latest is None or stamp > latest["stamp"]:
            latest = {
                "stamp": stamp,
                "year": year,
                "month": month,
                "co2_ppm": float(co2_ppm),
            }
    return latest


def _parse_nasa_latest_gistemp_anomaly(csv_text: str):
    latest = None
    reader = csv.reader(io.StringIO(csv_text))
    for row in reader:
        if len(row) < 13:
            continue
        first = str(row[0]).strip()
        if not first.isdigit():
            continue
        year = int(first)
        for month in range(1, 13):
            anomaly = _to_float(row[month] if month < len(row) else "")
            if anomaly is None:
                continue
            stamp = year * 100 + month
            if latest is None or stamp > latest["stamp"]:
                latest = {
                    "stamp": stamp,
                    "year": year,
                    "month": month,
                    "anomaly_c": float(anomaly),
                }
    return latest


@lru_cache(maxsize=1)
def get_observed_climate_baseline():
    baseline: dict[str, Any] = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        # Keep physically grounded defaults in case network calls fail.
        "solar_constant_w_m2": 1361.0,
        "albedo_ref": 0.30,
        "co2_baseline_ppm": 278.0,
        "co2_ppm_latest": 420.0,
        "co2_observation": None,
        "gistemp_latest_anomaly_c": None,
        "gistemp_observation": None,
        "sources": {
            "co2": "NOAA GML Mauna Loa monthly mean",
            "temperature": "NASA GISS GISTEMP v4 Land-Ocean anomaly",
        },
        "errors": [],
    }

    try:
        noaa_text = _download_text(NOAA_MAUNA_LOA_CO2_URL)
        latest_co2 = _parse_noaa_latest_co2(noaa_text)
        if latest_co2 is not None:
            baseline["co2_ppm_latest"] = float(latest_co2["co2_ppm"])
            baseline["co2_observation"] = {
                "year": int(latest_co2["year"]),
                "month": int(latest_co2["month"]),
            }
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        baseline["errors"].append(f"NOAA CO2 fetch failed: {exc}")

    try:
        gistemp_text = _download_text(NASA_GISTEMP_GLOBAL_URL)
        latest_temp = _parse_nasa_latest_gistemp_anomaly(gistemp_text)
        if latest_temp is not None:
            baseline["gistemp_latest_anomaly_c"] = float(latest_temp["anomaly_c"])
            baseline["gistemp_observation"] = {
                "year": int(latest_temp["year"]),
                "month": int(latest_temp["month"]),
            }
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        baseline["errors"].append(f"NASA GISTEMP fetch failed: {exc}")

    return baseline
