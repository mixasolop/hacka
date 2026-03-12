"""Run NASA exoplanet samples through the model and compute mean correctness."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import statistics
from pathlib import Path
from types import ModuleType
from typing import Any

from dotenv import load_dotenv

from htp.model.constants import SIM_DT_YEARS_DEFAULT, SIM_YEARS_DEFAULT
from htp.model.imports import normalize_imported_planet
from htp.model.scenario_io import scenario_to_flat_params
from htp.model.simulate import simulate_time_series
from htp.scenarios.exoplanets import fetch_exoplanet_rows


def _load_stability_module() -> ModuleType:
    module_path = Path(__file__).resolve().parent / "pages" / "03_Stability_and_Policy.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find page module: {module_path}")

    spec = importlib.util.spec_from_file_location("stability_policy_page", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create import spec for 03_Stability_and_Policy.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required_names = ("_safe_float", "_build_planet_inputs", "_run_realism_pipeline")
    missing = [name for name in required_names if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Missing required functions in 03_Stability_and_Policy.py: {', '.join(missing)}")
    return module


def _build_model_output(series: dict[str, Any], stability_module: ModuleType) -> dict[str, Any]:
    safe_float = stability_module._safe_float
    temp_series = list(series.get("global_temperature_k", []))
    co2_series = list(series.get("co2_ppm", []))
    hab_series = list(series.get("habitable_surface_percent", []))
    regime_series = list(series.get("regime", []))
    if not temp_series or not co2_series or not hab_series:
        raise ValueError("Simulation output is incomplete.")

    last_idx = len(temp_series) - 1
    return {
        "final_temp_k": safe_float(temp_series[last_idx], 0.0),
        "final_co2_ppm": safe_float(co2_series[last_idx], 0.0),
        "habitability_percent": safe_float(hab_series[last_idx], 0.0),
        "stability": str(regime_series[last_idx]) if regime_series else "Unknown",
    }


def _choose_rows(rows: list[dict[str, Any]], sample_count: int, seed: int) -> list[dict[str, Any]]:
    unique_rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("pl_name", "")).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        unique_rows.append(dict(row))

    rng = random.Random(seed)
    if len(unique_rows) <= sample_count:
        rng.shuffle(unique_rows)
        return unique_rows
    return rng.sample(unique_rows, k=sample_count)


def _run_one(stability_module: ModuleType, row: dict[str, Any], atmosphere_assumption: str) -> dict[str, Any]:
    scenario = normalize_imported_planet(dict(row), atmosphere_assumption=atmosphere_assumption)
    flat_params = scenario_to_flat_params(scenario)
    planet_inputs = stability_module._build_planet_inputs(flat_params)
    series = simulate_time_series(scenario, years=SIM_YEARS_DEFAULT, dt_years=SIM_DT_YEARS_DEFAULT)
    model_output = _build_model_output(series, stability_module)

    result = stability_module._run_realism_pipeline(planet_inputs, model_output)
    return {
        "planet_name": str(row.get("pl_name", "")),
        "host_name": str(row.get("hostname", "")),
        "planet_inputs": planet_inputs,
        "model_output": model_output,
        "physics_score": result.get("physics_score"),
        "ai_realism_score": result.get("ai_realism_score"),
        "confidence": result.get("confidence"),
        "final_score": result.get("final_score"),
        "temperature_realism": result.get("temperature_realism"),
        "co2_realism": result.get("co2_realism"),
        "habitability_realism": result.get("habitability_realism"),
        "explanation": result.get("explanation"),
        "error": result.get("error", ""),
    }


def _write_outputs(output_dir: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "sample_results.jsonl"
    summary_path = output_dir / "summary.json"

    with results_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, sort_keys=True))
            handle.write("\n")

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return results_path, summary_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute mean chat-evaluated correctness over NASA exoplanet samples.")
    parser.add_argument("--samples", type=int, default=200, help="How many unique exoplanet rows to evaluate.")
    parser.add_argument("--seed", type=int, default=20260312, help="Sampling seed for reproducibility.")
    parser.add_argument(
        "--atmosphere",
        type=str,
        default="Minimal",
        choices=["Minimal", "Earth-like", "Dense"],
        help="Atmosphere assumption used when normalizing imported exoplanets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "average_chat_percentage",
        help="Directory where per-sample and summary files are saved.",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="NASA API timeout in seconds.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        load_dotenv("token.env")
        api_key = os.getenv("AI")
        if not api_key:
            raise RuntimeError("Missing AI key in token.env (AI=...).")

        stability_module = _load_stability_module()
        model_name = str(getattr(stability_module, "OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")))
        rows = fetch_exoplanet_rows(timeout=float(args.timeout))
        if not rows:
            raise RuntimeError("No exoplanet rows returned from NASA API.")

        chosen_rows = _choose_rows(rows, sample_count=max(1, int(args.samples)), seed=int(args.seed))
        total = len(chosen_rows)
        print(f"Fetched {len(rows)} rows from NASA API. Running {total} samples.")

        results: list[dict[str, Any]] = []
        failed = 0
        for idx, row in enumerate(chosen_rows, start=1):
            try:
                item = _run_one(
                    stability_module=stability_module,
                    row=row,
                    atmosphere_assumption=str(args.atmosphere),
                )
                if item.get("error"):
                    failed += 1
                    print(f"[{idx}/{total}] {item['planet_name']}: error={item['error']}")
                else:
                    print(f"[{idx}/{total}] {item['planet_name']}: final_score={float(item['final_score']):.2f}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                item = {
                    "planet_name": str(row.get("pl_name", "")),
                    "host_name": str(row.get("hostname", "")),
                    "error": str(exc),
                }
                print(f"[{idx}/{total}] {item['planet_name']}: error={exc}")
            results.append(item)

        final_scores = [
            float(item["final_score"]) for item in results if isinstance(item.get("final_score"), (int, float))
        ]
        ai_scores = [
            float(item["ai_realism_score"]) for item in results if isinstance(item.get("ai_realism_score"), (int, float))
        ]
        physics_scores = [
            float(item["physics_score"]) for item in results if isinstance(item.get("physics_score"), (int, float))
        ]

        mean_correctness = statistics.fmean(final_scores) if final_scores else 0.0
        summary = {
            "requested_samples": int(args.samples),
            "processed_samples": int(total),
            "successful_samples": int(len(final_scores)),
            "failed_samples": int(failed),
            "mean_correctness": mean_correctness,
            "mean_ai_realism_score": statistics.fmean(ai_scores) if ai_scores else 0.0,
            "mean_physics_score": statistics.fmean(physics_scores) if physics_scores else 0.0,
            "model": model_name,
            "atmosphere": str(args.atmosphere),
        }

        results_path, summary_path = _write_outputs(Path(args.output_dir), results, summary)
        print("\nRun complete.")
        print(f"Mean correctness: {mean_correctness:.2f}%")
        print(f"Per-sample results: {results_path}")
        print(f"Summary: {summary_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
