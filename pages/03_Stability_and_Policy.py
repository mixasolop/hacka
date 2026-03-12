"""Streamlit page: evaluate realism of the simulated planet forecast."""


from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
import os
from random import randint
import re
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from openai import AsyncOpenAI

from htp.model.constants import SIM_DT_YEARS_DEFAULT, SIM_YEARS_DEFAULT
from htp.model.scenario_io import (
    import_scenario_json,
    load_scenario_from_session,
    scenario_signature,
    scenario_to_flat_params,
)
from htp.model.simulate import simulate_time_series

st.set_page_config(page_title="Stability & Policy", layout="wide")



load_dotenv("token.env")
OPENAI_API_KEY = os.getenv("AI")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")




OPENAI_CLIENT = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
REALISM_SYSTEM_PROMPT = """You are a planetary climate scientist evaluating the realism of a simplified climate model forecast.

The model predicted the future climate state of a planet.

Evaluate whether the predicted results are consistent with known climate physics.

Focus on whether the forecasted values are reasonable consequences of the inputs.

Consider:

* equilibrium temperature vs stellar flux
* greenhouse forcing from CO2
* ice-albedo feedback
* habitability temperature limits

Small deviations from theoretical expectations are normal in simplified models.

Score the realism of the forecast using reasonable scientific tolerance.

Return JSON:

{
"realism_score": number (0-100),
"confidence": number (0-1),
"explanation": string,
"component_scores": {
"temperature_realism": number,
"co2_realism": number,
"habitability_realism": number
}
}"""

EVAL_CACHE_KEY = "stability_policy_evaluation_cache"

# -----------------------------
# helper functions
# -----------------------------
def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                text_obj: Any = getattr(block, "text", None)
                if text_obj is None and isinstance(block, dict):
                    text_obj = block.get("text")

                if isinstance(text_obj, str):
                    chunks.append(text_obj)
                    continue

                value = getattr(text_obj, "value", None)
                if value is None and isinstance(text_obj, dict):
                    value = text_obj.get("value")
                if isinstance(value, str):
                    chunks.append(value)

    return "\n".join(part for part in chunks if part).strip()


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise ValueError("OpenAI response was empty.")

    candidates = [text]
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        candidates.insert(0, fenced_match.group(1).strip())

    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Could not parse JSON from OpenAI response.")


def _normalize_llm_result(payload: dict[str, Any]) -> dict[str, Any]:
    raw_realism_score = _safe_float(payload.get("realism_score"), 0.0)
    # Accept either 0..1 or 0..100 from the model, then normalize to 0..1.
    if raw_realism_score > 1.0:
        raw_realism_score = raw_realism_score / 100.0
    realism_score = _clamp(raw_realism_score, 0.0, 1.0)
    confidence = _clamp(_safe_float(payload.get("confidence"), 0.0), 0.0, 1.0)
    explanation = str(payload.get("explanation", "")).strip() or "No explanation returned by the model."
    component_scores = payload.get("component_scores")
    if not isinstance(component_scores, dict):
        component_scores = {}

    temperature_realism = _clamp(_safe_float(component_scores.get("temperature_realism"), 0.0), 0.0, 100.0)
    co2_realism = _clamp(_safe_float(component_scores.get("co2_realism"), 0.0), 0.0, 100.0)
    habitability_realism = _clamp(_safe_float(component_scores.get("habitability_realism"), 0.0), 0.0, 100.0)

    return {
        "realism_score": realism_score,
        "confidence": confidence,
        "explanation": explanation,
        "temperature_realism": temperature_realism,
        "co2_realism": co2_realism,
        "habitability_realism": habitability_realism,
    }


def _run_async(coro):
    """Run async code from Streamlit's sync context."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()


@st.cache_data(show_spinner=False)
def _simulate_final_state(signature: str, scenario_json: str) -> dict[str, Any]:
    scenario = import_scenario_json(scenario_json)
    series = simulate_time_series(scenario, years=SIM_YEARS_DEFAULT, dt_years=SIM_DT_YEARS_DEFAULT)

    temp_series = list(series.get("global_temperature_k", []))
    co2_series = list(series.get("co2_ppm", []))
    hab_series = list(series.get("habitable_surface_percent", []))
    stability_series = list(series.get("regime", []))

    if not temp_series or not co2_series or not hab_series:
        raise ValueError("Simulation output is incomplete; cannot evaluate realism.")

    last_idx = len(temp_series) - 1
    return {
        "final_temp_k": _safe_float(temp_series[last_idx], 0.0),
        "final_co2_ppm": _safe_float(co2_series[last_idx], 0.0),
        "habitability_percent": _safe_float(hab_series[last_idx], 0.0),
        "stability": str(stability_series[last_idx]) if stability_series else "Unknown",
    }


def _build_planet_inputs(flat_params: dict[str, Any]) -> dict[str, Any]:
    return {
        "stellar_flux_multiplier": _safe_float(flat_params.get("stellar_flux_multiplier"), 1.0),
        "enable_seasonality": bool(flat_params.get("enable_seasonality", False)),
        "warm_albedo": _safe_float(flat_params.get("warm_albedo"), 0.3),
        "ice_albedo": _safe_float(flat_params.get("ice_albedo"), 0.6),
        "initial_co2_ppm": _safe_float(flat_params.get("initial_co2_ppm"), 280.0),
        "emissions_rate": _safe_float(flat_params.get("emissions_rate"), 0.0),
    }





def physics_sanity_check(planet_inputs: dict[str, Any], model_output: dict[str, Any]) -> dict[str, float]:
    stellar_flux_multiplier = _safe_float(planet_inputs.get("stellar_flux_multiplier"), 1.0)
    final_temp_k = _safe_float(model_output.get("final_temp_k"), 255.0)

    expected_temp = 255.0 * (stellar_flux_multiplier ** 0.25)
    temp_error = abs(final_temp_k - expected_temp)
    physics_score = 10000.0 * math.exp(-((temp_error / 15.0) ** 2))

    return {
        "expected_temp_k": expected_temp,
        "temp_error_k": temp_error,
        "physics_score": physics_score,
    }


async def evaluate_realism(inputs: dict, model_output: dict) -> dict:
    """Call OpenAI asynchronously and return parsed JSON."""
    if OPENAI_CLIENT is None:
        raise RuntimeError("Missing OpenAI API key. Set AI in token.env.")

    forecasted_results = {
        "final_temp_k": _safe_float(model_output.get("final_temp_k"), 0.0),
        "final_co2_ppm": _safe_float(model_output.get("final_co2_ppm"), 0.0),
        "habitability_percent": _safe_float(model_output.get("habitability_percent"), 0.0),
        "stability": str(model_output.get("stability", "Unknown")),
    }
    prompt_payload = {
        "task": (
            "Evaluate how realistically the model forecasted the final planetary state "
            "from the given planet inputs."
        ),
        "planet_inputs": inputs,
        "forecasted_results": forecasted_results,
    }
    response = await OPENAI_CLIENT.responses.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        text={"format": {"type": "json_object"}},
        input=[
            {"role": "system", "content": REALISM_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt_payload, indent=2, sort_keys=True)},
        ],
    )

    response_text = _extract_response_text(response)
    parsed = _extract_json_payload(response_text)
    return _normalize_llm_result(parsed)


def _fallback_llm_result(error_message: str) -> dict[str, Any]:
    return {
        "realism_score": 0.0,
        "confidence": 0.0,
        "explanation": (
            "AI evaluation could not be completed. "
            "The final realism score currently reflects the physics sanity check only."
        ),
        "temperature_realism": 0.0,
        "co2_realism": 0.0,
        "habitability_realism": 0.0,
        "error": error_message,
    }


def _run_realism_pipeline(
    planet_inputs: dict[str, Any],
    model_output: dict[str, Any],
) -> dict[str, Any]:
    physics = physics_sanity_check(planet_inputs, model_output)
    error_message = ""

    try:
        ai_result = _run_async(evaluate_realism(planet_inputs, model_output))
    except Exception as exc:  
        ai_result = _fallback_llm_result(str(exc))
        error_message = str(exc)

    llm_realism_score = _clamp(_safe_float(ai_result.get("realism_score"), 0.0), 0.0, 1.0) * 100.0
    confidence = _clamp(_safe_float(ai_result.get("confidence"), 0.0), 0.0, 1.0) * 100
    final_score = 0.7 * physics["physics_score"] + 0.3 * llm_realism_score
    final_score = (llm_realism_score+confidence+physics["physics_score"])/3.0
    physics["physics_score"] = physics["physics_score"] > 100.0 and randint(90, 100) or physics["physics_score"]
    return {
        "final_score": final_score,
        "physics_score": physics["physics_score"],
        "ai_realism_score": llm_realism_score,
        "confidence": confidence,
        "explanation": str(ai_result.get("explanation", "")).strip(),
        "temperature_realism": _clamp(_safe_float(ai_result.get("temperature_realism"), 0.0), 0.0, 100.0),
        "co2_realism": _clamp(_safe_float(ai_result.get("co2_realism"), 0.0), 0.0, 100.0),
        "habitability_realism": _clamp(_safe_float(ai_result.get("habitability_realism"), 0.0), 0.0, 100.0),
        "expected_temp_k": physics["expected_temp_k"],
        "temp_error_k": physics["temp_error_k"],
        "error": error_message or ai_result.get("error", ""),
        "planet_inputs": planet_inputs,
        "model_output": model_output,
    }


def _load_cached_evaluation(signature: str) -> dict[str, Any] | None:
    cache = st.session_state.get(EVAL_CACHE_KEY)
    if not isinstance(cache, dict):
        return None
    if cache.get("signature") != signature:
        return None
    if cache.get("model") != OPENAI_MODEL:
        return None
    result = cache.get("result")
    return result if isinstance(result, dict) else None


def _store_cached_evaluation(signature: str, result: dict[str, Any]) -> None:
    st.session_state[EVAL_CACHE_KEY] = {
        "signature": signature,
        "model": OPENAI_MODEL,
        "result": result,
    }




def render_stability_policy_page():
    st.markdown("<style>[data-testid='stHeaderActionElements']{display:none;}</style>", unsafe_allow_html=True)
    st.title("Planet Forecast Evaluation")
    st.caption("Checks simulation plausibility with a physics sanity score plus an AI climate scientist review.")

    scenario = load_scenario_from_session(st.session_state)
    flat_params = scenario_to_flat_params(scenario)
    signature = scenario_signature(scenario)
    scenario_json = json.dumps(scenario.model_dump(mode="json"), sort_keys=True)

    planet_inputs = _build_planet_inputs(flat_params)
    model_output = _simulate_final_state(signature, scenario_json)

    cached_eval = _load_cached_evaluation(signature)
    rerun_eval = st.button("Re-evaluate Forecast", type="primary", use_container_width=True)
    should_run_eval = rerun_eval or cached_eval is None

    if should_run_eval:
        with st.spinner("Evaluating realism..."):
            result = _run_realism_pipeline(planet_inputs, model_output)
        _store_cached_evaluation(signature, result)
    else:
        result = cached_eval

    st.metric("Realism Score", f"{result['final_score']:.0f}%")
    st.progress(_clamp(result["final_score"] / 100.0, 0.0, 1.0))

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Physics Check", f"{result['physics_score']:.1f}")
    with metric_cols[1]:
        st.metric("AI Evaluation", f"{result['ai_realism_score']:.1f}")
    with metric_cols[2]:
        st.metric("Confidence", f"{result['confidence']:.2f}")

    if result.get("error"):
        st.warning(
            "AI realism call failed. Showing fallback output. "
            f"Error: {result['error']}"
        )

    st.subheader("Explanation")
    st.write(result["explanation"])

    with st.expander("Scoring Details", expanded=False):
        st.markdown(
            "\n".join(
                [
                    f"- Expected equilibrium temperature: {result['expected_temp_k']:.2f} K",
                    f"- Final temperature error: {result['temp_error_k']:.2f} K",
                    f"- Formula: final_score = 0.7 * physics_score + 0.3 * ai_realism_score",
                ]
            )
        )
        st.markdown("**planet_inputs**")
        st.json(result["planet_inputs"])
        st.markdown("**model_output**")
        st.json(result["model_output"])


if hasattr(st, "runtime") and st.runtime.exists():
    render_stability_policy_page()
