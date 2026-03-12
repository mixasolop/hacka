import numpy as np

from htp.model.imports import normalize_imported_planet
from htp.model.scenario_io import export_scenario_json, import_scenario_json
from htp.model.schema import CivilizationState, PlanetState, ScenarioModel
from htp.model.simulate import preview_state, simulate_time_series


def _assert_series_health(series: dict):
    assert np.all(np.isfinite(series["global_temperature_c"]))
    assert np.all(np.isfinite(series["co2_ppm"]))
    assert np.all(np.isfinite(series["habitable_surface_percent"]))
    assert np.all(np.isfinite(series["effective_albedo"]))
    assert float(np.min(series["co2_ppm"])) >= 1.0
    assert float(np.max(series["co2_ppm"])) <= 100000.0
    assert float(np.min(series["effective_albedo"])) >= 0.0
    assert float(np.max(series["effective_albedo"])) <= 1.0
    assert float(np.min(series["habitable_surface_percent"])) >= 0.0
    assert float(np.max(series["habitable_surface_percent"])) <= 100.0


def test_earthlike_baseline_500y_no_nan_inf():
    scenario = ScenarioModel()
    series = simulate_time_series(scenario, years=500, dt_years=1.0)
    _assert_series_health(series)


def test_very_low_co2_stable_run():
    scenario = ScenarioModel(
        planet=PlanetState(initial_co2_ppm=1.0),
        civilization=CivilizationState(),
    )
    series = simulate_time_series(scenario, years=500, dt_years=1.0)
    _assert_series_health(series)


def test_very_high_co2_stable_run():
    scenario = ScenarioModel(
        planet=PlanetState(initial_co2_ppm=100000.0),
        civilization=CivilizationState(),
    )
    series = simulate_time_series(scenario, years=500, dt_years=1.0)
    _assert_series_health(series)


def test_hot_imported_natural_mode_emissions_off():
    scenario = normalize_imported_planet(
        {
            "pl_name": "Hot Test",
            "pl_insol": 50.0,
            "pl_eqt": 1500.0,
            "pl_orbeccen": 0.2,
        },
        atmosphere_assumption="Minimal",
    )
    series = simulate_time_series(scenario, years=500, dt_years=1.0)
    _assert_series_health(series)
    assert np.allclose(series["human_emissions_eff"], 0.0)


def test_hot_imported_natural_mode_no_large_initial_temperature_spike():
    scenario = normalize_imported_planet(
        {
            "pl_name": "Hot Natural Anchor",
            "pl_insol": 8.6,
            "pl_eqt": 464.0,
            "pl_orbeccen": 0.1,
        },
        atmosphere_assumption="Minimal",
    )
    series = simulate_time_series(scenario, years=50, dt_years=1.0)
    _assert_series_health(series)
    temps = np.asarray(series["global_temperature_c"], dtype=float)
    co2 = np.asarray(series["co2_ppm"], dtype=float)
    assert np.isclose(temps[0], 464.0 - 273.15, atol=1e-6)
    assert abs(float(temps[1] - temps[0])) < 15.0
    assert float(np.max(co2)) <= 1.0 + 1e-9
    assert float(np.min(co2)) >= 1.0


def test_cold_imported_natural_mode_stable():
    scenario = normalize_imported_planet(
        {
            "pl_name": "Cold Test",
            "pl_insol": 0.05,
            "pl_eqt": 100.0,
            "pl_orbeccen": 0.0,
        },
        atmosphere_assumption="Minimal",
    )
    series = simulate_time_series(scenario, years=500, dt_years=1.0)
    _assert_series_health(series)


def test_year0_thresholds_are_initial_conditions_not_events():
    scenario = normalize_imported_planet(
        {
            "pl_name": "Extreme Hot Start",
            "pl_insol": 80.0,
            "pl_eqt": 1500.0,
            "pl_orbeccen": 0.0,
        },
        atmosphere_assumption="Minimal",
    )
    series = simulate_time_series(scenario, years=100, dt_years=1.0)
    labels = [item["label"] for item in series["initial_conditions"]]
    assert any("Initial condition" in label for label in labels)
    assert all(int(event["year"]) > 0 for event in series["events"])


def test_export_import_export_roundtrip_preserves_extremes_and_flags():
    scenario = ScenarioModel(
        planet=PlanetState(
            stellar_flux_multiplier=999.9,
            initial_co2_ppm=100000.0,
            seed=2147483647,
            natural_planet_mode=True,
            atmosphere_assumption="Dense",
            imported_equilibrium_temperature_k=1500.0,
            imported_stellar_flux_multiplier=999.9,
            import_classification="hot",
        ),
        civilization=CivilizationState(
            emissions_rate=99.9,
            emissions_growth_mode="Carefree",
            mitigation_start_year=49999,
            mitigation_strength=1.0,
        ),
    )
    text1 = export_scenario_json(scenario)
    restored = import_scenario_json(text1)
    text2 = export_scenario_json(restored)
    assert text1 == text2
    assert restored.planet.seed == 2147483647
    assert restored.planet.natural_planet_mode is True
    assert restored.planet.atmosphere_assumption == "Dense"
    assert restored.planet.initial_co2_ppm == 100000.0


def test_preview_and_simulation_year0_match():
    scenario = normalize_imported_planet(
        {
            "pl_name": "Warm Start",
            "pl_insol": 2.0,
            "pl_eqt": 450.0,
            "pl_orbeccen": 0.1,
        },
        atmosphere_assumption="Earth-like",
    )
    preview = preview_state(scenario, horizon_years=100)
    series = simulate_time_series(scenario, years=100, dt_years=1.0)
    assert np.isclose(float(preview["T0_c"]), float(series["meta"]["T0_c"]))
    assert np.isclose(float(preview["co2_ppm"]), float(series["co2_ppm"][0]))
