import streamlit as st

from htp.model.physics import equilibrium_temperature_c
from htp.model.scenario_io import load_scenario_from_session, scenario_to_flat_params

st.set_page_config(page_title="Stability & Policy", layout="wide")


def render_page():
    st.markdown("<style>[data-testid='stHeaderActionElements']{display:none;}</style>", unsafe_allow_html=True)
    st.title("Stability & Policy (WIP)")
    scenario = load_scenario_from_session(st.session_state)
    params = scenario_to_flat_params(scenario)

    eq_temp_c, eq_albedo = equilibrium_temperature_c(
        stellar_flux_multiplier=float(params["stellar_flux_multiplier"]),
        warm_albedo=float(params["warm_albedo"]),
        ice_albedo=float(params["ice_albedo"]),
        co2_ppm=float(params["initial_co2_ppm"]),
        K_CO2=float(params["K_CO2"]),
    )

    st.caption("Minimal backend-wired placeholder for upcoming policy/stability tools.")
    st.markdown(
        "\n".join(
            [
                f"- Equilibrium estimate: {eq_temp_c:.2f} C",
                f"- Effective albedo: {eq_albedo:.3f}",
                f"- Natural planet mode: {'On' if bool(params['natural_planet_mode']) else 'Off'}",
                f"- Mitigation schedule: start year {int(params['mitigation_start_year'])}, strength {float(params['mitigation_strength']):.2f}",
            ]
        )
    )


render_page()
