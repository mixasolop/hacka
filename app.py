import streamlit as st

pages = [
    st.Page("pages/00_Scenario_Builder.py", title="Scenario Builder", default=True),
    st.Page("pages/01_Climate_Twin.py", title="Climate Twin"),
    st.Page("pages/02_Refugia_Map.py", title="Refugia Map"),
    st.Page("pages/03_Stability_and_Policy.py", title="Stability & Policy"),
]
navigation = st.navigation(pages, position="sidebar")
navigation.run()
