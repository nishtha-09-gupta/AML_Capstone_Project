"""Streamlit entry point for the verified Heartlens study companion.

THESIS: Submit a fictional profile with Predict and inspect its frozen model response.
OWN-WORLD: Cream canvas, dark brown sidebar, terracotta scores, bold sans headings, native controls.
STORY: Try values, compare models, and inspect evidence and limitations.
FIRST VIEWPORT: Left sidebar, centered warm banner, two-column profile panel, live result below.
FORM: Warm research notebook, grounded candidate 4, seed ceb51ca6.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, and DESIGN.md
"""
import streamlit as st
from study import ROOT, evidence

st.set_page_config(page_title="Heartlens · A heart disease ML study", page_icon="◒", layout="wide", initial_sidebar_state="auto")
st.html((ROOT / "assets/app.css").read_text())
try:
    evidence()
except (ValueError, OSError, KeyError):
    st.error("The verified study artifacts could not be loaded. Please report this to the project team.")
    st.stop()

page = st.navigation([
    st.Page("views/live.py", title="Prediction", default=True),
    st.Page("views/models.py", title="Model studio", url_path="models"),
    st.Page("views/phase2.py", title="Phase 2", url_path="phase-2"),
], position="hidden")
with st.sidebar:
    st.html('<div class="sidebar-brand">HeartLens</div><div class="sidebar-description">Heart disease prediction · UCI dataset</div>')
    st.page_link("views/live.py", label="Prediction")
    st.page_link("views/models.py", label="Model studio")
    st.page_link("views/phase2.py", label="Phase 2")
    st.html('<div class="sidebar-notes">UCI-derived cohort · 918 records · 4 hospitals<br><br>Educational models. No diagnostic use.</div>')
page.run()
st.divider()
st.caption("HeartLens · Academic demonstration. Use fictional inputs. This app does not provide medical advice.")
