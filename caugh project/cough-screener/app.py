from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import LABELS
from src.infer import predict_bytes

st.set_page_config(page_title="Cough Screener", page_icon="CS", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #17231f; --muted: #6c7b75; --mint: #b8f2d0; --lime: #d7f26a; --paper: #f7faf4; --line: #dce6df; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #eef6ee; border-right: 1px solid var(--line); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; color: var(--ink); }
    p, label, .stMarkdown, .stCaption { font-family: 'DM Sans', sans-serif; }
    .hero { background: #17352b; color: #f8fff7; padding: 2rem 2.2rem; border-radius: 18px; margin-bottom: 1.2rem; }
    .hero h1 { color: #f8fff7; font-size: 2.8rem; margin: 0; }
    .hero p { color: #c9ddd0; max-width: 700px; margin: .5rem 0 0; font-size: 1.05rem; }
    .eyebrow { color: var(--lime); font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .stat { border: 1px solid var(--line); border-radius: 14px; padding: 1rem; background: white; min-height: 105px; }
    .stat-label { color: var(--muted); font-size: .8rem; }
    .stat-value { color: var(--ink); font-family: 'Space Grotesk'; font-size: 1.65rem; font-weight: 700; margin-top: .25rem; }
    .result { background: var(--mint); border-radius: 14px; padding: 1.2rem; }
    .privacy { background: #fff8df; border: 1px solid #f0df9a; border-radius: 10px; padding: .75rem 1rem; color: #5d4d1b; font-size: .9rem; }
    div.stButton > button[kind="primary"] { background: #17352b; border: 0; color: white; border-radius: 9px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def stat_card(label: str, value: str) -> None:
    st.markdown(f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>', unsafe_allow_html=True)


def history_frame() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.history)

with st.sidebar:
    st.markdown("## Cough Screener")
    st.caption("Research workspace")
    st.markdown("---")
    st.markdown("### Session status")
    stat_card("Assessments this session", str(len(st.session_state.history)))
    st.markdown("<div class='privacy'>Patient history stays in this browser session and is cleared when the app restarts.</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("This tool is for research and education. It is not a diagnosis.")

st.markdown('<div class="hero"><div class="eyebrow">Respiratory audio research</div><h1>Listen carefully.<br>Record responsibly.</h1><p>A focused workspace for organizing cough recordings, patient context, and experimental model estimates in one place.</p></div>', unsafe_allow_html=True)

overview, assessment, patients = st.tabs(["Overview", "New assessment", "Patient history"])

with overview:
    st.subheader("Session overview")
    first, second, third, fourth = st.columns(4)
    with first:
        stat_card("Patient records", str(len({item["patient_id"] for item in st.session_state.history})))
    with second:
        stat_card("Assessments", str(len(st.session_state.history)))
    with third:
        stat_card("Latest estimate", st.session_state.history[-1]["estimate"] if st.session_state.history else "--")
    with fourth:
        stat_card("Model status", "Checkpoint" if Path("artifacts/cough_cnn.pt").exists() else "Demo mode")
    st.markdown("### Workflow")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("**1. Create a patient profile**  \\n+Capture only the context you need for your research session.")
        st.markdown("**2. Add an audio sample**  \\n+Upload a file or record directly from a microphone.")
        st.markdown("**3. Review, then export**  \\n+Compare assessments without treating model output as clinical advice.")
    with right:
        st.info("Start from the New assessment tab. The app stores records in session memory only; no database or cloud upload is configured.")

with assessment:
    st.subheader("New assessment")
    st.caption("Use a study ID instead of a full name whenever possible.")
    with st.form("patient_form"):
        profile_left, profile_right = st.columns(2)
        with profile_left:
            patient_id = st.text_input("Patient / study ID *", placeholder="e.g. P-014")
            patient_name = st.text_input("Name or alias", placeholder="Optional")
            age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
            gender = st.selectbox("Gender", ["Prefer not to say", "Female", "Male", "Non-binary", "Other"])
        with profile_right:
            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=70.0, step=0.5)
            height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=170.0, step=0.5)
            smoking = st.selectbox("Smoking status", ["Unknown", "Never", "Former", "Current"])
            symptoms = st.multiselect("Reported symptoms", ["Cough", "Fever", "Shortness of breath", "Wheeze", "Sore throat", "Fatigue"])
        notes = st.text_area("Research notes", placeholder="Duration, context, or recording conditions. Avoid sensitive identifiers.")
        submitted = st.form_submit_button("Save profile and continue", type="primary", use_container_width=True)
    if submitted:
        if not patient_id.strip():
            st.error("Add a patient or study ID before recording an assessment.")
        else:
            st.session_state.active_patient = {"patient_id": patient_id.strip(), "name": patient_name.strip() or "Alias not provided", "age": int(age), "gender": gender, "weight_kg": float(weight), "height_cm": float(height), "smoking": smoking, "symptoms": ", ".join(symptoms) or "None recorded", "notes": notes.strip()}
            st.success("Profile ready. Add the audio sample below.")

    profile = st.session_state.get("active_patient")
    if profile:
        st.markdown(f"### Recording for {profile['patient_id']}")
        source = st.radio("Audio source", ["Upload recording", "Record in browser"], horizontal=True, key="audio_source")
        if source == "Upload recording":
            audio = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"], key="audio_file")
        else:
            audio = st.audio_input("Record a cough", key="audio_recording")
        checkpoint = st.text_input("Checkpoint path", value="artifacts/cough_cnn.pt", key="checkpoint")
        if audio is not None:
            st.audio(audio)
            if st.button("Analyze and save assessment", type="primary", use_container_width=True):
                with st.spinner("Detecting cough segments and estimating probabilities..."):
                    result = predict_bytes(audio.getvalue(), checkpoint if Path(checkpoint).exists() else None)
                entry = {**profile, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "estimate": result.label, "segments": len(result.segments), "confidence": round(float(result.probabilities.max()), 4)}
                st.session_state.history.append(entry)
                st.session_state.last_result = result
                st.success(f"Assessment saved for {profile['patient_id']}.")
        if st.session_state.last_result is not None:
            result = st.session_state.last_result
            result_col, chart_col = st.columns([1, 1.4])
            with result_col:
                st.markdown(f'<div class="result"><div class="eyebrow">Experimental estimate</div><h2>{result.label.title()}</h2><p>{len(result.segments)} cough-like segment(s) detected.</p></div>', unsafe_allow_html=True)
            with chart_col:
                st.bar_chart({LABELS[index]: float(value) for index, value in enumerate(result.probabilities)}, color="#5a9d73")
            st.caption("Model probabilities are estimates, not medical probabilities or diagnoses.")
            if result.segments:
                st.dataframe({"Start (s)": [round(item.start, 2) for item in result.segments], "End (s)": [round(item.end, 2) for item in result.segments]}, use_container_width=True, hide_index=True)
            else:
                st.info("No cough-like segments were detected. Try a clearer recording.")

with patients:
    st.subheader("Patient history")
    st.caption("History is session-only. Review local privacy requirements before entering identifiable information.")
    if st.session_state.history:
        frame = history_frame()
        filter_id = st.text_input("Filter by patient / study ID", key="history_filter")
        visible = frame[frame["patient_id"].str.contains(filter_id, case=False, na=False)] if filter_id else frame
        st.dataframe(visible[["timestamp", "patient_id", "name", "age", "gender", "weight_kg", "smoking", "estimate", "confidence", "segments"]], use_container_width=True, hide_index=True)
        csv = visible.to_csv(index=False).encode("utf-8")
        st.download_button("Download visible history (CSV)", csv, "cough-screener-history.csv", "text/csv")
        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()
    else:
        st.info("No saved assessments yet. Create a profile to start a patient history.")
