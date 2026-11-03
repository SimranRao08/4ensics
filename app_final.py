import os
import cv2
import hashlib
import numpy as np
import streamlit as st
from PIL import Image

from risk_engine import compute_risk

st.set_page_config(page_title="4ensics — Digital Evidence Triage", page_icon="🔍", layout="wide")

st.title("🔍 4ensics — AI-Assisted Digital Evidence Triage & Decision Support")
st.caption(
    "Analyzes incoming digital evidence and produces a forensic risk "
    "assessment to help prioritize human analyst review — it does not "
    "declare evidence 'real' or 'fake'. Final forensic and legal "
    "determinations remain with the human analyst."
)

# ============================================================
# STEP 1: Choose model BEFORE uploading
# ============================================================
MODEL_FILES = [
    "forensics_model.joblib",
    "forensics_model_whatsapp.joblib",
    "forensics_model_v4.joblib",
    "forensics_model_unified_rf.joblib",
    "forensics_model_unified_gb.joblib",
]

available_models = [p for p in MODEL_FILES if os.path.exists(p)]
if not available_models:
    st.error("No trained model files found. Place at least one .joblib model in this folder.")
    st.stop()

st.subheader("⚙️ Choose which model drives the primary result")
model_choice = st.radio(
    "Model",
    options=["All models combined (recommended)"] + available_models,
    index=0,
    horizontal=False,
)
if model_choice != "All models combined (recommended)":
    st.warning(
        "⚠️ You're selecting a single model to display prominently. We "
        # "tested this directly: no single model was reliably more "
        # "accurate across our test set — each one got some photos right "
        # "and others confidently wrong. The other models' results are "
        # "still shown below for comparison, and the combined Risk Score "
        # "(used for Priority Level) always reflects ALL models together, "
        # "regardless of which one you choose to feature here."
    )
model_override = None if model_choice == "All models combined (recommended)" else model_choice

st.divider()

# ============================================================
# STEP 2: Upload
# ============================================================
uploaded_file = st.file_uploader(
    "Upload evidence for analysis", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    temp_path = f"_temp_{uploaded_file.name}"
    file_bytes = uploaded_file.getbuffer()
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    sha256_hash = hashlib.sha256(bytes(file_bytes)).hexdigest()
    pil_img = Image.open(temp_path)
    width, height = pil_img.size
    file_size_kb = len(file_bytes) / 1024
    color_mode = pil_img.mode

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Evidence Image")
        st.image(pil_img, use_container_width=True)

        with st.expander("📋 Image Properties & Evidence Integrity"):
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Dimensions:** {width} × {height} px")
            st.write(f"**File size:** {file_size_kb:.1f} KB")
            st.write(f"**Color mode:** {color_mode}")
            st.code(f"SHA-256: {sha256_hash}", language=None)
            st.caption("Recompute this hash anytime to verify the evidence file has not been altered since upload.")

    with st.spinner("Running forensic risk assessment..."):
        result = compute_risk(temp_path, model_override=model_override)

    with col2:
        st.subheader("Forensic Risk Assessment")

        priority_colors = {"Low": "success", "Medium": "warning", "High": "error"}
        priority = result["priority_level"]
        getattr(st, priority_colors[priority])(f"**Priority Level: {priority}**")

        # ---- PRIMARY RESULT — big and prominent ----
        if model_override:
            featured_pct = result["individual_model_results"][model_override]["prob_fake"] * 100
            st.markdown(f"### Featured Model: `{model_override}`")
            st.markdown(
                f"<h1 style='font-size:64px; margin:0;'>{featured_pct:.1f}%</h1>"
                f"<p style='color:gray; margin-top:0;'>risk-leaning (this model only)</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("### Combined Risk Score (all models)")
            st.markdown(
                f"<h1 style='font-size:64px; margin:0;'>{result['risk_score']}/100</h1>",
                unsafe_allow_html=True,
            )
        st.progress(result["risk_score"] / 100)
        st.write(f"**Recommendation:** {result['recommendation']}")

        st.divider()
        st.write("**Forensic Indicators Detected:**")
        for indicator in result["forensic_indicators"]:
            st.write(f"• {indicator}")

        # ---- SECONDARY: all other models, shown smaller ----
        st.divider()
        st.markdown("##### Other model results (for comparison)")
        domains = result["model_training_domains"]
        for path, r in result["individual_model_results"].items():
            if model_override and path == model_override:
                continue  # already shown big above
            domain = domains.get(path, {})
            st.caption(f"**{path}** — {r['prob_fake']*100:.1f}% risk-leaning · _{domain.get('best_matches', '')}_")

        if not model_override:
            st.caption(
                "Combined Risk Score above always uses ALL models together — "
                "no single model has proven reliable enough to trust alone "
                "(tested directly during development)."
            )

        with st.expander("Metadata context (independent signal, not seen by ML models)"):
            st.write(result["exif_context"]["note"])

    st.divider()
    st.subheader("Supporting Forensic Evidence")
    st.caption("Presented for analyst review — not a standalone verdict.")

    from ela_fft_features_v2 import compute_ela_map, compute_fft_map
    ela_map, ela_mean, ela_var = compute_ela_map(temp_path, quality=95)
    gray = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    fft_map, peak_ratio, energy_var = compute_fft_map(gray)

    ev1, ev2 = st.columns(2)
    with ev1:
        st.image(ela_map, caption="ELA Heatmap (Q95) — edges/detail naturally light up in ANY photo; look for localized inconsistency, not the outline itself", use_container_width=True, clamp=True)
    with ev2:
        fft_display = (fft_map - fft_map.min()) / (fft_map.max() - fft_map.min() + 1e-8)
        st.image(fft_display, caption="FFT Magnitude Spectrum — the center cross is normal for any image; look for regular grid texture radiating outward", use_container_width=True, clamp=True)

    os.remove(temp_path)

else:
    st.info("Upload evidence above to run the forensic risk assessment.")

st.divider()
with st.expander("About this system's design philosophy"):
    st.markdown(
        """
        This system is designed as **decision support, not a decision-maker**.
        No AI model here declares evidence "fake" or "real" — that
        determination requires human forensic and legal judgment.

        Instead, the system:
        - Runs multiple forensic models trained on different data as one
          input (ML Consensus)
        - Treats model **disagreement** as a meaningful signal on its own —
          not something to average away, since it indicates the case is
          genuinely ambiguous and needs a human look
        - Adds independent signals the ML layer never sees (metadata
          integrity) without double-counting features already inside the
          ML consensus score
        - Lets the analyst choose to feature a specific model's result, but
          the underlying Risk Score and Priority Level always reflect the
          full multi-model consensus, since no single model has proven
          reliable enough to trust alone

        Priority-level thresholds and indicator cutoffs in this prototype
        are reasonable starting points, not values statistically validated
        against a large labeled forensic dataset.
        """
    )