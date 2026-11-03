"""
final_ensemble.py
------------------
FINAL, STABLE prediction pipeline. Combines all three models trained during
this project:

  forensics_model.joblib           - CIFAKE, 8 features, whole-image
  forensics_model_whatsapp.joblib  - 140k Faces + WhatsApp-sim, 8 features, whole-image
  forensics_model_v3.joblib        - 140k Faces + face-crop + filter-augmented, 10 features

Each model is automatically routed to the feature extractor it was actually
trained on (checked via feature count), then all available models are
averaged. An UNCERTAIN band is used instead of forcing every image into a
confident real/fake call — this is a deliberate design choice, not a
limitation being hidden: for a pre-triage tool, an honest "needs human
review" is more useful than a confident wrong answer.

Do NOT further hand-tune weights based on individual test photos — this
was tried during development and made things WORSE on some images while
improving others. Treat this ensemble as the stable final version.
"""

import os
import joblib
import numpy as np

try:
    import exifread
    HAS_EXIFREAD = True
except ImportError:
    HAS_EXIFREAD = False

from ela_fft_features_v2 import extract_features as extract_features_v2
from ela_fft_features_v3 import extract_features as extract_features_v3

UNCERTAIN_LOW = 0.35
UNCERTAIN_HIGH = 0.65

MODEL_FILES = [
    "forensics_model.joblib",
    "forensics_model_whatsapp.joblib",
    "forensics_model_v4.joblib",
    "forensics_model_unified_rf.joblib",
    "forensics_model_unified_gb.joblib",
]


def load_available_models(model_files=MODEL_FILES):
    bundles = []
    for path in model_files:
        if os.path.exists(path):
            b = joblib.load(path)
            bundles.append({"path": path, "bundle": b, "n_features": len(b["feature_names"])})
    return bundles


def check_exif_context(image_path):
    if not HAS_EXIFREAD:
        return {"available": False, "note": "exifread not installed"}
    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f)
    except Exception as e:
        return {"available": False, "note": f"could not read EXIF: {e}"}
    if len(tags) == 0:
        return {
            "available": False,
            "note": "No EXIF metadata found. Common and NOT itself suspicious — "
                    "WhatsApp, Instagram, and most platforms strip EXIF from every "
                    "photo they process, real or fake."
        }
    software = str(tags.get("Image Software", ""))
    return {
        "available": True,
        "tag_count": len(tags),
        "software_tag": software if software else None,
        "note": "EXIF present. One small extra signal, not a standalone verdict."
    }


def predict(image_path, model_files=MODEL_FILES):
    bundles = load_available_models(model_files)
    if not bundles:
        raise RuntimeError("No trained model files found in this folder.")

    # Extract each feature set only once, reused across every model that needs it
    feats_v2 = None
    feats_v3 = None

    individual = {}
    probs = []

    for entry in bundles:
        n = entry["n_features"]
        bundle = entry["bundle"]

        if n == 8:
            if feats_v2 is None:
                feats_v2 = extract_features_v2(image_path)
            feats = feats_v2
        elif n == 10:
            if feats_v3 is None:
                feats_v3 = extract_features_v3(image_path)
            feats = feats_v3
        else:
            continue  # unknown feature set, skip rather than crash

        fn = bundle["feature_names"]
        x = np.array([[feats[name] for name in fn]])
        x_scaled = bundle["scaler"].transform(x)
        proba = bundle["model"].predict_proba(x_scaled)[0]

        individual[entry["path"]] = {"prob_real": float(proba[0]), "prob_fake": float(proba[1])}
        probs.append(proba)

    avg = np.mean(probs, axis=0)
    prob_real, prob_fake = float(avg[0]), float(avg[1])

    if UNCERTAIN_LOW <= prob_fake <= UNCERTAIN_HIGH:
        verdict = "UNCERTAIN"
    elif prob_fake > UNCERTAIN_HIGH:
        verdict = "LIKELY FAKE / AI-GENERATED / TAMPERED"
    else:
        verdict = "LIKELY REAL / AUTHENTIC"

    return {
        "verdict": verdict,
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "individual_model_results": individual,
        "models_used": [e["path"] for e in bundles],
        "exif_context": check_exif_context(image_path),
        "features_v2": feats_v2,
        "features_v3": feats_v3,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python final_ensemble.py <image_path>")
        sys.exit(1)

    result = predict(sys.argv[1])
    print(f"\nImage: {sys.argv[1]}")
    print(f"Models used: {result['models_used']}")
    for path, r in result["individual_model_results"].items():
        print(f"  {path}: real={r['prob_real']*100:.1f}%  fake={r['prob_fake']*100:.1f}%")
    print(f"\nEnsemble: real={result['prob_real']*100:.1f}%  fake={result['prob_fake']*100:.1f}%")
    print(f"VERDICT: {result['verdict']}")