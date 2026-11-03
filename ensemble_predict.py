"""
ensemble_predict.py
--------------------
Fixes the core accuracy problem identified when testing against a real
uploaded photo: a single model trained on one narrow benchmark (e.g. CIFAKE's
tiny 32x32 generic-object images) gives confidently WRONG verdicts on
real-world photos it was never trained to see.

This module:
  1. Runs BOTH trained models (CIFAKE-based + WhatsApp-compression-simulated)
     and averages their probabilities. Two models trained on different data
     distributions disagreeing is itself useful information.
  2. Introduces an UNCERTAIN band. Forcing a binary real/fake call when the
     probability is near 50/50 produces false confidence. For a pre-triage
     tool feeding human reviewers, "inconclusive - review this one" is a
     more honest and more USEFUL output than a confident coin-flip.
  3. Adds an EXIF metadata check as auxiliary context only (NOT a verdict
     signal) — most real photos shared via WhatsApp/social media also have
     stripped EXIF, so presence/absence alone cannot reliably distinguish
     real from fake. It's shown to the human reviewer as extra context.

This intentionally does NOT include the CLIP zero-shot AI-detector or the
hardcoded ELA/text-blob thresholds from the other uploaded scripts
(app.py / detect_ai_edit.py) — those are unvalidated heuristics that
contradict the core "trained + evaluated classifier, not thresholds"
approach this project is built on, and CLIP zero-shot detection for this
specific task is known to be unreliable without fine-tuning.
"""

import os
import joblib
import numpy as np

try:
    import exifread
    HAS_EXIFREAD = True
except ImportError:
    HAS_EXIFREAD = False

from ela_fft_features_v2 import extract_features

UNCERTAIN_LOW = 0.35
UNCERTAIN_HIGH = 0.65


def load_models(model_paths):
    """model_paths: list of .joblib paths. Returns list of bundle dicts."""
    bundles = []
    for path in model_paths:
        if os.path.exists(path):
            bundles.append({"path": path, "bundle": joblib.load(path)})
        else:
            print(f"  [warning] model not found, skipping: {path}")
    return bundles


def predict_single(bundle, feats):
    clf = bundle["model"]
    scaler = bundle["scaler"]
    fn = bundle["feature_names"]
    x = np.array([[feats[name] for name in fn]])
    x_scaled = scaler.transform(x)
    proba = clf.predict_proba(x_scaled)[0]
    return proba  # [prob_real, prob_fake]


def check_exif_context(image_path):
    """
    Returns informational context only — NOT used to compute the verdict.
    Absence of EXIF is common and expected for legitimately shared photos
    (WhatsApp, Instagram, screenshots all strip it), so it is reported as
    context for the human reviewer, not scored as suspicious.
    """
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
            "note": "No EXIF metadata found. This is common and NOT itself "
                    "suspicious — WhatsApp, Instagram, and most social/"
                    "messaging platforms strip EXIF from every photo they "
                    "process, real or fake."
        }

    software = str(tags.get("Image Software", ""))
    return {
        "available": True,
        "tag_count": len(tags),
        "software_tag": software if software else None,
        "note": "EXIF present — camera/software metadata found. Presence "
                "of editing software here is one small extra signal, not a "
                "standalone verdict."
    }


def ensemble_predict(image_path, model_paths=("forensics_model.joblib", "forensics_model_whatsapp.joblib"),
                      weights=None):
    """
    Full pipeline: extract features once, run every available model,
    combine their probabilities (weighted if `weights` given, else equal),
    classify into real / uncertain / fake, and attach EXIF context.

    `weights`: optional dict {model_path: weight}. Use this to trust a
    model more when its training data is a closer match to the image
    domain — e.g. a face/photo-trained model should be weighted higher
    than a generic-object-trained model when scoring photos of people.
    """
    feats = extract_features(image_path)
    bundles = load_models(list(model_paths))

    if not bundles:
        raise RuntimeError("No trained model files found. Provide at least one .joblib model path.")

    individual = {}
    probs = []
    w_list = []
    for entry in bundles:
        proba = predict_single(entry["bundle"], feats)
        individual[entry["path"]] = {"prob_real": float(proba[0]), "prob_fake": float(proba[1])}
        probs.append(proba)
        w_list.append(weights.get(entry["path"], 1.0) if weights else 1.0)

    probs = np.array(probs)
    w_arr = np.array(w_list).reshape(-1, 1)
    avg = (probs * w_arr).sum(axis=0) / w_arr.sum()
    prob_real, prob_fake = float(avg[0]), float(avg[1])

    if UNCERTAIN_LOW <= prob_fake <= UNCERTAIN_HIGH:
        verdict = "UNCERTAIN"
    elif prob_fake > UNCERTAIN_HIGH:
        verdict = "LIKELY FAKE / AI-GENERATED / TAMPERED"
    else:
        verdict = "LIKELY REAL / AUTHENTIC"

    exif_context = check_exif_context(image_path)

    return {
        "verdict": verdict,
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "individual_model_results": individual,
        "features": feats,
        "exif_context": exif_context,
        "models_used": [e["path"] for e in bundles],
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ensemble_predict.py <image_path>")
        sys.exit(1)

    result = ensemble_predict(sys.argv[1])
    print(f"\nImage: {sys.argv[1]}")
    print(f"Models used: {result['models_used']}")
    print(f"\nIndividual model results:")
    for path, r in result["individual_model_results"].items():
        print(f"  {path}: real={r['prob_real']:.4f}  fake={r['prob_fake']:.4f}")
    print(f"\nEnsemble: real={result['prob_real']:.4f}  fake={result['prob_fake']:.4f}")
    print(f"VERDICT: {result['verdict']}")
    print(f"\nEXIF context: {result['exif_context']['note']}")