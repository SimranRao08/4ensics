"""
risk_engine.py
---------------
Forensic Risk Engine for 4ensics — Digital Evidence Triage & Decision
Support System.

Design principle (per the reframed architecture): the ML models (RF/GB
trained on ELA+FFT+compression+texture features) are ONE input into an
overall risk assessment, not the final verdict. Their output already
encodes ELA/FFT/compression signals internally — so this engine does NOT
re-add those as separate additive terms (that would double-count the same
underlying evidence twice). Instead it combines:

  1. ML Consensus Score   - what the trained models collectively indicate
  2. Model Agreement       - how much the models trained on different data
                              agree with each other (independent of direction)
  3. Independent Signals   - things the ML layer does NOT see at all:
                              metadata presence/integrity, suspicious
                              editing-software tags

Output is a Risk Score (0-100), a Priority Level (Low/Medium/High), a list
of human-readable forensic indicators, and a review recommendation. This
module deliberately never says "fake" or "real" — that determination is
left to the human forensic analyst.

NOTE ON THRESHOLDS: the priority-level cutoffs and indicator thresholds
below are reasonable starting points chosen for a working prototype, not
values empirically calibrated against a large labeled dataset. Say so
plainly if asked — do not present them as statistically validated.
"""

from final_ensemble import predict as ensemble_predict
from ela_fft_features_v3 import detect_and_crop_face
import cv2


# Factual description of what each model was actually trained on — used to
# tell the analyst which model's training domain matches this image's
# content type. This is NOT a claim about which model is "correct" — we
# tested that assumption directly and it does not hold reliably. It is
# transparency about training data composition, left for the human to weigh.
MODEL_TRAINING_DOMAIN = {
    "forensics_model.joblib": {
        "trained_on": "CIFAKE benchmark — small generic objects/scenes (planes, cars, animals) vs. Stable Diffusion equivalents",
        "best_matches": "general scenes, objects, non-portrait images",
    },
    "forensics_model_whatsapp.joblib": {
        "trained_on": "140k Real/Fake Faces + simulated WhatsApp compression",
        "best_matches": "face-centric photos that have been compressed/forwarded",
    },
    "forensics_model_v4.joblib": {
        "trained_on": "140k Real/Fake Faces, face-cropped, with beauty-filter and low-light/webcam augmentation",
        "best_matches": "face-centric photos, including filtered selfies and webcam captures",
    },
    "forensics_model_unified_rf.joblib": {
        "trained_on": "Combined CIFAKE + 140k Real/Fake Faces + all 3 augmentation types (filter/low-light/WhatsApp), single unified training run — Random Forest",
        "best_matches": "general-purpose — broadest single-model training coverage in the system, though tested to still miss some out-of-distribution AI generations",
    },
    "forensics_model_unified_gb.joblib": {
        "trained_on": "Same combined dataset as the unified RF model — Gradient Boosting",
        "best_matches": "general-purpose — included primarily to diversify the ensemble (different algorithm, same data)",
    },
}


def detect_content_type(image_path):
    """Returns (content_type_label, face_confidence)."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return "Unknown", 0.0
    _, conf = detect_and_crop_face(img_bgr)
    if conf > 0.5:
        return "Face-centric photo", conf
    return "General scene / object", conf



def compute_ml_consensus(ensemble_result):
    """
    Returns (consensus_fake_pct, agreement_score).
    consensus_fake_pct: 0-100, average fake-probability across all models.
    agreement_score: 0-1, how close the models are to each other
    (1.0 = perfect agreement, 0.0 = maximal disagreement). This is
    independent of WHICH direction they lean — two models both confidently
    saying opposite things is low agreement regardless of the average.
    """
    individual = ensemble_result["individual_model_results"]
    fake_probs = [r["prob_fake"] for r in individual.values()]

    consensus_fake_pct = (sum(fake_probs) / len(fake_probs)) * 100
    spread = max(fake_probs) - min(fake_probs)
    agreement_score = max(0.0, 1.0 - spread)

    return consensus_fake_pct, agreement_score


def independent_metadata_signal(exif_context):
    """
    Metadata is NOT seen by the ML models at all, so it's a genuinely
    independent signal. Returns (flag_points, indicator_text_or_None).
    Kept deliberately low-weight for "missing EXIF" since that is common
    and NOT itself strong evidence (WhatsApp/Instagram strip it from real
    photos constantly) — only a specific suspicious software tag gets a
    meaningful point bump.
    """
    if not exif_context.get("available"):
        return 3, None  # small, uninformative on its own — no indicator text needed

    software = (exif_context.get("software_tag") or "").lower()
    suspicious_tools = ["photoshop", "gimp", "midjourney", "stable diffusion",
                         "dall-e", "dalle", "faceapp", "facetune"]
    for tool in suspicious_tools:
        if tool in software:
            return 15, f"Editing/generation software detected in metadata: {exif_context['software_tag']}"

    return 0, None


def generate_forensic_indicators(features_v2, features_v3, consensus_fake_pct, agreement_score):
    """
    Human-readable bullets describing what was observed, for the analyst
    to weigh themselves — not a verdict, just surfaced evidence.
    Thresholds here are illustrative starting points, not statistically
    calibrated cutoffs.
    """
    indicators = []
    feats = features_v3 if features_v3 is not None else features_v2
    if feats is None:
        return indicators

    if feats.get("fft_high_freq_peak_ratio", 0) > 0.01:
        indicators.append("Elevated periodicity in frequency spectrum (possible generative upsampling artifact)")

    if feats.get("ela_multiscale_ratio", 0) > 5:
        indicators.append("Large shift in Error Level Analysis across compression levels (possible multi-generation compression or splicing)")

    if feats.get("lbp_energy", 0) > 0.16:
        indicators.append("Unusually smooth/low-texture regions detected (consistent with either heavy filtering or synthetic generation)")

    if feats.get("blockiness_score", 1.0) > 1.3:
        indicators.append("Strong 8x8 JPEG block-grid artifacts detected")

    if agreement_score < 0.6:
        indicators.append("Low agreement between forensic models trained on different data — signal is ambiguous")

    if consensus_fake_pct > 65:
        indicators.append("Multiple forensic models indicate elevated risk signal")

    if not indicators:
        indicators.append("No strong forensic anomalies detected by automated indicators")

    return indicators


def compute_risk(image_path, exif_context=None, model_override=None):
    """
    Full pipeline: runs the ensemble, computes ML consensus + agreement,
    folds in independent metadata signal, and returns a complete risk
    assessment dict. Does NOT declare the image real or fake.

    model_override: optional model filename (e.g. "forensics_model.joblib").
    If given, the risk score uses ONLY that model's output instead of the
    combined average. This is a manual analyst choice, not an automatic
    system decision — we tested automatic content-based routing directly
    and found it is not reliably more accurate, so the system does not
    make this choice for you. It's surfaced here because the analyst
    reviewing a specific case may have context (or a hunch worth testing)
    that justifies overriding the default combined score for that one file.
    """
    ensemble_result = ensemble_predict(image_path)
    consensus_fake_pct, agreement_score = compute_ml_consensus(ensemble_result)

    content_type, face_confidence = detect_content_type(image_path)

    if model_override and model_override in ensemble_result["individual_model_results"]:
        consensus_fake_pct = ensemble_result["individual_model_results"][model_override]["prob_fake"] * 100
        # Agreement score loses its meaning when using a single model — the
        # analyst chose this model deliberately, so treat it as fully "in use"
        # rather than penalizing for the other models' disagreement.
        agreement_score = 1.0

    exif_ctx = exif_context if exif_context is not None else ensemble_result["exif_context"]
    metadata_points, metadata_indicator = independent_metadata_signal(exif_ctx)

    # Risk score: ML consensus is the primary driver (already encodes
    # ELA/FFT/compression/texture — NOT re-added separately here).
    # Independent metadata signal adds on top since it's genuinely
    # separate evidence the ML layer never saw.
    risk_score = min(100, consensus_fake_pct + metadata_points)

    # Priority is driven by BOTH risk score AND model disagreement —
    # a low-agreement case needs human eyes regardless of which way the
    # average leans, since "the models can't agree" is itself the finding.
    if agreement_score < 0.6:
        priority = "Medium" if risk_score < 70 else "High"
    elif risk_score >= 70:
        priority = "High"
    elif risk_score >= 40:
        priority = "Medium"
    else:
        priority = "Low"

    indicators = generate_forensic_indicators(
        ensemble_result.get("features_v2"),
        ensemble_result.get("features_v3"),
        consensus_fake_pct,
        agreement_score,
    )
    if metadata_indicator:
        indicators.append(metadata_indicator)

    recommendations = {
        "Low": "No strong forensic indicators found. Routine review only if flagged by other case context.",
        "Medium": "Some forensic indicators present or models show reduced agreement. Manual analyst review recommended.",
        "High": "Multiple forensic indicators detected and/or high model consensus. Prioritize for immediate manual forensic review.",
    }

    return {
        "risk_score": round(risk_score, 1),
        "priority_level": priority,
        "ml_consensus_fake_pct": round(consensus_fake_pct, 1),
        "model_agreement_pct": round(agreement_score * 100, 1),
        "forensic_indicators": indicators,
        "recommendation": recommendations[priority],
        "individual_model_results": ensemble_result["individual_model_results"],
        "exif_context": exif_ctx,
        "content_type": content_type,
        "face_confidence": round(face_confidence, 3),
        "model_training_domains": MODEL_TRAINING_DOMAIN,
        "model_override_used": model_override,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python risk_engine.py <image_path>")
        sys.exit(1)

    result = compute_risk(sys.argv[1])
    print(f"\nImage: {sys.argv[1]}")
    print(f"Risk Score: {result['risk_score']}/100")
    print(f"Priority Level: {result['priority_level']}")
    print(f"ML Consensus (fake-leaning): {result['ml_consensus_fake_pct']}%")
    print(f"Model Agreement: {result['model_agreement_pct']}%")
    print(f"\nForensic Indicators:")
    for ind in result["forensic_indicators"]:
        print(f"  - {ind}")
    print(f"\nRecommendation: {result['recommendation']}")