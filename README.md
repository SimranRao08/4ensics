# 4ensics: Photo & Face Image Forensics Module

A standalone packaging of the 4ensics image forensic pipeline for integration into the unified digital evidence UI. The module analyzes incoming photographic evidence and produces a forensic risk assessment to help prioritize human analyst review.

**This system is decision support, not a decision-maker.** It does not declare evidence "real" or "fake". Final forensic and legal determinations remain with the human analyst.

Scope: photographs and face images only. Screenshot / chat-evidence detection is not part of this module.

---

## Core Feature: Pre-Upload Model Selection

The analyst chooses how the evidence is evaluated **before** the image is processed. When submitting an image through the UI or API, you select a **Model Choice**:

### "All models combined (recommended)"

- The image is routed through all 5 trained models.
- The system calculates a unified **ML Consensus** score from the aggregate results.
- Model disagreement is treated as a meaningful signal in its own right. If the models disagree strongly, the evidence is flagged as ambiguous and needs human review, instead of the scores being averaged away.

### A specific model override (e.g. `forensics_model_v4.joblib`)

If the analyst already knows what kind of image it is, they can feature the model trained on that domain:

| Evidence category | Model file |
|---|---|
| Scene (general photographs, CIFAKE-style synthetic scenes) | `forensics_model.joblib` |
| Human (WhatsApp-compression-simulated) | `forensics_model_whatsapp.joblib` |
| Filtered person (face-crop, filter, low-light) | `forensics_model_v4.joblib` |
| Unified Random Forest | `forensics_model_unified_rf.joblib` |
| Unified Gradient Boosting | `forensics_model_unified_gb.joblib` |

**Safety guardrail:** even when a single model is featured prominently, the underlying Risk Score and Priority Level are still calculated from the full ensemble consensus. This prevents forensic tunnel vision.

---

## Repository Structure

```
app_image.py                          Gradio application (UI + auto API)
app_final.py                          Streamlit version of the same interface
risk_engine.py                        Risk scoring and priority banding (REQUIRED)
final_ensemble.py                     Ensemble prediction pipeline (REQUIRED)
ela_fft_features_v2.py                Feature extractor: 8-feature models + heatmaps (REQUIRED)
ela_fft_features_v3.py                Feature extractor: 10-feature models + face detection (REQUIRED)
deploy.prototxt                       Face detector architecture (REQUIRED by v3 extractor)
res10_300x300_ssd_iter_140000.caffemodel   Face detector weights (REQUIRED by v3 extractor)
forensics_model.joblib                Scene variant
forensics_model_whatsapp.joblib       Human variant
forensics_model_v4.joblib             Filtered Person variant
forensics_model_unified_rf.joblib     Unified Random Forest
forensics_model_unified_gb.joblib     Unified Gradient Boosting
README.md                             This file
```

All five `.joblib` files are required for the ensemble consensus to work.

---

## Installation and Running

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install gradio joblib numpy scikit-learn opencv-python-headless pillow exifread
```

`exifread` is optional. Without it, the metadata evaluation degrades gracefully and the app still runs.

Run the application:

```bash
python app_image.py
```

This starts:

- a **user interface** at `http://localhost:7861`
- an **auto-generated API** at `http://localhost:7861/?view=api`

To run the Streamlit version instead:

```bash
streamlit run app_final.py
```

### Version notes

- **scikit-learn:** the models were trained with a specific scikit-learn version. If loading a `.joblib` fails or warns about a version mismatch, install the version used for training.
- **OpenCV:** the face detector uses OpenCV's DNN module with a Caffe model. OpenCV 5.x removed Caffe loading, so if face detection errors out, pin an OpenCV 4.x release (e.g. `pip install "opencv-python-headless<5"`).

### Port coordination

Each module in the unified 4ensics system must run on its own port. This module uses **7861**. Make sure it does not collide with other modules (for example, video on 7862 and audio on 7863).

---

## Input Format

| Input | Type | Notes |
|---|---|---|
| Evidence Image | File upload (`.jpg`, `.jpeg`, `.png`) | Passed to `risk_engine.compute_risk()` as a filepath |
| Model Choice | String (dropdown) | `"All models combined (recommended)"` or one of the five exact `.joblib` filenames |

Over the API, call with the image file and the model-choice string as the two positional inputs. The `/?view=api` page shows the exact call syntax for cURL, Python `gradio_client` and JavaScript.

---

## Output Fields

| Field | Type | Meaning |
|---|---|---|
| `priority_level` | String | Triage band: Low / Medium / High. Shown green / amber / red in the unified UI. |
| `risk_score` | String | Primary risk indicator, e.g. `"78.5/100"`. Combines the ML consensus with the independent EXIF metadata signal. |
| `recommendation` | String | Human-readable next-step guidance tied to the priority band. |
| `forensic_indicators` | String | The specific signals that drove the score, e.g. `"- Elevated periodicity..."`. Keeps the score auditable. |
| `model_agreement_pct` | String | How closely the 5 models agree with each other, regardless of direction. |
| `content_type` | String | What the image appears to contain, e.g. `"Face-centric photo"`. |
| `individual_model_results` | String | Per-model percentage breakdown, so an analyst can see where the models agree or conflict. |
| `model_training_domain_note` | String | What the featured model was trained on, for transparency about data bias. |
| `exif_note` | String | Plain-language assessment of the image's EXIF metadata state. |
| `sha256_hash` | String | Integrity hash of the uploaded file. Recompute it at any time to verify the evidence has not been altered. |
| `ela_heatmap` | Image | Error Level Analysis heatmap. Highlights localized re-compression inconsistencies. |
| `fft_magnitude_spectrum` | Image | FFT magnitude spectrum. Periodic grid texture radiating outward can indicate synthetic generation. |

---

## Known Limitations

### Model accuracy

Cross-dataset results for the three individually evaluated variants:

| Variant | Accuracy | ROC-AUC |
|---|---|---|
| Scene (`forensics_model.joblib`) | 79.4% | 0.875 |
| Human (`forensics_model_whatsapp.joblib`) | 59.6% | 0.629 |
| Filtered Person (`forensics_model_v4.joblib`) | 67.9% | 0.763 |

The Human (WhatsApp-compression-simulated) variant is the weakest. Keep this in mind if an analyst explicitly overrides the system to feature it. Per-variant accuracy is not reported for the two unified models.

### Thresholds

Priority-level thresholds and indicator cutoffs are reasonable starting points for a working prototype. They have not been statistically validated against a large labeled forensic dataset.
