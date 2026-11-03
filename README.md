# Image Integrity & AI-Synthetics Forensics Module (FOURensics / 4ensics)

Complete implementation of the pivot spec: ELA (spatial) + FFT (frequency)
dual-domain feature fusion, plus a trainable classifier and inference script.

## Files

| File | Purpose |
|---|---|
| `ela_fft_features.py` | Core module. Track A (ELA), Track B (FFT), Track C (fusion vector X), and the 3-panel visualization. |
| `build_dataset.py` | Walks a `real/` + `fake/` image folder, extracts features for every image, writes `features.csv`. |
| `train_classifier.py` | Trains RandomForest or SVM on `features.csv`, prints accuracy/confusion matrix/ROC-AUC, saves the model with `joblib`. |
| `predict.py` | Loads a saved model and scores a single new image (real vs. fake + confidence), can also save the visualization for a human reviewer. |
| `generate_demo_dataset.py` | Generates a small **synthetic** dataset so you can test the whole pipeline immediately. |

## Quick start (already verified working)

```bash
pip install opencv-python-headless numpy matplotlib scikit-learn joblib pillow pandas

# 1. Extract features for a single image + get the 3-panel visualization
python ela_fft_features.py path/to/image.jpg output_viz.png

# 2. Build a dataset CSV from a folder of real/ and fake/ images
python build_dataset.py --data_dir path/to/your_dataset --out features.csv

# 3. Train a classifier
python train_classifier.py --csv features.csv --model_out forensics_model.joblib --algo rf

# 4. Predict on a new image
python predict.py --model forensics_model.joblib --image path/to/new_image.jpg --viz result.png
```

Expected dataset folder structure for steps 2-3:

```
your_dataset/
    real/
        img001.jpg
        img002.jpg
        ...
    fake/
        img001.jpg
        img002.jpg
        ...
```

## About the included demo dataset

`generate_demo_dataset.py` creates 40 synthetic "real" images (smooth
organic gradients + camera-like JPEG compression) and 40 synthetic "fake"
images (same base content with a faint periodic checkerboard pattern
injected, mimicking GAN/diffusion upsampling grid artifacts). This is
**only** to prove the pipeline runs end to end — I ran it myself and
confirmed: feature extraction works, training converges, and the
saved model correctly classifies a held-out image. It is **not** a real
forensic dataset, and a model trained only on it will not generalize to
real photos or real AI-generated images — the synthetic "fake" signal is
too clean and easy to separate (that's why training accuracy hits 100%).

## Using a real dataset (what you'll want for FOURensics)

To get a model that's actually useful for your capstone, point
`build_dataset.py` at real labeled images instead of the demo set:

- **`real/`** — authentic camera/phone photos, ideally with varied JPEG
  compression histories (the more realistic the compression variety, the
  better ELA generalizes).
- **`fake/`** — AI-generated or manipulated images. Good public options:
  - **CASIA v2** (splicing/copy-move tampering) — classic forensics benchmark.
  - Images you generate yourself from Midjourney / DALL-E / Stable Diffusion
    (this is often the most relevant option since your spec specifically
    targets diffusion/GAN upsampling artifacts).
  - Any deepfake-image dataset you already have access to through your
    university or research guide (Dr. Neha Singhal) — check if CHRIST has
    institutional access to something like FaceForensics++ stills or a
    similar corpus.

I can't download these for you (no access to dataset hosting sites from
this environment), but once you have even ~100-200 images per class in the
`real/` / `fake/` folder structure above, the exact same `build_dataset.py`
→ `train_classifier.py` commands will train a real model — no code changes
needed.

## Feature vector reference

```
X = [ela_mean, ela_variance, fft_high_freq_peak_ratio, fft_energy_ratio_variance]
```

- `ela_mean`, `ela_variance` — Track A. Higher/patchier values flag localized
  splicing or copy-paste edits (ELA's known strength; also its known blind
  spot for pure synthetic images with no JPEG history).
- `fft_high_freq_peak_ratio`, `fft_energy_ratio_variance` — Track B. Spikes
  in the high-frequency ring of the spectrum flag the periodic checkerboard
  artifacts left by transposed-convolution/upsampling layers in GANs and
  diffusion models — this is what's meant to close ELA's generative-AI gap.

## Extending

- Swap `RandomForestClassifier` for `SVC` with `--algo svm` in
  `train_classifier.py` — both are wired up already.
- To fold this into your three-layer pipeline (ViT canvas classifier + this
  CV feature module + your DeepFace/RetinaFace soft-biometric layer), treat
  `feature_vector_array()` in `ela_fft_features.py` as one more feature block
  concatenated into your existing Random Forest adapter's input vector.
