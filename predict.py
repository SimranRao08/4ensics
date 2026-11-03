"""
predict.py
----------
Loads a trained model bundle (from train_classifier.py) and runs prediction
on a single new image, printing a risk score and optionally saving the
ELA/FFT visualization for a human reviewer.

Usage:
    python predict.py --model forensics_model.joblib --image path/to/image.jpg --viz out.png
"""

import argparse
import joblib
import numpy as np

from ela_fft_features import extract_features, FEATURE_NAMES, visualize


def predict(model_path, image_path, viz_path=None):
    bundle = joblib.load(model_path)
    clf = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]

    feats = extract_features(image_path)
    x = np.array([[feats[name] for name in feature_names]])
    x_scaled = scaler.transform(x)

    pred = clf.predict(x_scaled)[0]
    label = "FAKE / AI-GENERATED / TAMPERED" if pred == 1 else "REAL / AUTHENTIC"

    proba = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(x_scaled)[0]

    print(f"Image: {image_path}")
    print(f"Extracted features: {feats}")
    print(f"\nPrediction: {label}")
    if proba is not None:
        print(f"Confidence -> real: {proba[0]:.4f}  fake: {proba[1]:.4f}")

    if viz_path:
        visualize(image_path, save_path=viz_path)
        print(f"\nVisualization saved to: {viz_path}")

    return label, proba


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--viz", default=None)
    args = parser.parse_args()

    predict(args.model, args.image, viz_path=args.viz)
