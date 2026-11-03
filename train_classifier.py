"""
train_classifier.py
--------------------
Trains a classifier on the fused ELA+FFT feature vectors produced by
build_dataset.py, and saves the trained model + scaler to disk with joblib.

Usage:
    python train_classifier.py --csv features.csv --model_out forensics_model.joblib --algo rf
    python train_classifier.py --csv features.csv --model_out forensics_model.joblib --algo svm
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

from ela_fft_features import FEATURE_NAMES


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df[FEATURE_NAMES].values
    y = df["label"].values
    return X, y, df


def train(csv_path, model_out, algo="rf", test_size=0.2, random_state=42):
    X, y, df = load_data(csv_path)

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Dataset needs both classes (real=0 and fake=1) present. "
            "Check your dataset_root/real and dataset_root/fake folders."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if algo == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced",
        )
    elif algo == "svm":
        clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                   class_weight="balanced", random_state=random_state)
    else:
        raise ValueError("algo must be 'rf' or 'svm'")

    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["real", "fake"])
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {algo.upper()} Results ===")
    print(f"Test accuracy: {acc:.4f}")
    if len(np.unique(y_test)) == 2:
        try:
            y_proba = clf.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            print(f"ROC-AUC: {auc:.4f}")
        except Exception:
            pass
    print("\nConfusion matrix (rows=true, cols=pred) [real, fake]:")
    print(cm)
    print("\nClassification report:")
    print(report)

    if algo == "rf":
        importances = dict(zip(FEATURE_NAMES, clf.feature_importances_))
        print("Feature importances:")
        for k, v in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v:.4f}")

    bundle = {
        "model": clf,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "algo": algo,
    }
    joblib.dump(bundle, model_out)
    print(f"\nSaved trained model bundle to: {model_out}")

    return clf, scaler, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to features.csv from build_dataset.py")
    parser.add_argument("--model_out", default="forensics_model.joblib")
    parser.add_argument("--algo", choices=["rf", "svm"], default="rf")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    train(args.csv, args.model_out, algo=args.algo, test_size=args.test_size)
