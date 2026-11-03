"""
ela_fft_features_v3.py
------------------------
Fixes two root causes found during testing against real photos:

1. DOMAIN MISMATCH: the face-training dataset (140k Real/Fake Faces) is
   pre-cropped portraits with no background. Feeding it whole photos with
   cars/crowds/backgrounds gave systematically different feature statistics
   than what the model was trained on. FIX: detect and crop to the face
   (via OpenCV's DNN face detector — far more robust to angle/hats/glasses/
   crowd scenes than Haar cascades, confirmed via direct testing) before
   extracting features, for BOTH training and inference.

2. FILTER/AI CONFUSION: Instagram/Snapchat-style beautification filters
   smooth skin texture in a way that resembles the "unnaturally smooth"
   signal the model learned from GAN-generated faces. FIX: two new features
   (saturation_std, edge_density) that respond differently to filter-style
   smoothing than to raw texture — improves the model's information to
   separate the two causes, though it does not guarantee perfect separation
   (this remains a genuinely hard problem).

FEATURE_NAMES has grown to 10. Any model trained on the old 8-feature set
is NOT compatible with this module — retrain before using with this file.
"""

import io
import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

FEATURE_NAMES = [
    "ela_mean",
    "ela_variance",
    "ela_multiscale_ratio",
    "fft_high_freq_peak_ratio",
    "fft_energy_ratio_variance",
    "laplacian_var",
    "blockiness_score",
    "lbp_energy",
    "saturation_std",
    "edge_density",
]

_FACE_NET = None


def _load_face_net():
    """Loads OpenCV's DNN face detector. Model files must be in the same
    folder as this script: deploy.prototxt, res10_300x300_ssd_iter_140000.caffemodel"""
    global _FACE_NET
    if _FACE_NET is None:
        here = os.path.dirname(os.path.abspath(__file__))
        prototxt = os.path.join(here, "deploy.prototxt")
        model = os.path.join(here, "res10_300x300_ssd_iter_140000.caffemodel")
        if not os.path.exists(prototxt) or not os.path.exists(model):
            raise FileNotFoundError(
                "Face detector model files not found. Need deploy.prototxt and "
                "res10_300x300_ssd_iter_140000.caffemodel in the same folder as this script."
            )
        _FACE_NET = cv2.dnn.readNetFromCaffe(prototxt, model)
    return _FACE_NET


def detect_and_crop_face(img_bgr, padding=0.3, min_confidence=0.5):
    """
    Returns (cropped_face_bgr, confidence) or (None, 0.0) if no face found.
    Falls back to the full image if no face is detected, so the pipeline
    never hard-fails on non-face images (documents, scenes, objects).
    """
    net = _load_face_net()
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    best_conf, best_box = 0.0, None
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > best_conf:
            best_conf = confidence
            box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            best_box = box

    if best_conf < min_confidence or best_box is None:
        return None, 0.0

    x1, y1, x2, y2 = best_box
    pad_x, pad_y = int((x2 - x1) * padding), int((y2 - y1) * padding)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    return img_bgr[y1:y2, x1:x2], best_conf


# --------------------------------------------------------------------------
# Feature computations (operate on an in-memory BGR array, not a file path,
# so they can be reused for both the whole image and the face crop)
# --------------------------------------------------------------------------
def _ela_stats_from_array(img_bgr, quality):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = np.array(Image.open(buffer))
    diff = np.abs(rgb.astype(np.int16) - recompressed.astype(np.int16))
    ela_gray = diff.mean(axis=2)
    return float(ela_gray.mean()), float(ela_gray.var())


def _fft_stats_from_gray(gray, ring=(0.35, 0.9), z=3.0):
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    log_mag = np.log1p(np.abs(f))
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = r / np.sqrt(cx ** 2 + cy ** 2)
    mask = (r_norm >= ring[0]) & (r_norm <= ring[1])
    vals = log_mag[mask]
    if vals.size == 0:
        return 0.0, 0.0
    mu, sigma = vals.mean(), vals.std() + 1e-8
    peak_ratio = float(np.mean((vals - mu) / sigma > z))
    r_ring = r[mask].astype(np.int32)
    bins = np.bincount(r_ring, weights=vals)
    counts = np.bincount(r_ring)
    counts[counts == 0] = 1
    energy_var = float(np.var(bins / counts))
    return peak_ratio, energy_var


def _blockiness_from_gray(gray):
    diff = np.abs(np.diff(gray.astype(np.float32), axis=1))
    cols = np.arange(diff.shape[1])
    block_cols = cols[(cols % 8) == 7]
    non_block_cols = cols[(cols % 8) != 7]
    if len(block_cols) == 0 or len(non_block_cols) == 0:
        return 0.0
    return float(diff[:, block_cols].mean() / (diff[:, non_block_cols].mean() + 1e-8))


def _lbp_energy_from_gray(gray):
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, 11), density=True)
    return float(np.sum(hist ** 2))


def _saturation_std_from_bgr(img_bgr):
    """Filters/beautification apps apply saturation adjustments that shift
    this differently than raw AI-generation artifacts do."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].astype(np.float32).std())


def _edge_density_from_gray(gray):
    """Ratio of Canny edge pixels to total — captures fine detail loss from
    smoothing (filters OR AI) with a different sensitivity than Laplacian var."""
    edges = cv2.Canny(gray, 100, 200)
    return float(np.sum(edges > 0) / edges.size)


def _extract_from_bgr_array(img_bgr):
    """Runs all 10 features on an in-memory BGR image array."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Multi-scale ELA needs a JPEG round-trip — write to temp buffer approach via PIL
    ela_mean_95, ela_var_95 = _ela_stats_from_array(img_bgr, 95)
    ela_mean_70, _ = _ela_stats_from_array(img_bgr, 70)
    ela_ratio = ela_mean_70 / (ela_mean_95 + 1e-8)

    peak_ratio, energy_var = _fft_stats_from_gray(gray)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    block_score = _blockiness_from_gray(gray)
    lbp_e = _lbp_energy_from_gray(gray)
    sat_std = _saturation_std_from_bgr(img_bgr)
    edge_d = _edge_density_from_gray(gray)

    return {
        "ela_mean": ela_mean_95,
        "ela_variance": ela_var_95,
        "ela_multiscale_ratio": ela_ratio,
        "fft_high_freq_peak_ratio": peak_ratio,
        "fft_energy_ratio_variance": energy_var,
        "laplacian_var": lap_var,
        "blockiness_score": block_score,
        "lbp_energy": lbp_e,
        "saturation_std": sat_std,
        "edge_density": edge_d,
    }


def extract_features(image_path, use_face_crop=True):
    """
    Main entry point. If use_face_crop is True (default) and a face is
    detected, features are extracted from the padded face crop — matching
    what the face-trained model was actually trained on. Falls back to the
    whole image if no face is found.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if use_face_crop:
        crop, conf = detect_and_crop_face(img_bgr)
        if crop is not None and crop.size > 0:
            return _extract_from_bgr_array(crop)

    return _extract_from_bgr_array(img_bgr)


def feature_vector_array(image_path, use_face_crop=True):
    feats = extract_features(image_path, use_face_crop=use_face_crop)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)
