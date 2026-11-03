"""
ela_fft_features_v2.py
------------------------
Extended feature extraction module — matches the 8-feature model trained
on Kaggle (multi-scale ELA + FFT + blockiness + Laplacian + LBP texture).

This MUST stay in sync with whatever feature set your forensics_model.joblib
was trained on. If you retrain with different features, update FEATURE_NAMES
and extract_features() here to match.
"""

import io
import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_NAMES = [
    "ela_mean",
    "ela_variance",
    "ela_multiscale_ratio",
    "fft_high_freq_peak_ratio",
    "fft_energy_ratio_variance",
    "laplacian_var",
    "blockiness_score",
    "lbp_energy",
]


def compute_ela_map(image_path, quality=95, scale=15):
    """Returns the visual ELA heatmap (for display) at the given quality."""
    original = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)
    diff = np.abs(np.asarray(original).astype(np.int16) - np.asarray(recompressed).astype(np.int16))
    ela_gray = diff.mean(axis=2)
    ela_map = np.clip(ela_gray * scale, 0, 255).astype(np.uint8)
    return ela_map, float(ela_gray.mean()), float(ela_gray.var())


def compute_fft_map(gray, ring=(0.35, 0.9), z=3.0):
    """Returns the visual FFT magnitude spectrum plus scalar metrics."""
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
        return log_mag, 0.0, 0.0
    mu, sigma = vals.mean(), vals.std() + 1e-8
    peak_ratio = float(np.mean((vals - mu) / sigma > z))
    r_ring = r[mask].astype(np.int32)
    bins = np.bincount(r_ring, weights=vals)
    counts = np.bincount(r_ring)
    counts[counts == 0] = 1
    energy_var = float(np.var(bins / counts))
    return log_mag, peak_ratio, energy_var


def blockiness_score(gray):
    """8x8 JPEG grid blockiness — ratio of edge energy on block boundaries vs elsewhere."""
    diff = np.abs(np.diff(gray.astype(np.float32), axis=1))
    cols = np.arange(diff.shape[1])
    block_cols = cols[(cols % 8) == 7]
    non_block_cols = cols[(cols % 8) != 7]
    if len(block_cols) == 0 or len(non_block_cols) == 0:
        return 0.0
    block_energy = diff[:, block_cols].mean()
    non_block_energy = diff[:, non_block_cols].mean()
    return float(block_energy / (non_block_energy + 1e-8))


def lbp_energy(gray):
    """Local Binary Pattern texture energy — flags unnaturally smooth AI textures."""
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, 11), density=True)
    return float(np.sum(hist ** 2))


def extract_features(image_path):
    """Returns the full 8-feature dict, in FEATURE_NAMES order-compatible form."""
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    _, ela_mean_95, ela_var_95 = compute_ela_map(image_path, quality=95)
    _, ela_mean_70, _ = compute_ela_map(image_path, quality=70)
    ela_ratio = ela_mean_70 / (ela_mean_95 + 1e-8)

    _, peak_ratio, energy_var = compute_fft_map(gray)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    block_score = blockiness_score(gray)
    lbp_e = lbp_energy(gray)

    return {
        "ela_mean": ela_mean_95,
        "ela_variance": ela_var_95,
        "ela_multiscale_ratio": ela_ratio,
        "fft_high_freq_peak_ratio": peak_ratio,
        "fft_energy_ratio_variance": energy_var,
        "laplacian_var": lap_var,
        "blockiness_score": block_score,
        "lbp_energy": lbp_e,
    }


def feature_vector_array(image_path):
    feats = extract_features(image_path)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)


def visualize(image_path, save_path=None, show=False):
    """3-panel visualization: original | ELA heatmap | FFT spectrum."""
    original = Image.open(image_path).convert("RGB")
    ela_map, ela_mean, ela_var = compute_ela_map(image_path, quality=95)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    fft_map, peak_ratio, energy_var = compute_fft_map(gray)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im1 = axes[1].imshow(ela_map, cmap="inferno")
    axes[1].set_title(f"ELA Heatmap (Q95)\nmean={ela_mean:.2f}  var={ela_var:.2f}")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(fft_map, cmap="viridis")
    axes[2].set_title(f"FFT Magnitude Spectrum\npeak_ratio={peak_ratio:.4f}  energy_var={energy_var:.4f}")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path
