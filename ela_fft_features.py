"""
ela_fft_features.py
--------------------
Image Integrity & AI-Synthetics Forensics Module for the FOURensics / 4ensics
pre-triage system.

Implements:
  Track A - Spatial Error Level Analysis (ELA) at JPEG Q=95
  Track B - 2D FFT frequency periodicity check (detects AI upsampling grids)
  Track C - Fusion of both channels into a single scalar feature vector X

Author: generated for Simran Rao / FOURensics (Image Modality Lead)
"""

import os
import io
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # safe for headless / notebook use
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# TRACK A: Spatial Error Level Analysis (ELA)
# --------------------------------------------------------------------------
def compute_ela(image_path, quality=95, scale=15):
    """
    Re-compresses the image at JPEG quality `quality` and returns the
    absolute pixel-wise delta map: |I(x,y) - Q95(I(x,y))|

    Returns
    -------
    ela_map : np.ndarray (H, W) grayscale delta map, scaled for visibility
    mean_val : float  -- mean brightness of the delta map
    var_val  : float  -- variance (peak pixel variance) of the delta map
    """
    original = Image.open(image_path).convert("RGB")

    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    orig_arr = np.asarray(original).astype(np.int16)
    recompressed_arr = np.asarray(recompressed).astype(np.int16)

    diff = np.abs(orig_arr - recompressed_arr)

    # Collapse to single-channel intensity map, scaled up so it's visible
    ela_gray = diff.mean(axis=2)
    ela_map = np.clip(ela_gray * scale, 0, 255).astype(np.uint8)

    mean_val = float(ela_gray.mean())
    var_val = float(ela_gray.var())

    return ela_map, mean_val, var_val


# --------------------------------------------------------------------------
# TRACK B: 2D FFT Frequency Periodicity Check
# --------------------------------------------------------------------------
def compute_fft(image_path, high_freq_ring_frac=(0.35, 0.9), peak_zscore=3.0):
    """
    Converts the image to grayscale, computes the 2D FFT, centers the
    zero-frequency component, and returns the log-magnitude spectrum plus
    two scalar metrics that flag periodic AI-upsampling checkerboard grids:

      high_freq_peak_ratio : fraction of high-frequency ring pixels that
                              are statistically extreme "spikes" (z-score
                              above `peak_zscore`). Diffusion / GAN upsampling
                              leaves regular grid spikes here.
      energy_ratio_var     : variance of the radially-averaged energy
                              profile in the high-frequency ring. Natural
                              photos have smooth falloff; periodic artifacts
                              create a spiky, high-variance profile.

    Returns
    -------
    magnitude_spectrum : np.ndarray (H, W) log-scaled, for visualization
    high_freq_peak_ratio : float
    energy_ratio_var : float
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    log_magnitude = np.log1p(magnitude)

    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    r_norm = r / max_r

    low_frac, high_frac = high_freq_ring_frac
    ring_mask = (r_norm >= low_frac) & (r_norm <= high_frac)

    ring_vals = log_magnitude[ring_mask]
    if ring_vals.size == 0:
        return log_magnitude, 0.0, 0.0

    mu, sigma = ring_vals.mean(), ring_vals.std() + 1e-8
    z_scores = (ring_vals - mu) / sigma
    high_freq_peak_ratio = float(np.mean(z_scores > peak_zscore))

    # Radially-averaged energy profile variance within the ring
    r_ring = r[ring_mask].astype(np.int32)
    bins = np.bincount(r_ring, weights=ring_vals)
    counts = np.bincount(r_ring)
    counts[counts == 0] = 1
    radial_profile = bins / counts
    energy_ratio_var = float(np.var(radial_profile))

    return log_magnitude, high_freq_peak_ratio, energy_ratio_var


# --------------------------------------------------------------------------
# TRACK C: Fusion Feature Vector
# --------------------------------------------------------------------------
FEATURE_NAMES = [
    "ela_mean",
    "ela_variance",
    "fft_high_freq_peak_ratio",
    "fft_energy_ratio_variance",
]


def extract_features(image_path, ela_quality=95):
    """
    Runs both tracks and returns the fused feature vector X as a dict
    (keys match FEATURE_NAMES) ready to feed a classifier.
    """
    _, ela_mean, ela_var = compute_ela(image_path, quality=ela_quality)
    _, peak_ratio, energy_var = compute_fft(image_path)

    return {
        "ela_mean": ela_mean,
        "ela_variance": ela_var,
        "fft_high_freq_peak_ratio": peak_ratio,
        "fft_energy_ratio_variance": energy_var,
    }


def feature_vector_array(image_path, ela_quality=95):
    """Same as extract_features but returns a plain numpy array in
    FEATURE_NAMES order, which is what the classifier expects."""
    feats = extract_features(image_path, ela_quality=ela_quality)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)


# --------------------------------------------------------------------------
# Visualization: original | ELA heatmap | FFT spectrum, side by side
# --------------------------------------------------------------------------
def visualize(image_path, save_path=None, ela_quality=95, show=False):
    """
    Produces a 1x3 figure: original image, ELA heatmap, FFT magnitude
    spectrum. Saves to `save_path` if given, and/or shows inline.
    """
    original = Image.open(image_path).convert("RGB")
    ela_map, ela_mean, ela_var = compute_ela(image_path, quality=ela_quality)
    fft_map, peak_ratio, energy_var = compute_fft(image_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im1 = axes[1].imshow(ela_map, cmap="inferno")
    axes[1].set_title(f"ELA Heatmap (Q95)\nmean={ela_mean:.2f}  var={ela_var:.2f}")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(fft_map, cmap="viridis")
    axes[2].set_title(
        f"FFT Magnitude Spectrum\npeak_ratio={peak_ratio:.4f}  energy_var={energy_var:.4f}"
    )
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return save_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ela_fft_features.py <image_path> [output_png]")
        sys.exit(1)

    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "visualization.png"

    feats = extract_features(path)
    print("Extracted feature vector X:")
    for k, v in feats.items():
        print(f"  {k}: {v:.6f}")

    visualize(path, save_path=out)
    print(f"\nVisualization saved to: {out}")
