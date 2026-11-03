"""
generate_demo_dataset.py
-------------------------
IMPORTANT: This creates a SYNTHETIC demo dataset only, to prove the
pipeline (feature extraction -> training -> prediction) works end to end.
It is NOT a real forensic dataset and a model trained only on this data
will NOT generalize to real photos or real AI-generated images.

"real/" images: natural-looking smooth gradients + organic noise + JPEG
    compression artifacts (mimics camera photos).
"fake/" images: same base content but with a faint regular periodic grid
    pattern injected (mimics the checkerboard artifacts left by
    transposed-convolution / upsampling layers in GANs and diffusion
    models), and saved without the same JPEG history.

Replace this with your real labeled dataset before trusting the model
(see README.md, "Using a real dataset").
"""

import os
import numpy as np
from PIL import Image
import argparse


def make_organic_base(size=256, seed=0):
    rng = np.random.default_rng(seed)
    # smooth gradient
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    base = 128 + 60 * np.sin(2 * np.pi * (xx * rng.uniform(1, 3) + yy * rng.uniform(1, 3)))

    # organic blotches (simulate texture/lighting variation)
    for _ in range(5):
        cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        r = rng.uniform(size * 0.1, size * 0.4)
        yy_idx, xx_idx = np.ogrid[:size, :size]
        dist = np.sqrt((xx_idx - cx) ** 2 + (yy_idx - cy) ** 2)
        blob = np.exp(-(dist ** 2) / (2 * r ** 2)) * rng.uniform(-40, 40)
        base += blob

    noise = rng.normal(0, 6, (size, size))
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    rgb = np.stack([img, img * 0.95, img * 0.9], axis=-1).astype(np.uint8)
    return rgb


def inject_periodic_grid(rgb, size=256, grid_period=8, strength=6, seed=0):
    """Adds a faint regular checkerboard-like periodic pattern to mimic
    generative-model upsampling artifacts in frequency space."""
    rng = np.random.default_rng(seed + 999)
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)
    grid = strength * (
        np.sin(2 * np.pi * xx / grid_period) * np.sin(2 * np.pi * yy / grid_period)
    )
    out = rgb.astype(np.float32)
    for c in range(3):
        out[:, :, c] += grid
    out += rng.normal(0, 2, out.shape)  # tiny bit of noise so it's not *too* trivial
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="dataset")
    parser.add_argument("--n_per_class", type=int, default=40)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    real_dir = os.path.join(args.out_dir, "real")
    fake_dir = os.path.join(args.out_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    for i in range(args.n_per_class):
        base = make_organic_base(size=args.size, seed=i)
        real_img = Image.fromarray(base)
        # save with realistic JPEG compression history (like a camera photo)
        real_img.save(os.path.join(real_dir, f"real_{i:03d}.jpg"), quality=90)

        faked = inject_periodic_grid(base, size=args.size, seed=i)
        fake_img = Image.fromarray(faked)
        # save at high quality, mimicking a "fresh" AI export with no
        # prior compression history
        fake_img.save(os.path.join(fake_dir, f"fake_{i:03d}.jpg"), quality=98)

    print(f"Generated {args.n_per_class} real + {args.n_per_class} fake demo images")
    print(f"  -> {real_dir}")
    print(f"  -> {fake_dir}")


if __name__ == "__main__":
    main()
