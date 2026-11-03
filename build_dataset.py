"""
build_dataset.py
-----------------
Walks a dataset directory with the expected structure:

    dataset_root/
        real/
            img001.jpg
            img002.jpg
            ...
        fake/
            img001.jpg
            img002.jpg
            ...

Extracts the ELA + FFT fusion feature vector for every image and writes a
single CSV (features.csv) that train_classifier.py consumes.

Usage:
    python build_dataset.py --data_dir ./dataset --out features.csv
"""

import os
import argparse
import csv
from ela_fft_features import extract_features, FEATURE_NAMES

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def walk_class_folder(folder_path, label):
    rows = []
    if not os.path.isdir(folder_path):
        print(f"  [skip] folder not found: {folder_path}")
        return rows

    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in VALID_EXT]

    for i, fname in enumerate(files, 1):
        fpath = os.path.join(folder_path, fname)
        try:
            feats = extract_features(fpath)
            feats["label"] = label
            feats["filename"] = fname
            rows.append(feats)
        except Exception as e:
            print(f"  [error] {fpath}: {e}")

        if i % 20 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)} images in {os.path.basename(folder_path)}/")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                         help="Path to dataset root containing real/ and fake/ subfolders")
    parser.add_argument("--out", default="features.csv",
                         help="Output CSV path")
    args = parser.parse_args()

    real_dir = os.path.join(args.data_dir, "real")
    fake_dir = os.path.join(args.data_dir, "fake")

    print(f"Scanning REAL images in: {real_dir}")
    real_rows = walk_class_folder(real_dir, label=0)  # 0 = real / authentic

    print(f"Scanning FAKE/AI images in: {fake_dir}")
    fake_rows = walk_class_folder(fake_dir, label=1)  # 1 = fake / AI-generated / tampered

    all_rows = real_rows + fake_rows
    if not all_rows:
        print("No images processed. Check your --data_dir structure "
              "(expects real/ and fake/ subfolders).")
        return

    fieldnames = ["filename"] + FEATURE_NAMES + ["label"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\nWrote {len(all_rows)} feature rows "
          f"({len(real_rows)} real, {len(fake_rows)} fake) to {args.out}")


if __name__ == "__main__":
    main()
