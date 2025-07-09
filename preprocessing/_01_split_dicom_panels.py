"""
_01_split_dicom_panels.py
----------------------------------
Automates the splitting of two-panel DICOM images (identified as "b-mode ceus"
in the manifest) into separate B-mode and CEUS PNG files. It uses an
automated vertical split detection algorithm.

The script reads an input manifest, processes relevant DICOM files,
saves the split PNGs to an output directory, and does NOT modify the manifest.
The manifest update logic should be handled by 'split_panels.py' after this script runs.

Usage:
    python split_dicom_panels.py <input_manifest_path> <output_png_root_path>

Arguments:
    1. input_manifest_path (str): Path to the input Excel manifest
                                  (e.g., dicom_manifest_with_labels.xlsx)
                                  containing paths to two-panel DICOMs.
    2. output_png_root_path (str): The root directory where the split B-mode
                                   and CEUS PNGs will be saved.

Example:
    python split_dicom_panels.py "..\exploration\dicom_manifest_with_labels.xlsx" \
                                "D:\biomedical_research\preprocessing\split_panels"
"""

import os
import cv2
import numpy as np
import pydicom
import pandas as pd
from tqdm.auto import tqdm # For progress bar
from pathlib import Path
import sys

# ─────────────── Image Processing Helpers ───────────────

def find_vertical_split(img: np.ndarray) -> int:
    """
    Detects the vertical split line in a two-panel image.
    This function is adapted from the provided notebook.

    Args:
        img (np.ndarray): The input image array (expected RGB).

    Returns:
        int: The x-coordinate of the detected vertical split.
    """
    if img.ndim == 3 and img.shape[2] == 3: # Ensure it's an RGB image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 2: # If already grayscale
        gray = img
    else:
        raise ValueError("Input image must be 2D (grayscale) or 3D (RGB) for splitting.")

    blur = cv2.medianBlur(gray, 5)
    col = np.median(blur, axis=0) # Median pixel value for each column
    thr = np.percentile(col, 30)  # Threshold for dark regions
    dark = col < thr

    segs, on = [], False
    for x, flag in enumerate(dark):
        if flag and not on:
            start = x; on = True
        elif not flag and on:
            segs.append((start, x - 1)); on = False
    if on: segs.append((start, len(dark) - 1))

    W, mid = img.shape[1], img.shape[1] // 2
    band = int(img.shape[1] * 0.15) # Search band around the middle
    best_x, best_d = mid, W # Initialize with middle, and max distance

    for a, b in segs:
        c = (a + b) // 2
        # Find the dark segment closest to the center, within a reasonable band
        if abs(c - mid) < best_d and abs(c - mid) < band:
            best_x, best_d = c, abs(c - mid)
    return int(best_x)

def load_and_normalize_dicom_image(dicom_path: str) -> np.ndarray:
    """
    Loads a DICOM file, extracts pixel array, normalizes it to 0-255 (uint8),
    and converts grayscale to RGB if necessary.

    Args:
        dicom_path (str): Full path to the DICOM file.

    Returns:
        np.ndarray: The normalized 8-bit RGB image array.
    """
    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array

    # Handle 4D (multi-frame) DICOMs by taking the first frame
    if arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim not in [2, 3]:
        raise ValueError(f"Unsupported DICOM pixel array dimension: {arr.ndim} for {dicom_path}")

    # Normalize pixel array to 0-255 (8-bit)
    arr = arr.astype(np.float32)
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert grayscale to RGB if it's a 2D image or 3D single channel
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return arr

def safe_stem(full_path: str) -> str:
    """
    Generates a safe stem from a full file path for use in filenames.
    Uses the last 6 path components to create a unique, clean stem.
    """
    parts = Path(full_path).parts[-6:] # Adjust this slice if your unique identifier needs more/fewer parts
    stem = "__".join(parts).replace(".dcm", "").replace(":", "_").replace(" ", "_")
    return stem

# ─────────────── Main Processing Function ───────────────

def process_two_panel_dicoms(manifest_in_path: str, out_root_path: str):
    """
    Processes two-panel DICOMs listed in the manifest, splits them,
    and saves the B-mode and CEUS halves as PNGs.

    Args:
        manifest_in_path (str): Path to the input manifest Excel file.
        out_root_path (str): Root directory where split PNGs will be saved.
    """
    if not os.path.exists(manifest_in_path):
        print(f"[ERROR] Input manifest file not found: '{manifest_in_path}'")
        sys.exit(1)

    Path(out_root_path).mkdir(parents=True, exist_ok=True) # Ensure output root exists

    print(f"Reading manifest from: {manifest_in_path}")
    df = pd.read_excel(manifest_in_path)

    # Filter for two-panel DICOMs
    two_panel_paths = df[df["b-mode_ceus"].str.lower() == "b-mode ceus"]["full_path"].tolist()
    TOTAL = len(two_panel_paths)

    if TOTAL == 0:
        print("No 'b-mode ceus' entries found in the manifest. Nothing to split.")
        return

    print(f"Found {TOTAL} two-panel DICOMs to process.")

    for idx, dicom_path in tqdm(enumerate(two_panel_paths), total=TOTAL, desc="Splitting two-panel DICOMs"):
        try:
            # Load and normalize the image
            current_image_arr = load_and_normalize_dicom_image(dicom_path)

            # Find the split point
            split_x = find_vertical_split(current_image_arr)

            # Save the halves
            stem = safe_stem(dicom_path)
            png_b_path = os.path.join(out_root_path, f"{stem}_b_mode.png")
            png_c_path = os.path.join(out_root_path, f"{stem}_ceus.png")

            # Save only if files don't exist to avoid re-processing
            if not os.path.isfile(png_b_path):
                cv2.imwrite(png_b_path, current_image_arr[:, :split_x])
            if not os.path.isfile(png_c_path):
                cv2.imwrite(png_c_path, current_image_arr[:, split_x:])

        except Exception as e:
            print(f"\n[ERROR] Failed to process {dicom_path}: {e}")
            continue # Continue to the next image even if one fails

    print(f"All two-panel DICOMs processed. Split PNGs saved to '{out_root_path}'")
    print("Remember to run 'split_panels.py' next to update your manifest with the new PNG paths.")

# ─────────────── Main Execution ───────────────

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_dicom_panels.py <input_manifest_path> <output_png_root_path>")
        sys.exit(1)

    manifest_in = sys.argv[1]
    out_root = sys.argv[2]

    process_two_panel_dicoms(manifest_in, out_root)
