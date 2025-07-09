"""
split_panels.py
----------------------------------
Processes the DICOM manifest to account for split two-panel images.
It identifies entries where 'b-mode_ceus' is "b-mode ceus", and if
corresponding split PNGs (e.g., '_b_mode.png' and '_ceus.png') exist,
it replaces the original entry with two new entries for the split images.
If the split PNGs are not found, the original row is retained.

Usage:
    python split_panels.py <input_manifest_path> <split_png_root_path> <output_manifest_path>

Arguments:
    1. input_manifest_path (str): Path to the input Excel manifest
                                  (e.g., dicom_manifest_with_labels.xlsx).
    2. split_png_root_path (str): The root directory where the split B-mode
                                  and CEUS PNGs are located.
    3. output_manifest_path (str): Path for the output Excel manifest
                                   (e.g., dicom_manifest_with_labels_split.xlsx).

Example:
    python split_panels.py "..\exploration\dicom_manifest_with_labels.xlsx" \
                           "D:\biomedical_research\preprocessing\split_panels_example" \
                           "..\exploration\dicom_manifest_with_labels_split.xlsx"
"""

import os
import pandas as pd
import sys
from pathlib import Path # For robust path manipulation

def split_and_update_manifest(manifest_in_path, out_root_path, manifest_out_path):
    """
    Splits two-panel image entries in the manifest and updates with new rows for split images.

    Args:
        manifest_in_path (str): Path to the input manifest Excel file.
        out_root_path (str): Root directory where split PNGs are stored.
        manifest_out_path (str): Path to save the updated manifest Excel file.
    """
    if not os.path.exists(manifest_in_path):
        print(f"[ERROR] Input manifest file not found: '{manifest_in_path}'")
        sys.exit(1)
    if not os.path.exists(out_root_path):
        print(f"[ERROR] Split PNGs root directory not found: '{out_root_path}'")
        print("Please ensure the directory containing split B-mode and CEUS PNGs exists.")
        sys.exit(1)

    print(f"Reading manifest from: {manifest_in_path}")
    df = pd.read_excel(manifest_in_path)
    rows = []

    print("Processing manifest for two-panel splits...")
    for _, row in df.iterrows():
        # Check if the row represents a two-panel B-mode CEUS image
        if str(row.get("b-mode_ceus", "")).lower() == "b-mode ceus":
            # Derive stem to locate split PNGs
            # This logic needs to match how your split PNGs are named.
            # The original notebook snippet used `full_path.split(os.sep)[-4:]` or `[-6:]`
            # Let's make this more robust by using Path.stem and then reconstructing.
            # Assuming the original full_path ends in something like:
            # .../LiverUS-05/09-28-2012-NA-NA-18643/1.000000-NA-89160/1-01.dcm
            # And the stem for split images is derived from these parts.
            # We'll use the last 6 path components to form the stem, similar to your notebook.
            # This might need adjustment based on your exact splitting script's naming convention.
            path_parts = Path(row.full_path).parts
            # Take relevant parts to form a unique stem for the split images
            # Adjust the slice [-6:] based on how many parent directories contribute to the unique stem
            stem_parts = path_parts[-6:]
            stem = "__".join(stem_parts).replace(".dcm", "").replace(":", "_")

            png_b = os.path.join(out_root_path, f"{stem}_b_mode.png")
            png_c = os.path.join(out_root_path, f"{stem}_ceus.png")

            # Check if both split PNGs exist
            if os.path.isfile(png_b) and os.path.isfile(png_c):
                # If they exist, add two new rows for the split images
                for mod, png in [("b-mode", png_b), ("ceus", png_c)]:
                    new_row = row.copy()
                    new_row.file_name = os.path.basename(png)
                    new_row.full_path = png
                    new_row["b-mode_ceus"] = mod  # Overwrite column value
                    rows.append(new_row)
            else:
                # If split PNGs not found, keep original row unmodified
                print(f"[WARN] Split PNGs not found for {row.full_path}. Keeping original row.")
                rows.append(row)
        else:
            # If it's an original single-panel or already split row, keep it as is
            rows.append(row)

    new_df = pd.DataFrame(rows)

    # Ensure the output directory exists
    output_dir = os.path.dirname(manifest_out_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    new_df.to_excel(manifest_out_path, index=False)
    print(f"Wrote {len(new_df):,} rows to '{manifest_out_path}'")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python split_panels.py <input_manifest_path> <split_png_root_path> <output_manifest_path>")
        sys.exit(1)

    manifest_in = sys.argv[1]
    out_root = sys.argv[2]
    manifest_out = sys.argv[3]

    split_and_update_manifest(manifest_in, out_root, manifest_out)
