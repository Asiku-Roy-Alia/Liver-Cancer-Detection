"""
build_dicom_manifest.py
----------------------------------
Walks every *.dcm file under a specified ROOT directory,
harvests basic DICOM information, and saves an Excel sheet with:

    ┌────────────────┬──────────────────────────────────────────────────────────────────────────┬──────────────┬──────────────┬───────────┬────────────┬──────────┬────────────┬───────────┐
    │ file_name      │ full_path                                                                │ shape        │ is_4d        │ ok/error  │ patient_id │ modality │ image_type │ study_date│
    └────────────────┴──────────────────────────────────────────────────────────────────────────┴──────────────┴──────────────┴───────────┴────────────┴──────────┴────────────┴───────────┘

• `shape` is the NumPy array shape as a string.
• `is_4d`  = 0  → 3-D (H, W, C)      image (typical single frame RGB)
           = 1  → 4-D (F, H, W, C)   multi-frame cine
           = 2  → anything else (fallback for unexpected dimensions)
• `ok/error` = 1 on successful processing, otherwise the exception message.

Usage:
    python build_dicom_manifest.py "D:\\biomedical_research\\data" "output_manifest.xlsx"

If the ROOT directory argument is omitted, it defaults to a predefined path.
If the output filename is omitted, it defaults to "dicom_manifest.xlsx".
"""

import os
import sys
import numpy as np
import pydicom
import pandas as pd
from tqdm import tqdm # For progress bar

# Default root directory if not provided as a command-line argument
DEFAULT_ROOT = r"D:\biomedical_research\data"
# Default output Excel filename if not provided as a command-line argument
DEFAULT_OUT_XLSX = "dicom_manifest.xlsx"

def build_manifest(root_folder, output_xlsx_path):
    """
    Builds an Excel manifest from DICOM files found in the root_folder.

    Args:
        root_folder (str): The root directory to search for DICOM files.
        output_xlsx_path (str): The path where the Excel manifest will be saved.
    """
    records = []
    print(f"Scanning DICOM files in: {root_folder}")

    # Use tqdm to show progress for the file scanning
    for dirpath, _, filenames in tqdm(os.walk(root_folder), desc="Processing directories"):
        for fn in filenames:
            if fn.lower().endswith(".dcm"):
                full_path = os.path.join(dirpath, fn)
                rec = {"file_name": fn, "full_path": full_path}

                try:
                    ds = pydicom.dcmread(full_path)
                    arr = ds.pixel_array  # NumPy ndarray
                    shp = arr.shape

                    rec["shape"] = str(shp)
                    # Determine if it's 3D (H, W, C) or 4D (F, H, W, C)
                    rec["is_4d"] = 1 if len(shp) == 4 else (0 if len(shp) == 3 else 2)
                    rec["ok/error"] = 1

                    # Extract additional metadata as requested
                    rec["patient_id"] = getattr(ds, 'PatientID', 'N/A')
                    rec["modality"] = getattr(ds, 'Modality', 'N/A')
                    rec["image_type"] = "|".join(getattr(ds, 'ImageType', ['N/A']))
                    rec["study_date"] = getattr(ds, 'StudyDate', 'N/A')

                except Exception as e:
                    rec["shape"] = ""
                    rec["is_4d"] = ""
                    rec["ok/error"] = str(e)
                    # For consistency, set other fields to N/A on error too
                    rec["patient_id"] = "N/A"
                    rec["modality"] = "N/A"
                    rec["image_type"] = "N/A"
                    rec["study_date"] = "N/A"
                records.append(rec)

    if not records:
        print(f"[WARN] No DICOM files found in '{root_folder}'. No manifest generated.")
        return

    df = pd.DataFrame.from_records(records)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_xlsx_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_excel(output_xlsx_path, index=False)
    print(f"[+] Manifest written to '{output_xlsx_path}' ({len(df)} rows)")

if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) > 1:
        root_arg = sys.argv[1]
    else:
        root_arg = DEFAULT_ROOT
        print(f"No root directory provided. Using default: '{DEFAULT_ROOT}'")

    if len(sys.argv) > 2:
        output_file_arg = sys.argv[2]
    else:
        output_file_arg = DEFAULT_OUT_XLSX
        print(f"No output filename provided. Using default: '{DEFAULT_OUT_XLSX}'")

    if not os.path.exists(root_arg):
        print(f"[ERROR] The specified root directory does not exist: '{root_arg}'")
        sys.exit(1)

    build_manifest(root_arg, output_file_arg)

