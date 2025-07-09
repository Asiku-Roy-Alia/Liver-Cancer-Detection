"""
_03_extract_roi.py
----------------------------------
Performs enhanced Region of Interest (ROI) extraction for ultrasound images.
This script is designed to:
1.  Preprocess images (normalize to uint8, convert to BGR if grayscale).
2.  Detect and remove text annotations and scale markers using morphological operations and inpainting.
3.  Automatically detect the fan-shaped ultrasound scan area.
4.  Create a mask for the detected scan area.
5.  Apply the mask and crop the image to the clean scan area with a small padding.

The script can process both PNG and DICOM files, though it's intended to work
on the preprocessed PNG outputs from earlier stages (e.g., from split_dicom_panels.py
or normalize_bmode.py).

Usage:
    python extract_roi.py <input_folder_path> <output_folder_path>

Arguments:
    1. input_folder_path (str): Path to the folder containing input images (PNGs or DICOMs).
    2. output_folder_path (str): Path to the folder where extracted ROI images will be saved.

Example:
    python extract_roi.py "D:\biomedical_research\preprocessing\split_panels" \
                          "D:\biomedical_research\preprocessing\roi"
"""

import cv2
import numpy as np
import pydicom
import os
from pathlib import Path
from scipy import ndimage # Used for potential future enhancements, not directly in provided snippet
from skimage import measure, morphology # Used for potential future enhancements, not directly in provided snippet
from sklearn.cluster import DBSCAN # Used for potential future enhancements, not directly in provided snippet
import logging
from tqdm import tqdm
import warnings
from typing import Tuple, List, Optional 

warnings.filterwarnings('ignore')

# Configure logging for informative output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses an image by normalizing to uint8 and converting to BGR if grayscale.

    Args:
        image (np.ndarray): Input image array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the processed BGR image
                                       and its grayscale version.
    """
    # Normalize image to 0-255 uint8 if it's not already
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Convert to BGR if grayscale, and get grayscale version
    if image.ndim == 2:
        gray = image
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1: # Grayscale with 3rd dim as 1
        gray = image[:,:,0] # Take the single channel
        image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3: # Already 3-channel (assume BGR or RGB)
        image_bgr = image.copy() # Assume it's BGR if loaded by cv2.imread
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to gray from BGR
    else:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")
        
    return image_bgr, gray

def detect_text_regions(image_bgr: np.ndarray, min_area: int = 100) -> List[Tuple[int, int, int, int]]:
    """
    Detects bright text regions in the image based on thresholding and morphology.

    Args:
        image_bgr (np.ndarray): Input image in BGR format.
        min_area (int): Minimum contour area to consider as a text region.

    Returns:
        List[Tuple[int, int, int, int]]: A list of bounding boxes (x, y, w, h) for detected text.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding to get a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to connect text characters
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    
    # Dilate horizontally then vertically to connect text components
    dilated = cv2.dilate(binary, kernel_h, iterations=2)
    dilated = cv2.dilate(dilated, kernel_v, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    h_img = image_bgr.shape[0]
    for c in contours:
        x, y, w, hc = cv2.boundingRect(c)
        ar = w / float(hc) # Aspect ratio
        
        # Filter contours based on area, aspect ratio, and height (to exclude large non-text areas)
        if cv2.contourArea(c) > min_area and ar > 1.5 and hc < h_img * 0.15:
            regions.append((x, y, w, hc))
            
    return regions

def detect_scale_regions(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detects vertical scale markers, typically found on the left or right sides of ultrasound images.

    Args:
        image_bgr (np.ndarray): Input image in BGR format.

    Returns:
        List[Tuple[int, int, int, int]]: A list of bounding boxes (x, y, w, h) for detected scales.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    regions = []
    
    # Check both left and right strips for vertical scales
    for side_name, x_start_ratio in [("left", 0), ("right", 0.85)]:
        # Define a strip on the side of the image
        x_start = int(w * x_start_ratio)
        strip_width = int(w * 0.15)
        strip = gray[:, x_start : x_start + strip_width]
        
        if strip.shape[1] == 0: # Handle cases where strip width is 0 due to small image
            continue

        # Threshold the strip
        _, binarized_strip = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use morphological opening to find vertical lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20)) # Vertical kernel
        vertical_lines = cv2.morphologyEx(binarized_strip, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            # Filter based on height (must be a significant portion of image height)
            # and width (must be thin)
            if ch > h * 0.3 and cw < w * 0.1:
                regions.append((x + x_start, y, cw, ch)) # Adjust x-coordinate back to original image frame
                
    return regions

def detect_ultrasound_fan_shape(image_bgr: np.ndarray, min_area_ratio: float = 0.1) -> Optional[np.ndarray]:
    """
    Detects the main fan-shaped (or rectangular) ultrasound scan area.
    Uses Canny edge detection, morphological closing, and contour analysis.

    Args:
        image_bgr (np.ndarray): Input image in BGR format.
        min_area_ratio (float): Minimum area of a contour as a ratio of total image area
                                to be considered a potential scan area.

    Returns:
        Optional[np.ndarray]: The best matching contour for the scan area, or None if not found.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Apply CLAHE for better contrast before edge detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Canny edge detection with two different thresholds and combine
    edges1 = cv2.Canny(enhanced_gray, 30, 100)
    edges2 = cv2.Canny(enhanced_gray, 50, 150)
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Morphological closing to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_contour, best_score = None, 0
    
    for c in contours:
        area = cv2.contourArea(c)
        # Skip contours that are too small
        if area < h * w * min_area_ratio:
            continue
            
        x, y, cw, ch = cv2.boundingRect(c)
        
        # Calculate convex hull properties
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area # Solidity: area of contour / area of its convex hull
        
        ar = cw / float(ch) # Aspect ratio of bounding box
        extent = area / (cw * ch) # Extent: area of contour / area of its bounding box
        
        score = 0
        # Scoring based on typical ultrasound scan area characteristics
        if 0.6 <= solidity <= 0.9: # Fan shape is not perfectly convex
            score += 30
        if 0.8 <= ar <= 2.0: # Typical aspect ratio for ultrasound images
            score += 25
        if extent > 0.4: # Contour should fill a good portion of its bounding box
            score += 20
        
        # Reward larger contours (up to a point)
        score += min(25, (area / (h * w)) * 100)
        
        # Reward contours near the horizontal center and not too high up
        cx, cy = x + cw // 2, y + ch // 2
        if abs(cx - w // 2) < w * 0.3: # Within 30% of image width from center
            score += 10
        if cy > h * 0.1: # Not too close to the top edge
            score += 5
            
        if score > best_score:
            best_score, best_contour = score, c
            
    return best_contour

def create_scan_mask(image_shape: Tuple[int, int], contour: Optional[np.ndarray]) -> np.ndarray:
    """
    Creates a binary mask from the detected scan area contour.

    Args:
        image_shape (Tuple[int, int]): Shape (height, width) of the original image.
        contour (Optional[np.ndarray]): The contour of the scan area.

    Returns:
        np.ndarray: A binary mask (255 for scan area, 0 otherwise).
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    if contour is None:
        # If no contour found, return a full white mask (no masking applied)
        return np.ones(image_shape, dtype=np.uint8) * 255
    
    # Fill the contour to create the mask
    cv2.fillPoly(mask, [contour], 255)
    
    # Apply a slight Gaussian blur and re-threshold to smooth mask edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

def inpaint_artifacts(image_bgr: np.ndarray, text_regions: List, scale_regions: List) -> np.ndarray:
    """
    Inpaints (removes) specified text and scale regions from the image.

    Args:
        image_bgr (np.ndarray): Input image in BGR format.
        text_regions (List): List of bounding boxes for text regions.
        scale_regions (List): List of bounding boxes for scale regions.

    Returns:
        np.ndarray: Image with specified regions inpainted.
    """
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    
    # Draw rectangles on the mask for all regions to be inpainted
    for x, y, w, h in text_regions + scale_regions:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1) # -1 fills the rectangle
        
    # Dilate the mask slightly to ensure full coverage for inpainting
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    
    # Perform inpainting
    return cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)

def extract_clean_scan_area(img: np.ndarray) -> np.ndarray:
    """
    Applies the full ROI extraction pipeline to a single image.

    Args:
        img (np.ndarray): Input image (can be grayscale or RGB/BGR).

    Returns:
        np.ndarray: The extracted and cleaned scan area.
    """
    # Step 1: Preprocess image (normalize, convert to BGR, get grayscale)
    img_bgr, gray_img = preprocess_image(img)
    
    # Step 2: Detect text and scale regions
    text_regions = detect_text_regions(img_bgr)
    scale_regions = detect_scale_regions(img_bgr)
    
    # Step 3: Inpaint detected artifacts
    cleaned_img = inpaint_artifacts(img_bgr, text_regions, scale_regions)
    
    # Step 4: Detect the main ultrasound fan shape
    scan_contour = detect_ultrasound_fan_shape(cleaned_img)
    
    # Step 5: Create a mask from the detected scan area
    scan_mask = create_scan_mask(cleaned_img.shape[:2], scan_contour)
    
    # Step 6: Apply the mask to the cleaned image
    # This will black out regions outside the scan area
    res = cv2.bitwise_and(cleaned_img, cleaned_img, mask=scan_mask)
    
    # Step 7: Crop to the bounding box of the scan area with padding
    if scan_contour is None:
        logger.warning("No main scan contour found. Returning entire cleaned image.")
        return res # Return the full cleaned image if no contour is found

    x, y, w, h = cv2.boundingRect(scan_contour)
    pad = 10 # Small padding around the cropped ROI
    
    # Adjust crop coordinates with padding, ensuring they stay within image bounds
    x_start = max(0, x - pad)
    y_start = max(0, y - pad)
    x_end = min(res.shape[1], x + w + pad)
    y_end = min(res.shape[0], y + h + pad)
    
    cropped_roi = res[y_start:y_end, x_start:x_end]
    
    return cropped_roi


def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python extract_roi.py <input_folder_path> <output_folder_path>")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])
    
    if not input_folder.is_dir():
        logger.error(f"Input folder not found: {input_folder}")
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")

    image_files = []
    for ext in [".png", ".dcm"]:
        image_files.extend(input_folder.rglob(f"*{ext}"))

    if not image_files:
        logger.warning(f"No .png or .dcm files found in {input_folder}. Exiting.")
        return

    logger.info(f"Found {len(image_files)} images to process for ROI extraction.")

    for src_path in tqdm(image_files, desc="Extracting ROIs"):
        try:
            # Load image based on file type
            if src_path.suffix.lower() == ".dcm":
                ds = pydicom.dcmread(str(src_path))
                arr = ds.pixel_array
                if arr.ndim == 4: # Handle multi-frame DICOMs
                    arr = arr[0]
                # Normalize and convert to uint8 before passing to preprocessing
                img = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if img.ndim == 2: # Ensure it's 3-channel for consistent processing later
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.ndim == 3 and img.shape[2] == 1:
                    img = cv2.cvtColor(img[:,:,0], cv2.COLOR_GRAY2BGR)
            else: # Assume .png, .jpg etc.
                img = cv2.imread(str(src_path))
                if img is None:
                    logger.error(f"Could not load image: {src_path}. Skipping.")
                    continue
                # cv2.imread loads in BGR, which is consistent with the functions' expectations
                # No need to convert to RGB here, as internal functions expect BGR for cv2 operations

            # Extract the clean scan area
            result_roi = extract_clean_scan_area(img)

            # Construct output path, maintaining relative directory structure
            relative_path = src_path.relative_to(input_folder)
            # Ensure output is always PNG
            dst_path = (output_folder / relative_path).with_suffix(".png")
            
            dst_path.parent.mkdir(parents=True, exist_ok=True) # Create parent directories if they don't exist
            
            # Save the result
            cv2.imwrite(str(dst_path), result_roi)
            
        except Exception as e:
            logger.error(f"Error processing {src_path}: {e}")
            continue # Continue to the next image even if one fails

    logger.info(f"ROI extraction complete. Processed images saved to {output_folder}")

if __name__ == "__main__":
    main()
