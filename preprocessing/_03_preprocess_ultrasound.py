"""
_02_preprocess_ultrasound.py
----------------------------------
A comprehensive preprocessing pipeline for ultrasound images, including:
- Noise reduction (median, Gaussian, bilateral filtering)
- Contrast enhancement (CLAHE, histogram equalization, top-hat for CEUS)
- Intensity normalization (Z-score or Min-Max)
- Annotation and caliper removal using inpainting
- Content-based cropping to remove black borders
- Resizing with optional aspect ratio maintenance
- Quality checks (image area, intensity variation)
- Data augmentation (rotation, flipping, elastic deformation - currently not implemented fully)

The script defines a PreprocessingConfig dataclass for easy parameter management
and an UltrasoundPreprocessor class to apply the pipeline.

Usage:
    python preprocess_ultrasound.py

    Modify the 'input_directory' and 'output_directory' in the
    if __name__ == "__main__": block to match your data paths.
    You can also define different configurations for training and evaluation.
"""

import os
import cv2
import numpy as np
import pandas as pd
import pydicom # Although not directly used for image loading here, it's a core dependency for DICOM projects
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import json
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler # Not directly used in the provided snippet but good to keep
from scipy import ndimage # Not directly used in the provided snippet but good to keep
from skimage import filters, morphology, measure, restoration, exposure # For potential future use or specific implementations
import matplotlib.pyplot as plt # For visualization, though not used in batch processing
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure logging for informative output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration class for preprocessing parameters"""
    # Image dimensions
    target_size: Tuple[int, int] = (256, 256)
    maintain_aspect_ratio: bool = True
    
    # Noise reduction
    apply_noise_reduction: bool = True
    median_kernel_size: int = 3
    gaussian_sigma: float = 0.8
    
    # Intensity normalization
    apply_histogram_eq: bool = True
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    apply_zscore_norm: bool = True # If False, min-max normalization is applied
    
    # Artifact removal
    remove_annotations: bool = True
    remove_calipers: bool = True # This is handled within detect_and_remove_annotations
    inpaint_method: str = 'telea'  # 'telea' or 'ns'
    
    # ROI extraction
    crop_to_content: bool = True
    content_threshold: int = 10 # Threshold for detecting non-black content
    roi_padding: int = 20 # Padding around the detected content
    
    # Data augmentation (for training set)
    augmentation_enabled: bool = False
    rotation_range: Tuple[int, int] = (-15, 15)
    flip_horizontal: bool = True
    flip_vertical: bool = False
    elastic_alpha: float = 50.0 # For elastic deformation (skimage)
    elastic_sigma: float = 5.0  # For elastic deformation (skimage)
    
    # Quality control
    min_image_area: int = 10000 # Minimum pixel area for an image to be considered valid
    max_image_area: int = 1000000 # Maximum pixel area
    min_intensity_std: float = 5.0 # Minimum standard deviation of pixel intensities


class UltrasoundPreprocessor:
    """
    Comprehensive preprocessing pipeline for ultrasound images.
    Applies a series of transformations including noise reduction, contrast
    enhancement, normalization, artifact removal, cropping, resizing,
    and optional data augmentation.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initializes the preprocessor with a given configuration.

        Args:
            config (PreprocessingConfig): An instance of PreprocessingConfig
                                          defining preprocessing parameters.
        """
        self.config = config
        self.stats = {
            'processed_count': 0,
            'rejected_count': 0,
            'b_mode_stats': {'mean': [], 'std': []},
            'ceus_stats': {'mean': [], 'std': []},
            'intensity_ranges': {'b_mode': [], 'ceus': []}
        }
        
    def detect_and_remove_annotations(self, image: np.ndarray) -> np.ndarray:
        """
        Detects and removes calipers, measurements, and text annotations from the image
        using thresholding, morphological operations, and inpainting.

        Args:
            image (np.ndarray): Input image (can be grayscale or RGB).

        Returns:
            np.ndarray: Image with annotations removed via inpainting.
        """
        if not self.config.remove_annotations:
            return image
            
        # Convert to grayscale if needed for mask creation
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # 1. Detect bright text/numbers (typical for measurements)
        # Thresholding to find very bright pixels
        _, text_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to connect text characters and lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel_h)
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel_v)
        
        # 2. Detect calipers (usually thin, bright lines with specific patterns)
        # Use opening to find thin bright lines (calipers)
        kernel_line_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        kernel_line_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        
        lines_h = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_line_h)
        lines_v = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_line_v)
        
        # Threshold to get bright lines
        _, lines_h = cv2.threshold(lines_h, 150, 255, cv2.THRESH_BINARY)
        _, lines_v = cv2.threshold(lines_v, 150, 255, cv2.THRESH_BINARY)
        
        # Combine all annotation masks
        mask = cv2.bitwise_or(mask, text_mask)
        mask = cv2.bitwise_or(mask, lines_h)
        mask = cv2.bitwise_or(mask, lines_v)
        
        # 3. Detect crosshairs and measurement markers (small bright spots)
        # Use Top-hat filter to find bright spots on a dark background
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_small)
        _, markers = cv2.threshold(tophat, 50, 255, cv2.THRESH_BINARY)
        
        mask = cv2.bitwise_or(mask, markers)
        
        # Dilate mask to ensure complete coverage of annotations for inpainting
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, dilate_kernel, iterations=2)
        
        # Inpaint to remove annotations
        inpaint_flag = cv2.INPAINT_TELEA if self.config.inpaint_method == 'telea' else cv2.INPAINT_NS
        
        # Ensure image is 8-bit for inpainting
        if image.dtype != np.uint8:
            image_for_inpaint = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            image_for_inpaint = image.copy()

        # If the original image was grayscale, convert to 3-channel for inpaint if needed, then back
        if len(image_for_inpaint.shape) == 2:
            image_for_inpaint_rgb = cv2.cvtColor(image_for_inpaint, cv2.COLOR_GRAY2BGR)
            result_bgr = cv2.inpaint(image_for_inpaint_rgb, mask, 3, inpaint_flag)
            result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
        else: # Assumed to be 3-channel (RGB or BGR)
            # OpenCV inpaint expects BGR, so convert if input is RGB
            if image_for_inpaint.shape[2] == 3:
                image_for_inpaint_bgr = cv2.cvtColor(image_for_inpaint, cv2.COLOR_RGB2BGR)
                result_bgr = cv2.inpaint(image_for_inpaint_bgr, mask, 3, inpaint_flag)
                result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            else: # Fallback for other 3-channel formats if any (unlikely for typical images)
                result = cv2.inpaint(image_for_inpaint, mask, 3, inpaint_flag)
                
        return result
    
    def noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction techniques specific to ultrasound speckle.
        Uses median filter, Gaussian blur, and bilateral filter.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Denoised image.
        """
        if not self.config.apply_noise_reduction:
            return image
            
        # Ensure image is 8-bit for cv2 filters
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply median filter for speckle noise reduction
        denoised = cv2.medianBlur(image, self.config.median_kernel_size)
        
        # Apply gentle Gaussian smoothing
        denoised = cv2.GaussianBlur(denoised, (3, 3), self.config.gaussian_sigma)
        
        # Apply edge-preserving bilateral filter
        # Bilateral filter needs 3 channels if input is color, or 1 if grayscale
        if len(image.shape) == 3:
            denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        else: # Grayscale
            denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
            
        return denoised
    
    def enhance_contrast(self, image: np.ndarray, modality: str = 'b_mode') -> np.ndarray:
        """
        Apply contrast enhancement techniques such as CLAHE and histogram equalization.
        Includes specific enhancement for CEUS images.

        Args:
            image (np.ndarray): Input image.
            modality (str): Modality of the image ('b_mode' or 'ceus').

        Returns:
            np.ndarray: Contrast-enhanced image.
        """
        enhanced = image.copy()
        
        # Convert to grayscale for processing
        if len(enhanced.shape) == 3:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        else:
            gray = enhanced.copy()
        
        # Ensure image is 8-bit for CLAHE/equalizeHist
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.config.apply_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_grid_size
            )
            enhanced_gray = clahe.apply(gray)
        else:
            enhanced_gray = gray
            
        # Apply histogram equalization if requested
        if self.config.apply_histogram_eq:
            enhanced_gray = cv2.equalizeHist(enhanced_gray)
        
        # For CEUS images, enhance vascular structures using top-hat
        if modality == 'ceus':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            tophat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_TOPHAT, kernel)
            enhanced_gray = cv2.add(enhanced_gray, tophat)
        
        # Convert back to original format (RGB if input was RGB, else grayscale)
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        else:
            enhanced = enhanced_gray
            
        return enhanced
    
    def normalize_intensity(self, image: np.ndarray, modality: str) -> np.ndarray:
        """
        Normalize image intensities using either Z-score normalization
        or Min-Max scaling to [0, 1]. Stores statistics.

        Args:
            image (np.ndarray): Input image.
            modality (str): Modality of the image ('b_mode' or 'ceus').

        Returns:
            np.ndarray: Intensity-normalized image (float32).
        """
        normalized = image.astype(np.float32)
        
        if self.config.apply_zscore_norm:
            # Z-score normalization
            mean_val = np.mean(normalized)
            std_val = np.std(normalized)
            
            if std_val > 0:
                normalized = (normalized - mean_val) / std_val
            else:
                # Handle case where std_val is 0 (e.g., uniform image)
                normalized = normalized - mean_val # Center around 0
                
            # Store statistics
            self.stats[f'{modality}_stats']['mean'].append(float(mean_val))
            self.stats[f'{modality}_stats']['std'].append(float(std_val))
        else:
            # Min-max normalization to [0, 1]
            min_val = np.min(normalized)
            max_val = np.max(normalized)
            if max_val > min_val:
                normalized = (normalized - min_val) / (max_val - min_val)
            # If max_val == min_val, image is uniform, remains 0 after subtraction
        
        # Store intensity range
        self.stats['intensity_ranges'][modality].append((float(np.min(normalized)), float(np.max(normalized))))
        
        return normalized
    
    def crop_to_content(self, image: np.ndarray) -> np.ndarray:
        """
        Crops the image to remove black borders and focus on the actual content.
        Adds padding around the detected content.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Cropped image.
        """
        if not self.config.crop_to_content:
            return image
            
        # Convert to grayscale for content detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Threshold to find content (pixels above a certain intensity)
        _, mask = cv2.threshold(gray, self.config.content_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours of the content
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No content found for cropping. Returning original image.")
            return image
            
        # Get the bounding box of the largest contour (assumed to be the main content)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding and ensure coordinates are within image bounds
        padding = self.config.roi_padding
        img_h, img_w = image.shape[:2]

        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img_w, x + w + padding)
        y_end = min(img_h, y + h + padding)
        
        # Crop image
        if len(image.shape) == 3:
            cropped = image[y_start:y_end, x_start:x_end, :]
        else:
            cropped = image[y_start:y_end, x_start:x_end]
            
        return cropped
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resizes the image to target dimensions, optionally maintaining aspect ratio
        and padding with zeros.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Resized and/or padded image.
        """
        target_h, target_w = self.config.target_size
        
        # Ensure image is 8-bit for cv2.resize
        if image.dtype != np.uint8:
            image_for_resize = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            image_for_resize = image.copy()

        if self.config.maintain_aspect_ratio:
            h, w = image_for_resize.shape[:2]
            scale = min(target_w / w, target_h / h)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize with aspect ratio maintained
            resized = cv2.resize(image_for_resize, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Pad to target size
            if len(image_for_resize.shape) == 3:
                padded = np.zeros((target_h, target_w, image_for_resize.shape[2]), dtype=image_for_resize.dtype)
            else:
                padded = np.zeros((target_h, target_w), dtype=image_for_resize.dtype)
                
            # Center the resized image
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            
            if len(image_for_resize.shape) == 3:
                padded[start_y:start_y+new_h, start_x:start_x+new_w, :] = resized
            else:
                padded[start_y:start_y+new_h, start_x:start_x+new_w] = resized
                
            return padded
        else:
            # Direct resize without maintaining aspect ratio
            return cv2.resize(image_for_resize, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    def quality_check(self, image: np.ndarray) -> bool:
        """
        Performs basic quality checks on the image (area, intensity variation).

        Args:
            image (np.ndarray): Input image.

        Returns:
            bool: True if image passes quality checks, False otherwise.
        """
        h, w = image.shape[:2]
        area = h * w
        
        # Check image size
        if area < self.config.min_image_area or area > self.config.max_image_area:
            logger.warning(f"Image size {area} (H:{h}, W:{w}) outside acceptable range "
                           f"[{self.config.min_image_area}, {self.config.max_image_area}].")
            return False
            
        # Check intensity variation (avoid blank or near-blank images)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        intensity_std = np.std(gray)
        if intensity_std < self.config.min_intensity_std:
            logger.warning(f"Image has low intensity variation: {intensity_std} (min_std: {self.config.min_intensity_std}).")
            return False
            
        return True
    
    def apply_augmentation(self, image: np.ndarray, seed: Optional[int] = None) -> List[np.ndarray]:
        """
        Apply data augmentation techniques (rotation, horizontal/vertical flip).

        Args:
            image (np.ndarray): Input image.
            seed (Optional[int]): Random seed for reproducibility.

        Returns:
            List[np.ndarray]: A list containing the original image and its augmented versions.
        """
        if not self.config.augmentation_enabled:
            return [image]
            
        if seed is not None:
            np.random.seed(seed)
            
        augmented_images = [image]  # Always include the original image
        
        # Rotation
        angle = np.random.uniform(self.config.rotation_range[0], self.config.rotation_range[1])
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        augmented_images.append(rotated)
        
        # Horizontal flip
        if self.config.flip_horizontal:
            flipped_h = cv2.flip(image, 1)
            augmented_images.append(flipped_h)
            
        # Vertical flip
        if self.config.flip_vertical:
            flipped_v = cv2.flip(image, 0)
            augmented_images.append(flipped_v)
            
        # Elastic deformation (using skimage for this) - requires float input
        # Note: This part was in the config but not implemented in the snippet.
        # Adding a basic implementation here.
        # if self.config.elastic_alpha > 0 and self.config.elastic_sigma > 0:
        #     try:
        #         # Convert to float for elastic deformation
        #         image_float = image.astype(np.float32) / 255.0
        #         elastic_deformed = morphology.elastic_deform_image(
        #             image_float,
        #             alpha=self.config.elastic_alpha,
        #             sigma=self.config.elastic_sigma,
        #             random_state=seed
        #         )
        #         # Convert back to original dtype and scale
        #         elastic_deformed = (elastic_deformed * 255).astype(image.dtype)
        #         augmented_images.append(elastic_deformed)
        #     except Exception as e:
        #         logger.warning(f"Failed to apply elastic deformation: {e}")
                
        return augmented_images
    
    def process_single_image(self, image: np.ndarray, modality: str, 
                             apply_augmentation: bool = False) -> Union[np.ndarray, List[np.ndarray], None]:
        """
        Processes a single image through the complete preprocessing pipeline.

        Args:
            image (np.ndarray): Input image.
            modality (str): Modality of the image ('b_mode' or 'ceus').
            apply_augmentation (bool): Whether to apply data augmentation.

        Returns:
            Union[np.ndarray, List[np.ndarray], None]: The processed image(s)
            or None if the image fails quality checks.
        """
        
        # Step 0: Initial Quality check
        if not self.quality_check(image):
            self.stats['rejected_count'] += 1
            return None
            
        # Step 1: Remove annotations and artifacts
        processed = self.detect_and_remove_annotations(image)
        
        # Step 2: Crop to content
        processed = self.crop_to_content(processed)
        
        # Step 3: Noise reduction
        processed = self.noise_reduction(processed)
        
        # Step 4: Contrast enhancement
        processed = self.enhance_contrast(processed, modality)
        
        # Step 5: Resize to target dimensions
        processed = self.resize_image(processed)
        
        # Step 6: Intensity normalization (this will output float32)
        processed = self.normalize_intensity(processed, modality)
        
        # Step 7: Data augmentation (if requested)
        if apply_augmentation:
            result = self.apply_augmentation(processed)
        else:
            result = processed
            
        self.stats['processed_count'] += 1
        return result
    
    def process_dataset(self, input_dir: str, output_dir: str, 
                        manifest_file: Optional[str] = None,
                        apply_augmentation_for_training: bool = False) -> Dict:
        """
        Processes an entire dataset of images. It can read image paths from
        a manifest file or discover them in the input directory.
        Processed images are saved to the output directory, organized by modality.
        Processing metadata and statistics are also saved.

        Args:
            input_dir (str): The root directory containing input images.
            output_dir (str): The root directory to save processed images and metadata.
            manifest_file (Optional[str]): Path to an Excel manifest file. If provided,
                                           image paths are read from 'full_path' column.
                                           Otherwise, images are discovered in input_dir.
            apply_augmentation_for_training (bool): If True, augmentation is applied
                                                     during processing (typically for training data).

        Returns:
            Dict: A dictionary containing processing summary, metadata DataFrame, and statistics.
        """
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for modalities and metadata
        (output_path / 'b_mode').mkdir(exist_ok=True)
        (output_path / 'ceus').mkdir(exist_ok=True)
        (output_path / 'metadata').mkdir(exist_ok=True)
        
        image_files = []
        if manifest_file and os.path.exists(manifest_file):
            logger.info(f"Loading image paths from manifest: {manifest_file}")
            df = pd.read_excel(manifest_file)
            for _, row in df.iterrows():
                # Assuming 'full_path' column contains the path to the image
                if pd.notna(row.get('full_path')):
                    image_files.append(Path(row['full_path']))
        else:
            logger.info(f"No manifest provided or found. Discovering images in: {input_dir}")
            # Find all common image file types recursively
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']:
                image_files.extend(input_path.rglob(ext))
        
        logger.info(f"Found {len(image_files)} images to process.")
        
        processed_metadata = []
        
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                # Determine modality based on filename (e.g., "_b_mode.png", "_ceus.png")
                filename = img_path.name.lower()
                modality = None
                if 'b_mode' in filename or 'bmode' in filename:
                    modality = 'b_mode'
                elif 'ceus' in filename:
                    modality = 'ceus'
                
                if modality is None:
                    logger.warning(f"Could not infer modality for {img_path}. Skipping.")
                    self.stats['rejected_count'] += 1
                    continue
                
                # Load image (cv2.imread reads in BGR format)
                image = cv2.imread(str(img_path))
                if image is None:
                    logger.error(f"Could not load image: {img_path}. Skipping.")
                    self.stats['rejected_count'] += 1
                    continue
                    
                # Convert BGR to RGB for consistent internal processing (matplotlib/skimage typically use RGB)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process image through the pipeline
                processed_result = self.process_single_image(image, modality, apply_augmentation_for_training)
                
                if processed_result is not None:
                    # Generate output filename stem (e.g., 'patientID__studyDate__series_instance_ID')
                    # This stem should be robust and unique. Re-using safe_stem from previous scripts.
                    stem = safe_stem(str(img_path)) # Use the safe_stem logic from split_dicom_panels.py
                    
                    if isinstance(processed_result, list):
                        # Multiple augmented versions
                        for i, aug_img in enumerate(processed_result):
                            aug_output_file_path = output_path / modality / f"{stem}_aug_{i:02d}.png"
                            
                            # Convert back to uint8 for saving (if normalization made it float)
                            if aug_img.dtype != np.uint8:
                                save_img = cv2.normalize(aug_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            else:
                                save_img = aug_img
                            
                            # Convert back to BGR for saving with OpenCV
                            cv2.imwrite(str(aug_output_file_path), cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
                            
                            processed_metadata.append({
                                'original_path': str(img_path),
                                'processed_path': str(aug_output_file_path),
                                'modality': modality,
                                'original_shape': image.shape,
                                'processed_shape': aug_img.shape,
                                'augmented': True,
                                'augmentation_index': i
                            })
                    else:
                        # Single processed image
                        output_file_path = output_path / modality / f"{stem}_processed.png"
                        
                        if processed_result.dtype != np.uint8:
                            save_img = cv2.normalize(processed_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        else:
                            save_img = processed_result
                        
                        cv2.imwrite(str(output_file_path), cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
                        
                        processed_metadata.append({
                            'original_path': str(img_path),
                            'processed_path': str(output_file_path),
                            'modality': modality,
                            'original_shape': image.shape,
                            'processed_shape': processed_result.shape,
                            'augmented': False,
                            'augmentation_index': -1
                        })
                        
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                self.stats['rejected_count'] += 1 # Increment rejected count on error
                continue
        
        # Save metadata to CSV
        metadata_df = pd.DataFrame(processed_metadata)
        metadata_df.to_csv(output_path / 'metadata' / 'processing_metadata.csv', index=False)
        
        # Save preprocessing statistics to JSON
        with open(output_path / 'metadata' / 'preprocessing_stats.json', 'w') as f:
            # Convert numpy arrays/scalars to lists/floats for JSON serialization
            stats_json = {}
            for key, value in self.stats.items():
                if isinstance(value, dict):
                    stats_json[key] = {k: [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in v]
                                       for k, v in value.items()}
                else:
                    stats_json[key] = value
            json.dump(stats_json, f, indent=2)
            
        logger.info(f"Processing complete. Processed: {self.stats['processed_count']}, "
                    f"Rejected: {self.stats['rejected_count']}")
        
        return {
            'processed_count': self.stats['processed_count'],
            'rejected_count': self.stats['rejected_count'],
            'metadata': metadata_df,
            'stats': self.stats
        }

def create_preprocessing_pipeline(config_dict: Optional[Dict] = None) -> UltrasoundPreprocessor:
    """
    Factory function to create an UltrasoundPreprocessor instance with a custom configuration.

    Args:
        config_dict (Optional[Dict]): A dictionary to override default
                                       PreprocessingConfig parameters.

    Returns:
        UltrasoundPreprocessor: An initialized preprocessor instance.
    """
    
    if config_dict is None:
        config = PreprocessingConfig()
    else:
        config = PreprocessingConfig(**config_dict)
    
    return UltrasoundPreprocessor(config)

# Helper function for safe_stem (copied from split_dicom_panels.py for self-containment)
def safe_stem(full_path: str) -> str:
    """
    Generates a safe stem from a full file path for use in filenames.
    Uses the last 6 path components to create a unique, clean stem.
    """
    parts = Path(full_path).parts[-6:] # Adjust this slice if your unique identifier needs more/fewer parts
    stem = "__".join(parts).replace(".dcm", "").replace(":", "_").replace(" ", "_")
    return stem


if __name__ == "__main__":
    # Define configurations for different processing needs (e.g., training vs. evaluation)

    # Configuration for training data (with augmentation)
    training_config = {
        'target_size': (256, 256),
        'apply_noise_reduction': True,
        'apply_clahe': True,
        'remove_annotations': True,
        'augmentation_enabled': True, # Enable augmentation for training
        'rotation_range': (-5, 5),
        'flip_horizontal': True,
        'min_image_area': 5000, # Adjust based on your data characteristics
        'max_image_area': 5000000, # Adjust based on your data characteristics
        'min_intensity_std': 2.0 # Adjust based on your data characteristics
    }
    
    # Configuration for validation/test data (no augmentation)
    eval_config = {
        'target_size': (256, 256),
        'apply_noise_reduction': True,
        'apply_clahe': True,
        'remove_annotations': True,
        'augmentation_enabled': False, # Disable augmentation for evaluation
        'min_image_area': 5000,
        'max_image_area': 5000000,
        'min_intensity_std': 2.0
    }
    
    # Create preprocessors based on configurations
    train_preprocessor = create_preprocessing_pipeline(training_config)
    eval_preprocessor = create_preprocessing_pipeline(eval_config) # Example if you need a separate eval pipeline

    # --- IMPORTANT: Set your input and output directories ---
    # input_directory should point to where your PNGs (split or normalized) are stored.
    # For example, if you ran split_dicom_panels.py and normalize_bmode.py,
    # this might be the output_png_root_path from those scripts.
    input_directory = r"D:\biomedical_research\preprocessing\split_panels" # Example path
    
    # output_directory is where the final processed images and metadata will be saved.
    output_directory_train = r"D:\biomedical_research\preprocessing\processed_dataset_train"
    output_directory_eval = r"D:\biomedical_research\preprocessing\processed_dataset_eval" # For evaluation set

    # Optional: Path to your manifest file (e.g., dicom_manifest_with_labels_split_bmode.xlsx)
    # This manifest should contain the 'full_path' to the PNGs you want to process.
    # If not provided, the script will discover images in input_directory.
    manifest_file_path = r"..\exploration\dicom_manifest_with_labels_split_bmode.xlsx"

    print("\n--- Processing Training Dataset ---")
    results_train = train_preprocessor.process_dataset(
        input_dir=input_directory,
        output_dir=output_directory_train,
        manifest_file=manifest_file_path,
        apply_augmentation_for_training=True # Ensure augmentation is applied for training
    )
    
    print(f"\nTraining Processing Results:")
    print(f"- Processed: {results_train['processed_count']} images")
    print(f"- Rejected: {results_train['rejected_count']} images")
    print(f"- Metadata saved to: {output_directory_train}/metadata/")
    print(f"- Statistics saved to: {output_directory_train}/metadata/preprocessing_stats.json")

    # Example for processing an evaluation/test dataset (without augmentation)
    # You would typically have a separate input_directory for eval data if it's split differently
    # print("\n--- Processing Evaluation Dataset ---")
    # results_eval = eval_preprocessor.process_dataset(
    #     input_dir=input_directory, # Or a different directory for eval images
    #     output_dir=output_directory_eval,
    #     manifest_file=manifest_file_path, # Or a separate manifest for eval
    #     apply_augmentation_for_training=False # Ensure no augmentation for evaluation
    # )
    
    # print(f"\nEvaluation Processing Results:")
    # print(f"- Processed: {results_eval['processed_count']} images")
    # print(f"- Rejected: {results_eval['rejected_count']} images")
    # print(f"- Metadata saved to: {output_directory_eval}/metadata/")
    # print(f"- Statistics saved to: {output_directory_eval}/metadata/preprocessing_stats.json")
