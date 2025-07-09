# Cancer Detection Pipeline

This repository contains the code and resources for a complete biomedical ultrasound imaging pipeline, from raw DICOM files through data exploration, preprocessing, model training, and deployment.

## Project Overview

This project implements a comprehensive pipeline for processing biomedical ultrasound data, specifically focusing on liver cancer detection using paired B-mode and CEUS (Contrast-Enhanced Ultrasound) images.

*Key components*:
- **Exploration**: Initial data analysis, metadata extraction from DICOM files, and manifest generation.
- **Preprocessing**: Scripts to convert DICOMs to PNGs, split two-panel images, normalize intensities, and extract regions of interest (ROIs).
- **Training**: Implementation of a Dual-Path ResNet model with 5-fold cross-validation, data augmentation, and robust metric logging for classification.
- **Deployment**: Tools for optimized model inference, including quantization, ONNX export, performance benchmarking, and a Gradio-based web interface for clinical use.
- **Documentation & Analysis**: Scripts for generating various plots and summaries of the data and training experiments.
## Repository Structure

```
.
├── deployment/                     # Model deployment scripts and sample inputs
│   ├── deploy.py                   # Main deployment script (inference, UI, profiling)
│   └── flagged/                    # Directory for flagged/sample outputs 
│       ├── B-mode Ultrasound/
│       ├── CEUS Ultrasound/
│       ├── log.csv
│       └── Results Visualization/
├── documentation/                  # Scripts and notebooks for data/model analysis and reporting
│   ├── draw_graphs.py              # Script to generate various plots from manifests/logs
│   ├── graphs.ipynb                # Original notebook for graph generation
│   └── preprocessing_stats.json    # Statistics from preprocessing
├── exploration/                    # Initial data exploration and manifest generation
│   ├── _01_build_dicom_manifest.py # Script to build initial DICOM manifest
│   ├── _02_add_labels_to_manifest.py # Script to add labels to the manifest
│   ├── _03_split_panels.py         # Script to update manifest after panel splitting
│   ├── B-mode-and-CEUS-Liver_ReferenceStandards_v2_20220218.xlsx # Reference labels
│   ├── dicom_manifest.xlsx         # Output: Initial DICOM manifest
│   ├── dicom_manifest_with_labels.xlsx # Output: Manifest with labels
│   ├── dicom_manifest_with_labels_split.xlsx # Output: Manifest with split panel info
│   └── dicom_manifest_with_labels_split_bmode.xlsx # Output: Manifest with normalized B-mode info
├── LICENSE                         # Project license file
├── model/                          # Model definition and training scripts
│   └── model.py                    # Main model training and cross-validation pipeline
├── preprocessing/                  # Image preprocessing scripts
│   ├── _01_split_dicom_panels.py   # Script to split two-panel DICOMs into B-mode/CEUS PNGs
│   ├── _02_preprocess_ultrasound.py # Comprehensive image preprocessing pipeline
│   ├── _03_extract_roi.py          # Script for Region of Interest (ROI) extraction
├── README.md                       # This README file
├── requirements.txt                # Python package dependencies


# Note: The `training/` directory and its subcontents (experiments, best models, metrics, plots 
# are generated during the model training process (by `model/model.py`).
```

## Installation

1. **Clone the repository:**
    
    ```
    git https://github.com/Asiku-Roy-Alia/Liver-Cancer-Detection.git
    cd biomedical-imaging-pipeline
    ```
    
2. **Create a virtual environment (recommended):**
    
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    
3. **Install dependencies:**
    
    ```
    pip install -r requirements.txt
    ```
    

## Usage

The pipeline is designed to be run in a sequential manner. Ensure you adjust input and output paths within each script's `if __name__ == "__main__":` block or via command-line arguments as specified.

### Exploration

Start by building your initial DICOM manifest and adding labels.

1. **Build DICOM Manifest:**
    
    ```
    python exploration/_01_build_dicom_manifest.py "path/to/your/raw_dicom_data" "exploration/dicom_manifest.xlsx"
    ```
    
    (Replace `path/to/your/raw_dicom_data` with your actual DICOM root folder.)
    
2. **Add Labels to Manifest:**
    
    ```
    python exploration/_02_add_labels_to_manifest.py "exploration/dicom_manifest.xlsx" \
                                                   "exploration/B-mode-and-CEUS-Liver_ReferenceStandards_v2_20220218.xlsx" \
                                                   "exploration/dicom_manifest_with_labels.xlsx"
    ```
    

### Preprocessing

Prepare your images for model training.

1. **Split Two-Panel DICOMs (if applicable):**
    
    This script performs the _actual image splitting_ into PNGs.
    
    ```
    python preprocessing/_01_split_dicom_panels.py "exploration/dicom_manifest_with_labels.xlsx" \
                                                  "path/to/output/split_panels_images"
    ```
    
    (The `path/to/output/split_panels_images` should be where the `_b_mode.png` and `_ceus.png` files are saved.)
    
2. **Update Manifest with Split Panel Info:**
    
    This script updates the manifest to reflect the newly split images.
    
    ```
    python exploration/_03_split_panels.py "exploration/dicom_manifest_with_labels.xlsx" \
                                           "path/to/output/split_panels_images" \
                                           "exploration/dicom_manifest_with_labels_split.xlsx"
    ```
    
3. **Extract Region of Interest (ROI):**
    
    This step processes the split images to extract the relevant scan area.
    
    ```
    python preprocessing/_02_extract_roi.py "path/to/output/split_panels_images" \
                                            "path/to/output/roi_images"
    ```
    
4. **Comprehensive Ultrasound Preprocessing:**
    
    Applies noise reduction, contrast enhancement, normalization, etc.
    
    ```
    python preprocessing/_03_preprocess_ultrasound.py
    # Modify input/output paths within the script's __main__ block for this one.
    ```
    

### Training

Train your Dual-Path ResNet model.

1. **Run Model Training and Cross-Validation:**
    
    ```
    python model/model.py
    # Adjust manifest_path and processed_root within the script's __main__ block.
    ```
    
    This script will save trained models, metrics, and plots in the `training/experiments/` directory.
    

### Deployment

Deploy your trained model for inference.

1. **Run Deployment Utilities:**
    
    The `deploy.py` script offers various modes:
    
    - **Single Prediction:**
        
        ```
        python deployment/deploy.py --mode predict_single --model_path "path/to/best_model.pth" \
                                   --config_path "path/to/config.json" \
                                   --b_mode_image "path/to/b_mode_sample.png" \
                                   --ceus_image "path/to/ceus_sample.png"
        ```
        
    - **Launch Gradio Web UI:**
        
        ```
        python deployment/deploy.py --mode gradio --model_path "path/to/best_model.pth" \
                                   --config_path "path/to/config.json"
        ```
        
        (This will output a local URL.)
        
    - **Benchmark Performance:**
        
        ```
        python deployment/deploy.py --mode benchmark --model_path "path/to/best_model.pth" \
                                   --config_path "path/to/config.json"
        ```
        
    - **Export to ONNX:**
        
        ```
        python deployment/deploy.py --mode export_onnx --model_path "path/to/best_model.pth" \
                                   --config_path "path/to/config.json" --output_path "exported_model.onnx"
        ```
        
    - **Create Mobile Model:**
        
        ```
        python deployment/deploy.py --mode create_mobile --model_path "path/to/best_model.pth" \
                                   --config_path "path/to/config.json" --output_path "mobile_model.ptl"
        ```
        
    - **Clinical Report Generation:**
        
        ```
        python deployment/deploy.py --mode clinical --model_path "path/to/best_model.pth" \
                                   --config_path "path/to/config.json" \
                                   --b_mode_image "path/to/b_mode_sample.png" \
                                   --ceus_image "path/to/ceus_sample.png" \
                                   --patient_id "P001" --output_path "clinical_reports/P001_report.txt"
        ```
        

### Documentation & Analysis

Generate visualizations and summary reports.

1. **Draw Graphs from Manifests/Logs:**
    
    ```
    python documentation/draw_graphs.py --manifest_path "exploration/dicom_manifest.xlsx" \
                                       --raw_data_path "training/experiments/DualResNet_CV_YYYYMMDD_HHMMSS/metrics/epoch_metrics.json" \
                                       --output_dir "documentation/graphs"
    ```
    
    (Adjust `raw_data_path` to point to the actual `epoch_metrics.json` file generated by your training run.)
    

## Contributing

Contributions are welcome! Please submit issues or pull requests at:

[https://github.com/Asiku-Roy-Alia/Liver-Cancer-Detection](https://github.com/Asiku-Roy-Alia/Liver-Cancer-Detection)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or feedback, please contact [amroy776@gmail.com](mailto:amroy776@gmail.com "null").