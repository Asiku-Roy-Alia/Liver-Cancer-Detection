# Biomedical Imaging Pipeline

This repository contains the code and resources for a complete biomedical imaging pipeline, including data exploration, preprocessing, model training, documentation, and deployment.

## Table of Contents

* [Project Overview](#project-overview)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Usage](#usage)

  * [Exploration](#exploration)
  * [Preprocessing](#preprocessing)
  * [Training](#training)
  * [Deployment](#deployment)
* [Documentation](#documentation)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

## Project Overview

This project implements a pipeline for processing biomedical ultrasound data, from raw DICOM files through model training and deployment.

Key components:

* **Exploration**: Initial data analysis and visualization of DICOM manifests and sample images.
* **Preprocessing**: Scripts and notebooks to convert, mask, and split images for downstream tasks.
* **Training**: Model definitions, training scripts, cross-validation experiments, and analysis of results.
* **Deployment**: Simple scripts and sample inputs to deploy the trained model for inference.

## Repository Structure

```text
code/
├── exploration/           # Data exploration notebooks and manifests
├── preprocessing/         # Image conversion, ROI extraction, and splitting scripts
├── training/              # Model code, training scripts, and experiment outputs
│   └── experiments/       # Cross-validation experiment folders
├── documentation/         # Logs, statistics, and auxiliary documentation
├── deployment/            # Deployment scripts and sample outputs
├── README.md              # This file
└── ls_R.txt               # Directory listing
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/biomedical-imaging-pipeline.git
   cd biomedical-imaging-pipeline/code
   ```
2. Create a Python virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate       # Linux/macOS
   venv\Scripts\activate.bat    # Windows
   pip install -r requirements.txt
   ```

> **Note:** Generate a `requirements.txt` listing packages like `numpy`, `pandas`, `opencv-python`, `torch`, `albumentations`, etc., from your working environment.

## Usage

### Exploration

* Navigate to `exploration/` and run the Jupyter notebooks:

  * `visualize_dicom.ipynb` for initial DICOM visualization
  * Inspect the Excel manifests for data summaries

### Preprocessing

* In `preprocessing/`:

  * `convert_to_png.py`: Convert DICOM to PNG images
  * `roi_extraction.ipynb`: Extract regions of interest
  * `split_images.ipynb`: Split data into training/validation/test sets

### Training

* In `training/`:

  * `model.py` and `model_5fold.py`: Define network architectures and CV routines
  * `analysis.py`: Compute metrics from cross-validation and generate plots
  * `model_final.py`: Train final model on full dataset
  * Experiment outputs saved under `training/experiments/`

### Deployment

* In `deployment/`:

  * `deploy.py`: Load a saved model and perform inference on new samples
  * Sample images (`sample_b.png`, `sample_c.png`) demonstrate expected input formats

## Documentation

* See `documentation/` for generated stats, logs, and auxiliary spreadsheets.

## Contributing

Contributions are welcome! Please submit issues or pull requests at:

[https://github.com/your-username/biomedical-imaging-pipeline](https://github.com/your-username/biomedical-imaging-pipeline)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions or feedback, contact:

* **Your Name** ([youremail@example.com](mailto:youremail@example.com))
* GitHub: [@your-username](https://github.com/your-username)
