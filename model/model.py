"""
File: model.py
Purpose: A merged script combining the effective DualResNet model with the advanced training, evaluation, and logging framework.
         This script performs a 5-fold cross-validation with robust metric collection,
         on-the-fly data augmentation, and proper test set evaluation. 
"""
import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union 
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
import torchvision.transforms.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Suppress specific warnings that might arise from libraries
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.models._api')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.transforms.functional_tensor')

# =====================================================================================
# 1. Configuration and Setup
# =====================================================================================

@dataclass
class TrainingConfig:
    """Centralized configuration for the training pipeline."""
    # Data parameters
    manifest_path: str = "dicom_manifest_with_labels.xlsx"
    processed_root: str = r"processed_dataset" # Path to folder containing b_mode/ceus subfolders
    
    # Model parameters
    model_architecture: str = "dual_resnet34"
    pretrained: bool = True # Use pre-trained weights for ResNet encoders
    
    # Training parameters
    num_epochs: int = 15
    batch_size: int = 16
    learning_rate: float = 1e-4
    
    # Cross-validation & Splitting
    n_folds: int = 5
    test_size: float = 0.2 # Proportion of data to reserve for final test set
    
    # Infrastructure & Logging
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    early_stopping_patience: int = 3 # Number of epochs to wait for improvement
    experiment_name: str = f"DualResNet_CV_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Reproducibility
    seed: int = 42

def setup_logging(experiment_name: str) -> Path:
    """
    Configures the logging system to save logs and results.
    Creates an 'experiments' directory with a subdirectory for the current experiment.

    Args:
        experiment_name (str): Name of the current experiment, used for log directory.

    Returns:
        Path: The path to the experiment's logging directory.
    """
    log_dir = Path(f"experiments/{experiment_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'), # Log to file
            logging.StreamHandler() # Log to console
        ]
    )
    
    logging.info(f"Logging configured. Saving artifacts to: {log_dir}")
    return log_dir

def setup_device(device_str: str) -> torch.device:
    """
    Sets up the computation device (CPU or GPU) and logs the choice.

    Args:
        device_str (str): Desired device string (e.g., 'cuda', 'cpu').

    Returns:
        torch.device: The PyTorch device object.
    """
    device = torch.device(device_str)
    if device.type == 'cuda':
        if torch.cuda.is_available():
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            # Enable cuDNN benchmark for faster training if input sizes are consistent
            torch.backends.cudnn.benchmark = True
        else:
            logging.warning("CUDA not available, falling back to CPU.")
            device = torch.device('cpu')
            logging.info("Using CPU.")
    else:
        logging.info("Using CPU.")
    return device

# =====================================================================================
# 2. Dataset and Transforms
# =====================================================================================

class HCCDataset(Dataset):
    """
    Dataset class to load paired B-mode and CEUS images for HCC classification.
    It expects processed images (e.g., PNGs) organized in 'b_mode' and 'ceus'
    subdirectories within the `processed_root`.
    Labels are derived from a manifest file.
    """
    def __init__(self, manifest_xlsx: str, processed_root: str, transform=None):
        """
        Initializes the dataset.

        Args:
            manifest_xlsx (str): Path to the Excel manifest file containing image paths and labels.
            processed_root (str): Root directory where processed B-mode and CEUS images are stored.
                                  Expected structure: `processed_root/b_mode/*.png`,
                                  `processed_root/ceus/*.png`.
            transform (callable, optional): A function/transform to apply to the images.
        """
        df = pd.read_excel(manifest_xlsx)
        # Filter out rows where 'full_path' is NaN
        df = df[pd.notna(df['full_path'])]
        
        def compute_stem(fp: str) -> str:
            """
            Computes a unique stem for an image from its full path.
            This stem is used to match paired B-mode and CEUS images and their labels.
            Assumes the stem is derived from the last 6 path components before the .dcm suffix.
            This should match the `safe_stem` logic used in preprocessing.
            """
            parts = Path(fp).parts[-6:] # Adjust this slice if your stem derivation differs
            return '__'.join(parts).replace('.dcm','').replace(':', '_').replace(' ', '_') # Ensure consistency with safe_stem
            
        df['stem'] = df['full_path'].apply(compute_stem)
        # Convert 'Label' column to binary (0 for Non-HCC, 1 for HCC)
        df['label_bin'] = df['Label'].str.lower().eq('hcc').astype(int)
        
        # Create a dictionary to map stems to binary labels
        label_map = dict(zip(df['stem'], df['label_bin']))

        root = Path(processed_root)
        # List all processed B-mode and CEUS PNGs
        b_paths = list((root/'b_mode').glob('*_processed.png')) # Assuming processed images end with _processed.png
        c_paths = list((root/'ceus').glob('*_processed.png'))
        
        def stem_from_processed_path(p: Path, modality_suffix: str) -> Optional[str]:
            """
            Extracts the stem from a processed image filename.
            Example: 'patientID__studyDate__series_instance_ID_b_mode_processed.png' -> 'patientID__studyDate__series_instance_ID'
            """
            s = p.stem # Gets 'patientID__studyDate__series_instance_ID_b_mode_processed'
            tag = f"_{modality_suffix}_processed"
            return s[:-len(tag)] if s.endswith(tag) else None
            
        # Create maps from stem to full path for B-mode and CEUS images
        b_map = {stem_from_processed_path(p, 'b_mode'):p for p in b_paths if stem_from_processed_path(p, 'b_mode')}
        c_map = {stem_from_processed_path(p, 'ceus'):p for p in c_paths if stem_from_processed_path(p, 'ceus')}

        self.samples = []
        # Pair B-mode and CEUS images by their common stem and ensure a label exists
        for st, b_path in b_map.items():
            c_path = c_map.get(st)
            lbl = label_map.get(st)
            if c_path and lbl is not None:
                self.samples.append((b_path, c_path, lbl))
        
        if not self.samples:
            raise RuntimeError(f"No paired samples found in '{processed_root}'. "
                               f"Please check your manifest and processed image paths. "
                               f"Found {len(b_paths)} B-mode and {len(c_paths)} CEUS images.")
        
        self.transform = transform
        logging.info(f"Dataset initialized with {len(self.samples)} paired samples.")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the B-mode image, CEUS image, and their corresponding label for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
            the transformed B-mode image, transformed CEUS image, and the label.
        """
        b_path, c_path, label = self.samples[idx]
        
        # Open images using PIL and convert to RGB (even if grayscale, for consistency)
        img_b = Image.open(b_path).convert('RGB')
        img_c = Image.open(c_path).convert('RGB')
        
        if self.transform:
            # Apply transforms (including augmentation)
            img_b, img_c = self.transform(img_b, img_c)
            
        return img_b, img_c, torch.tensor(label, dtype=torch.long)

class PairedTransform:
    """
    Applies synchronized data augmentation transforms to paired B-mode and CEUS images.
    Ensures that both images undergo the same random transformations (e.g., rotation, flip).
    """
    def __init__(self, size: Tuple[int, int] = (256, 256)):
        """
        Initializes the paired transform.

        Args:
            size (Tuple[int, int]): Target size for resizing the images.
        """
        self.resize = transforms.Resize(size)
        self.to_tensor = transforms.ToTensor()
        # Normalization for pre-trained models (often expects values in [-1, 1] or [0, 1])
        # [0.5]*3, [0.5]*3 normalizes to [-1, 1] range for 3 channels
        self.norm = transforms.Normalize([0.5]*3, [0.5]*3) 
        
    def __call__(self, x1: Image.Image, x2: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies paired transforms to two input PIL images.

        Args:
            x1 (Image.Image): First input image (e.g., B-mode).
            x2 (Image.Image): Second input image (e.g., CEUS).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed and normalized PyTorch tensors.
        """
        # Apply synchronized augmentations
        # Generate random parameters once for both images
        angle = random.uniform(-10, 10)
        hflip_apply = random.random() < 0.5
        
        if hflip_apply:
            x1 = F.hflip(x1)
            x2 = F.hflip(x2)
        
        x1 = F.rotate(x1, angle)
        x2 = F.rotate(x2, angle)
        
        # Resize, convert to tensor, and normalize
        x1 = self.resize(x1)
        x2 = self.resize(x2)
        x1 = self.to_tensor(x1)
        x2 = self.to_tensor(x2)
        
        return self.norm(x1), self.norm(x2)

class EvalTransform:
    """
    Simple transform for validation/testing, applying only resizing,
    conversion to tensor, and normalization, without augmentation.
    """
    def __init__(self, size: Tuple[int, int] = (256, 256)):
        """
        Initializes the evaluation transform.

        Args:
            size (Tuple[int, int]): Target size for resizing the images.
        """
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    def __call__(self, x1: Image.Image, x2: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies evaluation transforms to two input PIL images.

        Args:
            x1 (Image.Image): First input image (e.g., B-mode).
            x2 (Image.Image): Second input image (e.g., CEUS).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed and normalized PyTorch tensors.
        """
        return self.transform(x1), self.transform(x2)

# =====================================================================================
# 3. Model Architecture
# =====================================================================================

class DualResNet(nn.Module):
    """
    The Dual-Path ResNet-34 model for processing paired B-mode and CEUS images.
    It uses two separate ResNet-34 encoders (one for each modality) and
    concatenates their features before passing them to a shared classifier head.
    """
    def __init__(self, pretrained: bool = True):
        """
        Initializes the DualResNet model.

        Args:
            pretrained (bool): If True, use pre-trained ImageNet weights for ResNet encoders.
        """
        super().__init__()
        
        # B-mode stream: ResNet-34 encoder
        b_net = models.resnet34(pretrained=pretrained)
        # Remove the final classification layer and average pooling layer to get features
        self.b_encoder = nn.Sequential(*list(b_net.children())[:-1]) 
        
        # CEUS stream: ResNet-34 encoder (separate weights from B-mode)
        c_net = models.resnet34(pretrained=pretrained)
        # Remove the final classification layer and average pooling layer
        self.c_encoder = nn.Sequential(*list(c_net.children())[:-1])
        
        # Classifier head: takes concatenated features from both encoders
        # ResNet-34's last layer before avg pool is 512 features. After avg pool, it's 512.
        # So, combined features will be 512 (from B-mode) + 512 (from CEUS) = 1024.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256), # Input 1024 features, output 256
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout for regularization
            nn.Linear(256, 2) # Output 2 classes (e.g., Non-HCC, HCC)
        )
    
    def forward(self, x_b: torch.Tensor, x_c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DualResNet model.

        Args:
            x_b (torch.Tensor): Input tensor for B-mode images.
            x_c (torch.Tensor): Input tensor for CEUS images.

        Returns:
            torch.Tensor: Logits for the two classes.
        """
        # Pass through encoders and flatten
        feat_b = self.b_encoder(x_b).flatten(1)
        feat_c = self.c_encoder(x_c).flatten(1)
        
        # Concatenate features from both modalities
        combined_features = torch.cat([feat_b, feat_c], dim=1)
        
        # Pass through classifier head
        output = self.classifier(combined_features)
        return output

# =====================================================================================
# 4. Metrics Logger
# =====================================================================================

class ComprehensiveMetricsLogger:
    """
    Advanced metrics logging with plotting and reporting capabilities.
    Saves epoch-wise metrics, fold-wise results, ROC curves, confusion matrices,
    and training history plots.
    """
    def __init__(self, log_dir: Path):
        """
        Initializes the metrics logger.

        Args:
            log_dir (Path): The root directory for saving logs and plots.
        """
        self.log_dir = log_dir
        self.metrics_dir = log_dir / "metrics"
        self.plots_dir = log_dir / "plots"
        self.metrics_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Stores epoch-level metrics (e.g., loss, accuracy per epoch)
        self.epoch_metrics = defaultdict(lambda: defaultdict(list))
        # Stores final results for each fold
        self.fold_results = {}

    def log_epoch(self, fold: int, epoch: int, phase: str, metrics: Dict[str, float]):
        """
        Logs metrics for a specific epoch and phase (e.g., 'train', 'val').

        Args:
            fold (int): Current fold number (0-indexed).
            epoch (int): Current epoch number (0-indexed).
            phase (str): Phase of the epoch ('train' or 'val').
            metrics (Dict[str, float]): Dictionary of metric names and their values.
        """
        for key, value in metrics.items():
            self.epoch_metrics[f"fold_{fold}_{phase}"][key].append(value)
        self.save_epoch_metrics() # Save after each epoch for persistence

    def log_fold_results(self, fold: int, results: Dict[str, Any]):
        """
        Logs the final results for a completed cross-validation fold.
        Also triggers plotting of ROC curve, confusion matrix, and training history.

        Args:
            fold (int): Current fold number (0-indexed).
            results (Dict[str, Any]): Dictionary containing final metrics and data
                                      (e.g., 'fpr', 'tpr', 'confusion_matrix').
        """
        self.fold_results[f"fold_{fold+1}"] = results
        self._plot_roc_curve(fold + 1, results)
        self._plot_confusion_matrix(fold + 1, results)
        self._plot_training_history(fold + 1)
        self.save_fold_metrics() # Save after each fold for persistence

    def _plot_roc_curve(self, fold: int, results: Dict[str, Any]):
        """Plots and saves the Receiver Operating Characteristic (ROC) curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(results['fpr'], results['tpr'], label=f"AUC = {results['auc']:.3f}")
        plt.plot([0, 1], [0, 1], 'k--', label='Random Chance') # Diagonal line for random classifier
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Fold {fold}'); plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.plots_dir / f'roc_curve_fold_{fold}.png', dpi=300)
        plt.close() # Close plot to free memory

    def _plot_confusion_matrix(self, fold: int, results: Dict[str, Any]):
        """Plots and saves the Confusion Matrix."""
        plt.figure(figsize=(8, 6))
        # Ensure confusion_matrix is a numpy array if it was converted to list for JSON
        cm = np.array(results['confusion_matrix']) if isinstance(results['confusion_matrix'], list) else results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-HCC', 'HCC'], yticklabels=['Non-HCC', 'HCC'])
        plt.title(f'Confusion Matrix - Fold {fold}')
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.savefig(self.plots_dir / f'confusion_matrix_fold_{fold}.png', dpi=300)
        plt.close()

    def _plot_training_history(self, fold: int):
        """Plots and saves the training and validation loss/accuracy history over epochs."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Access metrics for the current fold (note: fold in epoch_metrics is 0-indexed)
        train_loss = self.epoch_metrics[f"fold_{fold-1}_train"]["loss"]
        val_loss = self.epoch_metrics[f"fold_{fold-1}_val"]["loss"]
        train_acc = self.epoch_metrics[f"fold_{fold-1}_train"]["accuracy"]
        val_acc = self.epoch_metrics[f"fold_{fold-1}_val"]["accuracy"]
        epochs = range(1, len(train_loss) + 1)

        # Plot Loss
        ax1.plot(epochs, train_loss, 'b-o', label='Training Loss')
        ax1.plot(epochs, val_loss, 'r-o', label='Validation Loss')
        ax1.set_title(f'Loss vs. Epochs - Fold {fold}'); ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

        # Plot Accuracy
        ax2.plot(epochs, train_acc, 'b-o', label='Training Accuracy')
        ax2.plot(epochs, val_acc, 'r-o', label='Validation Accuracy')
        ax2.set_title(f'Accuracy vs. Epochs - Fold {fold}'); ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'training_history_fold_{fold}.png', dpi=300)
        plt.close()

    def save_epoch_metrics(self):
        """Saves all collected epoch-level metrics to a JSON file."""
        with open(self.metrics_dir / 'epoch_metrics.json', 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2)

    def save_fold_metrics(self):
        """Saves all collected fold-level results to a JSON file."""
        with open(self.metrics_dir / 'fold_metrics.json', 'w') as f:
            # Convert numpy arrays (like fpr, tpr, confusion_matrix) to lists for JSON serialization
            serializable_results = {}
            for fold, data in self.fold_results.items():
                serializable_results[fold] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}
            json.dump(serializable_results, f, indent=2)

# =====================================================================================
# 5. Training and Evaluation Logic
# =====================================================================================

def safe_metric(func: Any, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
    """
    Calculates a metric safely, returning NaN if an error occurs (e.g., due to single class presence).

    Args:
        func (callable): The metric function (e.g., accuracy_score, precision_score).
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels or probabilities.
        **kwargs: Additional keyword arguments for the metric function.

    Returns:
        float: The calculated metric value or NaN if an error occurs.
    """
    try:
        return func(y_true, y_pred, **kwargs)
    except ValueError as e:
        logging.warning(f"Could not compute metric {func.__name__}: {e}. Returning NaN.")
        return float('nan')
    except Exception as e:
        logging.error(f"Unexpected error computing metric {func.__name__}: {e}. Returning NaN.")
        return float('nan')

def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
              optimizer: Optional[optim.Optimizer], device: torch.device, 
              is_training: bool) -> Union[Dict[str, float], Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]]:
    """
    Runs a single epoch of training or evaluation.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): DataLoader for the current phase (train/val).
        criterion (nn.Module): Loss function.
        optimizer (Optional[optim.Optimizer]): Optimizer (only for training phase).
        device (torch.device): Computation device ('cuda' or 'cpu').
        is_training (bool): True for training phase, False for evaluation phase.

    Returns:
        Union[Dict[str, float], Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]]:
        For training, returns a dictionary of metrics. For evaluation, returns
        metrics dictionary along with true labels, predicted probabilities, and predicted labels.
    """
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    # Context manager for gradient computation
    context = torch.enable_grad() if is_training else torch.no_grad()
    
    pbar_desc = "Training" if is_training else "Validating"
    pbar = tqdm(loader, desc=pbar_desc, leave=False)

    with context:
        for x_b, x_c, y in pbar:
            x_b, x_c, y = x_b.to(device), x_c.to(device), y.to(device)

            if is_training:
                if optimizer: # Ensure optimizer exists for training
                    optimizer.zero_grad()

            outputs = model(x_b, x_c)
            loss = criterion(outputs, y)

            if is_training:
                loss.backward()
                if optimizer:
                    optimizer.step()

            total_loss += loss.item() * y.size(0) # Accumulate loss weighted by batch size
            
            # Detach from graph, move to CPU, and convert to NumPy for metric calculation
            probs = torch.softmax(outputs, 1)[:, 1].detach().cpu().numpy() # Probability of the positive class
            preds = outputs.argmax(1).detach().cpu().numpy() # Predicted class (0 or 1)
            
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)
            
            pbar.set_postfix(loss=loss.item()) # Update progress bar with current batch loss

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    # Calculate metrics for the epoch
    metrics = {
        'loss': total_loss / len(y_true),
        'accuracy': safe_metric(accuracy_score, y_true, y_pred),
        'precision': safe_metric(precision_score, y_true, y_pred, zero_division=0),
        'recall': safe_metric(recall_score, y_true, y_pred, zero_division=0),
        'f1': safe_metric(f1_score, y_true, y_pred, zero_division=0),
        'auc': safe_metric(roc_auc_score, y_true, y_prob)
    }
    
    # For validation/test phase, return raw predictions for further analysis (e.g., ROC curve)
    if not is_training:
        return metrics, y_true, y_prob, y_pred
        
    return metrics

# =====================================================================================
# 6. Main Cross-Validation and Testing Pipeline
# =====================================================================================

def run_pipeline(config: TrainingConfig):
    """
    Main function to run the entire cross-validation and testing pipeline.
    Orchestrates data loading, model training, validation, early stopping,
    model saving, and final test set evaluation.
    """
    # Setup logging and device
    log_dir = setup_logging(config.experiment_name)
    device = setup_device(config.device)
    logger_obj = ComprehensiveMetricsLogger(log_dir) # Renamed to avoid conflict with logging module

    # Set seeds for reproducibility across runs
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed) # For multi-GPU setups

    # Save the current configuration to the experiment directory
    with open(log_dir / 'config.json', 'w') as f:
        json.dump(asdict(config), f, indent=2)
    logging.info(f"Configuration saved to {log_dir / 'config.json'}")

    # Load the full dataset
    full_dataset = HCCDataset(config.manifest_path, config.processed_root)
    # Extract labels for stratified splitting
    labels = [sample[2] for sample in full_dataset.samples]
    indices = list(range(len(labels)))

    # --- Initial Train/Test Split (stratified to maintain class distribution) ---
    train_val_idx, test_idx, train_val_labels, test_labels = train_test_split(
        indices, labels,
        test_size=config.test_size,
        stratify=labels,
        random_state=config.seed
    )
    logging.info(f"Data split: {len(train_val_idx)} samples for Train/Validation, "
                 f"{len(test_idx)} samples for Test.")

    # --- K-Fold Cross-Validation on the Train/Validation set ---
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    
    all_fold_best_metrics = [] # To store best validation metrics for each fold

    # Iterate through each fold
    for fold, (train_split_idx, val_split_idx) in enumerate(skf.split(train_val_idx, train_val_labels)):
        logging.info(f"\n{'='*50}\nStarting Fold {fold + 1}/{config.n_folds}\n{'='*50}")
        
        # Map indices back to the original full_dataset
        fold_train_original_idx = [train_val_idx[i] for i in train_split_idx]
        fold_val_original_idx = [train_val_idx[i] for i in val_split_idx]

        # Create Subset datasets for the current fold's train and validation sets
        # IMPORTANT: Assign transforms directly to the Subset's dataset attribute
        # This ensures the correct transform is used for each subset.
        train_dataset_subset = Subset(full_dataset, fold_train_original_idx)
        val_dataset_subset = Subset(full_dataset, fold_val_original_idx)
        
        # Assign transforms for the current fold's subsets
        # Note: These transforms will be applied to the underlying full_dataset when accessed via subset
        # This is a common pattern when using Subset with a single underlying Dataset instance.
        # It's crucial that `transform` is an attribute of the `full_dataset` object itself.
        # If `full_dataset` had `transform` as a property or was immutable, this approach would fail.
        # Given the HCCDataset's __init__ and __getitem__ structure, this should work.
        full_dataset.transform = PairedTransform() # For training data with augmentation
        train_loader = DataLoader(train_dataset_subset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        full_dataset.transform = EvalTransform() # For validation data without augmentation
        val_loader = DataLoader(val_dataset_subset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Initialize Model, Loss, and Optimizer for the current fold
        model = DualResNet(pretrained=config.pretrained).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop for the current fold
        for epoch in range(config.num_epochs):
            logging.info(f"\n--- Fold {fold+1}, Epoch {epoch+1}/{config.num_epochs} ---")
            
            # Training phase
            # Temporarily set transform for training
            full_dataset.transform = PairedTransform()
            train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, is_training=True)
            logger_obj.log_epoch(fold, epoch, 'train', train_metrics)
            logging.info(f"Train -> Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}")

            # Validation phase
            # Temporarily set transform for validation
            full_dataset.transform = EvalTransform()
            val_metrics, val_y_true, val_y_prob, val_y_pred = run_epoch(model, val_loader, criterion, None, device, is_training=False)
            logger_obj.log_epoch(fold, epoch, 'val', val_metrics)
            logging.info(f"Valid -> Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")

            # Early Stopping logic and model saving
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save the best model state dictionary
                torch.save(model.state_dict(), log_dir / f'best_model_fold_{fold+1}.pth')
                logging.info(f"Validation loss improved. Saved best model for fold {fold+1}.")
            else:
                patience_counter += 1
                logging.info(f"Validation loss did not improve. Patience: {patience_counter}/{config.early_stopping_patience}")

            if patience_counter >= config.early_stopping_patience:
                logging.info("Early stopping triggered for this fold.")
                break # Exit epoch loop for current fold
        
        # After training for the fold, load the best model and evaluate one last time
        # on the validation set to capture final metrics for plotting/reporting
        best_model_path = log_dir / f'best_model_fold_{fold+1}.pth'
        if best_model_path.exists():
            best_model = DualResNet(pretrained=config.pretrained).to(device)
            best_model.load_state_dict(torch.load(best_model_path))
            
            # Set transform for validation
            full_dataset.transform = EvalTransform()
            final_val_metrics, y_true_fold, y_prob_fold, y_pred_fold = run_epoch(best_model, val_loader, criterion, None, device, is_training=False)
            
            # Add detailed metrics for logging (confusion matrix, MCC, ROC curve data)
            final_val_metrics['confusion_matrix'] = confusion_matrix(y_true_fold, y_pred_fold).tolist() # Convert to list for JSON
            final_val_metrics['mcc'] = safe_metric(matthews_corrcoef, y_true_fold, y_pred_fold)
            fpr, tpr, _ = roc_curve(y_true_fold, y_prob_fold)
            final_val_metrics['fpr'] = fpr.tolist() # Convert to list for JSON
            final_val_metrics['tpr'] = tpr.tolist() # Convert to list for JSON
            
            logger_obj.log_fold_results(fold, final_val_metrics)
            all_fold_best_metrics.append(final_val_metrics)
            logging.info(f"Fold {fold+1} completed. Best validation AUC: {final_val_metrics['auc']:.4f}")
        else:
            logging.error(f"Best model for fold {fold+1} not found at {best_model_path}. Skipping final validation metrics for this fold.")

    # --- Final Test Set Evaluation ---
    logging.info(f"\n{'='*50}\nStarting Final Test Set Evaluation\n{'='*50}")
    
    # Create test dataset and loader with evaluation transforms
    test_dataset_subset = Subset(full_dataset, test_idx)
    full_dataset.transform = EvalTransform() # Ensure eval transform for test set
    test_loader = DataLoader(test_dataset_subset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    all_test_probs = [] # To store probabilities from each fold's best model on the test set

    # Evaluate each best model from each fold on the test set
    for fold in range(config.n_folds):
        model_path = log_dir / f'best_model_fold_{fold+1}.pth'
        if not model_path.exists():
            logging.warning(f"Model for fold {fold+1} not found at {model_path}. Skipping test evaluation for this fold.")
            continue

        model = DualResNet(pretrained=config.pretrained).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval() # Set to evaluation mode
        
        # Run epoch on test loader to get probabilities
        _, _, fold_probs, _ = run_epoch(model, test_loader, nn.CrossEntropyLoss(), None, device, is_training=False)
        all_test_probs.append(fold_probs)
    
    if not all_test_probs:
        logging.error("No test probabilities collected. Cannot perform ensemble evaluation.")
        test_metrics = {}
    else:
        # Ensemble predictions by averaging probabilities across folds
        ensemble_probs = np.mean(all_test_probs, axis=0)
        ensemble_preds = (ensemble_probs > 0.5).astype(int) # Threshold at 0.5 for binary prediction

        # Calculate final test metrics
        test_metrics = {
            'accuracy': safe_metric(accuracy_score, test_labels, ensemble_preds),
            'precision': safe_metric(precision_score, test_labels, ensemble_preds, zero_division=0),
            'recall': safe_metric(recall_score, test_labels, ensemble_preds, zero_division=0),
            'f1': safe_metric(f1_score, test_labels, ensemble_preds, zero_division=0),
            'auc': safe_metric(roc_auc_score, test_labels, ensemble_probs),
            'mcc': safe_metric(matthews_corrcoef, test_labels, ensemble_preds),
            'confusion_matrix': confusion_matrix(test_labels, ensemble_preds).tolist() # Convert to list for JSON
        }
        logging.info(f"Test Set (Ensemble) -> Acc: {test_metrics['accuracy']:.4f}, AUC: {test_metrics['auc']:.4f}")
    
    # Save test results to a JSON file
    with open(log_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # --- Generate Final Summary Report ---
    summary_report = {
        "experiment_name": config.experiment_name,
        "config": asdict(config), # Convert dataclass to dict for JSON
        "cross_validation_summary": {},
        "test_set_results": test_metrics
    }
    
    # Calculate mean and std for CV metrics
    if all_fold_best_metrics:
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']:
            values = [m.get(metric_name, float('nan')) for m in all_fold_best_metrics] # Use .get for robustness
            # Filter out NaNs if any
            values = [v for v in values if not np.isnan(v)]
            if values:
                summary_report["cross_validation_summary"][metric_name] = {
                    "mean": float(np.mean(values)), # Convert numpy types to native Python types
                    "std": float(np.std(values))
                }
            else:
                 summary_report["cross_validation_summary"][metric_name] = {"mean": float('nan'), "std": float('nan')}
    else:
        logging.warning("No fold metrics collected for CV summary.")

    with open(log_dir / 'final_summary_report.json', 'w') as f:
        json.dump(summary_report, f, indent=2)

    logging.info(f"\n{'='*50}\nPipeline Finished Successfully!\n{'='*50}")
    logging.info(f"All results, logs, and plots are saved in: {log_dir}")
    
    if "accuracy" in summary_report['cross_validation_summary']:
        logging.info("\nCV Mean Accuracy: {:.4f} +/- {:.4f}".format(
            summary_report['cross_validation_summary']['accuracy']['mean'],
            summary_report['cross_validation_summary']['accuracy']['std']
        ))
    if "auc" in summary_report['cross_validation_summary']:
        logging.info("CV Mean AUC: {:.4f} +/- {:.4f}".format(
            summary_report['cross_validation_summary']['auc']['mean'],
            summary_report['cross_validation_summary']['auc']['std']
        ))
    
    if "accuracy" in test_metrics:
        logging.info("\nEnsemble Test Accuracy: {:.4f}".format(test_metrics['accuracy']))
    if "auc" in test_metrics:
        logging.info("Ensemble Test AUC: {:.4f}".format(test_metrics['auc']))


if __name__ == '__main__':
    # Initialize and run the pipeline with your configuration
    # IMPORTANT: Adjust these paths to your actual data locations
    config = TrainingConfig(
        # Path to your manifest file (e.g., generated by add_labels_to_manifest.py)
        # This file should contain 'full_path' and 'Label' columns.
        manifest_path="../exploration/dicom_manifest_with_labels.xlsx", 
        
        # Root directory where your processed B-mode and CEUS images are located.
        # This is typically the output of your preprocessing pipeline (e.g., preprocess_ultrasound.py).
        # Expected structure: processed_root/b_mode/*.png and processed_root/ceus/*.png
        processed_root=r"D:\biomedical_research\preprocessing\processed_dataset_train", # Example path
        
        num_epochs=15,
        batch_size=16,
        n_folds=5,
        early_stopping_patience=3,
        seed=42 # Ensure reproducibility
    )
    
    run_pipeline(config)
