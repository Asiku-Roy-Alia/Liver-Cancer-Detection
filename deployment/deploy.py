"""
File: deploy.py
Purpose: Model Inference and Deployment Utilities
Optimized inference pipeline with model quantization and deployment tools.
Includes:
- OptimizedInference: Handles model loading, preprocessing, and single/batch predictions.
- ModelEnsemble: Combines predictions from multiple models.
- GradioInterface: Creates a web-based UI for interactive predictions.
- ModelProfiler: Tools for benchmarking and profiling model performance.
- ClinicalDeployment: Utilities for clinical reporting with confidence checks.
"""

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic # For dynamic quantization
import numpy as np
from PIL import Image # For image loading
import time
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any, Union # Added Union for type hints
import logging
import onnx # For ONNX model loading and checking
import onnxruntime as ort # For ONNX Runtime inference
from torchvision import transforms, models # For image transformations and pre-trained models
import cv2 # For image processing (e.g., saving PIL images from numpy arrays)
import gradio as gr # For creating the web interface
import matplotlib.pyplot as plt   
import pandas as pd 
import tqdm  
import os 
import datetime
# Configure logging for informative output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================================
# Model Architecture (Copied from model.py for self-containment)
# =====================================================================================

class DualResNet(nn.Module):
    """
    The Dual-Path ResNet-34 model for processing paired B-mode and CEUS images.
    It uses two separate ResNet-34 encoders (one for each modality) and
    concatenates their features before passing them to a shared classifier head.
    This is the model architecture that OptimizedInference will load.
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
# Optimized Inference Class
# =====================================================================================

class OptimizedInference:
    """
    Optimized inference pipeline with multiple backend support (PyTorch, Quantized PyTorch, ONNX).
    Handles model loading, image preprocessing, and prediction.
    """
    
    def __init__(self, model_path: str, config_path: str, device: str = 'auto'):
        """
        Initializes the OptimizedInference handler.

        Args:
            model_path (str): Path to the PyTorch model checkpoint (.pth file).
                              This should be a state_dict.
            config_path (str): Path to the training configuration JSON file.
                               Used to get model parameters like 'pretrained'.
            device (str): Computation device ('auto', 'cuda', or 'cpu').
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
        # Configure logging for this class
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load configuration (e.g., to get 'pretrained' status for DualResNet)
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {config_path}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in config file: {config_path}")
            raise
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
        # Setup preprocessing transforms for inference
        # These should match the EvalTransform used during training
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), # Target image size
            transforms.ToTensor(), # Converts PIL Image to FloatTensor and scales to [0, 1]
            transforms.Normalize([0.5]*3, [0.5]*3) # Normalizes to [-1, 1]
        ])
        
        # Initialize metrics for profiling
        self.inference_times = []
        
    def _load_model(self) -> nn.Module:
        """
        Loads the trained PyTorch model from the checkpoint.
        Assumes the model architecture is DualResNet.
        """
        # Instantiate the model architecture (DualResNet from model_combined.py)
        # We need to know if the model was trained with pretrained weights
        pretrained_status = self.config.get('pretrained', True) 
        model = DualResNet(pretrained=pretrained_status)
        
        # Load the state dictionary
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            # If the checkpoint saved the entire model dict directly, use that
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else: # Assume checkpoint is just the state_dict
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval() # Set model to evaluation mode (disables dropout, batchnorm updates)
            self.logger.info(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            self.logger.error(f"Model checkpoint not found: {self.model_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading model from {self.model_path}: {e}")
            raise
        
        return model
    
    def quantize_model(self, save_path: Optional[str] = None) -> nn.Module:
        """
        Applies dynamic quantization to the model for faster inference on CPU.
        Quantizes Linear and Conv2d layers to int8.

        Args:
            save_path (Optional[str]): Path to save the quantized model's state_dict.

        Returns:
            nn.Module: The quantized model.
        """
        # Ensure model is on CPU for quantization if it's not already,
        # as dynamic quantization is primarily for CPU inference.
        model_cpu = self.model.to('cpu')
        
        # Apply dynamic quantization to specified module types
        quantized_model = quantize_dynamic(
            model_cpu, 
            {nn.Linear, nn.Conv2d}, # Layers to quantize
            dtype=torch.qint8 # Quantize to 8-bit integers
        )
        self.logger.info("Model quantized dynamically.")
        
        if save_path:
            torch.save(quantized_model.state_dict(), save_path)
            self.logger.info(f"Quantized model saved to {save_path}")
        
        # Set the current model to the quantized one if we intend to use it for subsequent inferences
        self.model = quantized_model.to(self.device) # Move back to original device if needed
        return quantized_model
    
    def export_to_onnx(self, save_path: str, opset_version: int = 11):
        """
        Exports the current PyTorch model to ONNX format.
        This allows deployment with ONNX Runtime for optimized inference.

        Args:
            save_path (str): Path to save the ONNX model file.
            opset_version (int): ONNX opset version to use for export.
        """
        # Create dummy inputs for tracing (must match model's expected input shape)
        dummy_b = torch.randn(1, 3, 256, 256).to(self.device)
        dummy_c = torch.randn(1, 3, 256, 256).to(self.device)
        
        try:
            torch.onnx.export(
                self.model,
                (dummy_b, dummy_c), # Model inputs
                save_path,
                export_params=True, # Export trained parameter weights
                opset_version=opset_version,
                do_constant_folding=True, # Optimize constants
                input_names=['b_mode_input', 'ceus_input'], # Names for input nodes
                output_names=['output'], # Name for output node (assuming a single output from classifier)
                # dynamic_axes allows variable batch size
                dynamic_axes={
                    'b_mode_input': {0: 'batch_size'},
                    'ceus_input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Verify the ONNX model's structure and validity
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            
            self.logger.info(f"Model exported to ONNX: {save_path}")
        except Exception as e:
            self.logger.error(f"Error exporting model to ONNX: {e}")
            raise
    
    def preprocess_image(self, image_path: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Preprocesses a single image (from path or PIL Image) for model inference.

        Args:
            image_path (Union[str, Path, Image.Image]): Path to the image file or a PIL Image object.

        Returns:
            torch.Tensor: The preprocessed image tensor, ready for model input (unsqueeze(0) for batch dim).
        """
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, Image.Image):
            image = image_path.convert('RGB')
        else:
            raise TypeError("image_path must be a string, Path, or PIL Image object.")

        return self.transform(image).unsqueeze(0) # Add batch dimension
    
    def predict_single(self, b_mode_path: Union[str, Path, Image.Image], 
                       ceus_path: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Performs inference on a single pair of B-mode and CEUS images.

        Args:
            b_mode_path (Union[str, Path, Image.Image]): Path to B-mode image or PIL Image.
            ceus_path (Union[str, Path, Image.Image]): Path to CEUS image or PIL Image.

        Returns:
            Dict[str, Any]: A dictionary containing prediction, confidence, probabilities,
                            and inference time.
        """
        start_time = time.time()
        
        # Preprocess images
        b_mode_tensor = self.preprocess_image(b_mode_path).to(self.device)
        ceus_tensor = self.preprocess_image(ceus_path).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(b_mode_tensor, ceus_tensor)
            probs = torch.softmax(outputs, dim=1) # Get probabilities
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time) # Log inference time
        
        # Get prediction result
        pred_class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class_idx].item()
        
        # Map class index to label
        class_labels = ['Non-HCC', 'HCC'] # Assuming 0: Non-HCC, 1: HCC
        pred_label = class_labels[pred_class_idx]
        
        return {
            'prediction': pred_label,
            'confidence': confidence,
            'probabilities': {
                class_labels[0]: probs[0, 0].item(),
                class_labels[1]: probs[0, 1].item()
            },
            'inference_time': inference_time
        }
    
    def predict_batch(self, image_pairs: List[Tuple[Union[str, Path, Image.Image], Union[str, Path, Image.Image]]], 
                      batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Performs batch inference on a list of image pairs.

        Args:
            image_pairs (List[Tuple[Union[str, Path, Image.Image], Union[str, Path, Image.Image]]]):
                A list of tuples, where each tuple contains (b_mode_path_or_image, ceus_path_or_image).
            batch_size (int): Number of image pairs to process in a single batch.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing prediction results
                                  for an image pair.
        """
        results = []
        
        for i in tqdm(range(0, len(image_pairs), batch_size), desc="Batch Predicting"):
            batch_pairs = image_pairs[i:i+batch_size]
            
            # Prepare batch tensors
            b_mode_batch_tensors = []
            ceus_batch_tensors = []
            
            for b_path, c_path in batch_pairs:
                b_mode_batch_tensors.append(self.preprocess_image(b_path))
                ceus_batch_tensors.append(self.preprocess_image(c_path))
            
            # Concatenate list of tensors into a single batch tensor
            b_mode_batch = torch.cat(b_mode_batch_tensors).to(self.device)
            ceus_batch = torch.cat(ceus_batch_tensors).to(self.device)
            
            # Perform batch inference
            with torch.no_grad():
                outputs = self.model(b_mode_batch, ceus_batch)
                probs = torch.softmax(outputs, dim=1)
            
            # Process results for each item in the batch
            class_labels = ['Non-HCC', 'HCC']
            for j in range(len(batch_pairs)):
                pred_class_idx = torch.argmax(probs[j]).item()
                results.append({
                    'image_pair_paths': (str(batch_pairs[j][0]), str(batch_pairs[j][1])), # Store paths for reference
                    'prediction': class_labels[pred_class_idx],
                    'confidence': probs[j, pred_class_idx].item(),
                    'probabilities': {
                        class_labels[0]: probs[j, 0].item(),
                        class_labels[1]: probs[j, 1].item()
                    }
                })
        
        return results
    
    def get_inference_stats(self) -> Dict[str, float]:
        """
        Calculates and returns performance statistics for all recorded inferences.

        Returns:
            Dict[str, float]: A dictionary containing mean, std, min, max inference times,
                              total inferences, and inferences per second (FPS).
        """
        if not self.inference_times:
            self.logger.warning("No inference times recorded yet.")
            return {}
        
        times = np.array(self.inference_times)
        return {
            'mean_time_s': float(np.mean(times)),
            'std_time_s': float(np.std(times)),
            'min_time_s': float(np.min(times)),
            'max_time_s': float(np.max(times)),
            'total_inferences': len(times),
            'fps': float(1.0 / np.mean(times)) if np.mean(times) > 0 else 0.0
        }

# =====================================================================================
# Model Ensemble Class
# =====================================================================================

class ModelEnsemble:
    """
    Ensembles predictions from multiple trained models (e.g., from different CV folds)
    to improve robustness and accuracy.
    """
    
    def __init__(self, model_paths: List[str], config_path: str, 
                 weights: Optional[List[float]] = None):
        """
        Initializes the ModelEnsemble.

        Args:
            model_paths (List[str]): List of paths to individual model checkpoints.
            config_path (str): Path to the common training configuration JSON file.
            weights (Optional[List[float]]): Optional list of weights for each model.
                                             If None, models are weighted equally.
        """
        self.models: List[OptimizedInference] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load all individual models using OptimizedInference
        for path in model_paths:
            try:
                inference_instance = OptimizedInference(path, config_path)
                self.models.append(inference_instance)
                self.logger.info(f"Loaded model for ensemble: {path}")
            except Exception as e:
                self.logger.error(f"Failed to load model {path} for ensemble: {e}. Skipping.")
        
        if not self.models:
            raise RuntimeError("No models successfully loaded for ensemble. Cannot proceed.")

        # Set ensemble weights
        if weights:
            if len(weights) != len(self.models):
                raise ValueError("Number of weights must match number of models.")
            self.weights = np.array(weights) / np.sum(weights) # Normalize weights
            self.logger.info(f"Using custom ensemble weights: {self.weights.tolist()}")
        else:
            self.weights = np.array([1.0 / len(self.models)] * len(self.models))
            self.logger.info(f"Using equal ensemble weights: {self.weights.tolist()}")
    
    def predict(self, b_mode_path: Union[str, Path, Image.Image], 
                ceus_path: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Performs an ensemble prediction on a single image pair by averaging
        probabilities from all individual models.

        Args:
            b_mode_path (Union[str, Path, Image.Image]): Path to B-mode image or PIL Image.
            ceus_path (Union[str, Path, Image.Image]): Path to CEUS image or PIL Image.

        Returns:
            Dict[str, Any]: A dictionary containing the ensemble prediction, confidence,
                            and probabilities.
        """
        all_probs = []
        
        # Collect probabilities from each model
        for model_inference in self.models:
            result = model_inference.predict_single(b_mode_path, ceus_path)
            all_probs.append([
                result['probabilities']['Non-HCC'], # Assuming order 0: Non-HCC
                result['probabilities']['HCC']      # Assuming order 1: HCC
            ])
        
        # Convert to numpy array for weighted average
        all_probs_array = np.array(all_probs) # Shape: (num_models, num_classes)
        
        # Apply weighted average across models
        ensemble_probs = np.sum(all_probs_array * self.weights[:, np.newaxis], axis=0)
        
        # Get ensemble prediction
        pred_class_idx = np.argmax(ensemble_probs)
        class_labels = ['Non-HCC', 'HCC']
        
        return {
            'prediction': class_labels[pred_class_idx],
            'confidence': float(ensemble_probs[pred_class_idx]),
            'probabilities': {
                class_labels[0]: float(ensemble_probs[0]),
                class_labels[1]: float(ensemble_probs[1])
            },
            'ensemble_size': len(self.models)
        }

# =====================================================================================
# Gradio Interface Class
# =====================================================================================

class GradioInterface:
    """
    Provides a user-friendly web interface using Gradio for real-time model inference.
    Allows users to upload B-mode and CEUS images and get predictions.
    """
    
    def __init__(self, model_path: str, config_path: str):
        """
        Initializes the Gradio interface with an OptimizedInference instance.

        Args:
            model_path (str): Path to the model checkpoint.
            config_path (str): Path to the training configuration JSON.
        """
        self.inference = OptimizedInference(model_path, config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def predict_interface(self, b_mode_image: np.ndarray, ceus_image: np.ndarray) -> Tuple[str, plt.Figure, Dict]:
        """
        The core prediction function for the Gradio interface.
        Takes numpy arrays (images from Gradio), performs inference, and generates a visualization.

        Args:
            b_mode_image (np.ndarray): B-mode image as a NumPy array (from Gradio input).
            ceus_image (np.ndarray): CEUS image as a NumPy array (from Gradio input).

        Returns:
            Tuple[str, plt.Figure, Dict]: A tuple containing:
                - A summary string of the prediction.
                - A Matplotlib figure for visualization.
                - A dictionary with detailed prediction results.
        """
        if b_mode_image is None or ceus_image is None:
            self.logger.warning("One or both images not uploaded.")
            # Return empty plot and JSON for Gradio
            fig, ax = plt.subplots(1,1)
            ax.text(0.5, 0.5, "Please upload both B-mode and CEUS images", 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.axis('off')
            return "Please upload both B-mode and CEUS images", fig, {}
        
        # Convert numpy arrays to PIL Images for preprocessing
        pil_b_mode = Image.fromarray(b_mode_image)
        pil_ceus = Image.fromarray(ceus_image)
        
        # Perform prediction using the OptimizedInference instance
        try:
            result = self.inference.predict_single(pil_b_mode, pil_ceus)
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            fig, ax = plt.subplots(1,1)
            ax.text(0.5, 0.5, f"Error during prediction: {e}", 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')
            ax.axis('off')
            return "Prediction Error", fig, {"error": str(e)}
            
        # Create visualization using Matplotlib
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display input images
        axes[0].imshow(b_mode_image)
        axes[0].set_title('B-mode Input')
        axes[0].axis('off')
        
        axes[1].imshow(ceus_image)
        axes[1].set_title('CEUS Input')
        axes[1].axis('off')
        
        # Display probability bar chart
        probabilities = result['probabilities']
        classes = list(probabilities.keys())
        values = [probabilities[c] for c in classes]
        
        # Highlight predicted class
        colors = ['gray'] * len(classes)
        if result['prediction'] == 'Non-HCC':
            colors[0] = 'green'
        else: # HCC
            colors[1] = 'red'

        axes[2].bar(classes, values, color=colors)
        axes[2].set_ylim(0, 1)
        axes[2].set_ylabel('Probability')
        axes[2].set_title(f'Prediction: {result["prediction"]} '
                          f'({result["confidence"]:.1%} confidence)')
        
        plt.tight_layout()
        
        # Prepare output string
        output_text = (f"Prediction: {result['prediction']}\n"
                       f"Confidence: {result['confidence']:.1%}\n"
                       f"Inference Time: {result['inference_time']:.3f}s")
        
        return output_text, fig, result
    
    def launch(self):
        """Launches the Gradio web interface."""
        self.logger.info("Launching Gradio interface...")
        interface = gr.Interface(
            fn=self.predict_interface,
            inputs=[
                gr.Image(label="B-mode Ultrasound", type="numpy"), # Gradio provides numpy array
                gr.Image(label="CEUS Ultrasound", type="numpy")
            ],
            outputs=[
                gr.Textbox(label="Prediction Summary"),
                gr.Plot(label="Results Visualization"), # Gradio handles Matplotlib figures
                gr.JSON(label="Detailed Results")
            ],
            title="Liver Cancer Detection System (Dual-Modality Ultrasound)",
            description="Upload paired B-mode and CEUS ultrasound images to detect HCC.",
            # Example images (ensure these paths exist relative to where Gradio is launched)
            examples=[
                ["example_b_mode_1.png", "example_ceus_1.png"], # Placeholder examples
                # You should replace these with actual paths to sample images in your project
            ],
            theme="default", # Or "huggingface", "soft", etc.
            allow_flagging="never" # Disable flagging for production use
        )
        
        interface.launch(share=True) # share=True generates a public link for easy sharing

# =====================================================================================
# Model Profiler Class
# =====================================================================================

class ModelProfiler:
    """
    Provides utilities for profiling and benchmarking model performance
    across different backends (PyTorch, Quantized PyTorch, ONNX Runtime).
    """
    
    def __init__(self, model_path: str, config_path: str):
        """
        Initializes the ModelProfiler with an OptimizedInference instance.

        Args:
            model_path (str): Path to the model checkpoint.
            config_path (str): Path to the training configuration JSON.
        """
        self.inference = OptimizedInference(model_path, config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def profile_inference(self, num_runs: int = 100, trace_output_path: str = "inference_trace.json"):
        """
        Profiles PyTorch model inference using torch.profiler.

        Args:
            num_runs (int): Number of inference runs to profile.
            trace_output_path (str): Path to save the Chrome trace file.
        """
        import torch.profiler as profiler # Import here to keep it local to this function
        
        self.logger.info(f"Starting PyTorch inference profiling for {num_runs} runs...")
        
        # Create dummy inputs (batch size 1 for single inference profiling)
        dummy_b = torch.randn(1, 3, 256, 256).to(self.inference.device)
        dummy_c = torch.randn(1, 3, 256, 256).to(self.inference.device)
        
        # Warmup runs to stabilize performance
        self.logger.info("Warmup runs (10 iterations)...")
        for _ in range(10):
            with torch.no_grad():
                _ = self.inference.model(dummy_b, dummy_c)
        
        # Start profiling
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA, # Only if CUDA is available
            ],
            record_shapes=True, # Record tensor shapes
            profile_memory=True, # Profile memory usage
            with_stack=True # Record call stack for detailed analysis
        ) as prof:
            with profiler.record_function("model_inference_loop"):
                for _ in tqdm(range(num_runs), desc="Profiling Inference"):
                    with torch.no_grad():
                        _ = self.inference.model(dummy_b, dummy_c)
        
        # Generate and print a summary table
        self.logger.info("\n--- Profiling Results (Top 10 CUDA Time) ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # Save detailed trace for visualization in Chrome's `chrome://tracing`
        prof.export_chrome_trace(trace_output_path)
        self.logger.info(f"Detailed trace saved to: {trace_output_path}")
        
        return prof # Return profiler object for further analysis if needed
    
    def benchmark_backends(self, num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmarks inference performance across different backends:
        PyTorch (native), Quantized PyTorch, and ONNX Runtime.

        Args:
            num_runs (int): Number of inference runs for each backend.

        Returns:
            Dict[str, float]: Dictionary with mean inference time (in ms) for each backend.
        """
        results = {}
        
        # Create dummy image paths for `predict_single`
        # These dummy files need to exist for `Image.open`
        dummy_b_path = "temp_dummy_b.png"
        dummy_c_path = "temp_dummy_c.png"
        
        # Create dummy PNGs
        Image.new('RGB', (256, 256), color = 'red').save(dummy_b_path)
        Image.new('RGB', (256, 256), color = 'blue').save(dummy_c_path)
        self.logger.info("Created dummy images for benchmarking.")

        try:
            # --- Benchmark PyTorch Native ---
            self.logger.info(f"Benchmarking PyTorch native ({num_runs} runs)...")
            start = time.time()
            for _ in tqdm(range(num_runs), desc="PyTorch Native"):
                self.inference.predict_single(dummy_b_path, dummy_c_path)
            results['pytorch_native_ms'] = (time.time() - start) / num_runs * 1000
            self.logger.info(f"PyTorch Native: {results['pytorch_native_ms']:.2f} ms/inference")
            
            # --- Benchmark Quantized Model ---
            self.logger.info(f"Benchmarking PyTorch quantized ({num_runs} runs)...")
            quantized_model = self.inference.quantize_model() # Quantize the model
            # Temporarily set the inference model to the quantized one
            original_model = self.inference.model
            self.inference.model = quantized_model
            
            start = time.time()
            for _ in tqdm(range(num_runs), desc="PyTorch Quantized"):
                self.inference.predict_single(dummy_b_path, dummy_c_path)
            results['pytorch_quantized_ms'] = (time.time() - start) / num_runs * 1000
            self.logger.info(f"PyTorch Quantized: {results['pytorch_quantized_ms']:.2f} ms/inference")
            
            # Restore original model
            self.inference.model = original_model

            # --- Benchmark ONNX Runtime ---
            self.logger.info(f"Benchmarking ONNX Runtime ({num_runs} runs)...")
            onnx_model_path = "temp_model.onnx"
            self.inference.export_to_onnx(onnx_model_path) # Export to ONNX
            
            # Initialize ONNX Runtime session
            ort_session = ort.InferenceSession(onnx_model_path)
            
            # Prepare dummy inputs for ONNX Runtime (must be numpy arrays)
            # These should be preprocessed just like the model expects
            dummy_b_np = self.inference.preprocess_image(dummy_b_path).cpu().numpy()
            dummy_c_np = self.inference.preprocess_image(dummy_c_path).cpu().numpy()
            
            start = time.time()
            for _ in tqdm(range(num_runs), desc="ONNX Runtime"):
                # ONNX Runtime expects inputs as a dictionary mapping input names to numpy arrays
                _ = ort_session.run(None, {
                    'b_mode_input': dummy_b_np,
                    'ceus_input': dummy_c_np
                })
            results['onnx_runtime_ms'] = (time.time() - start) / num_runs * 1000
            self.logger.info(f"ONNX Runtime: {results['onnx_runtime_ms']:.2f} ms/inference")
            
            os.remove(onnx_model_path) # Clean up ONNX model file

        except Exception as e:
            self.logger.error(f"Error during backend benchmarking: {e}")
        finally:
            # Clean up dummy image files
            if os.path.exists(dummy_b_path): os.remove(dummy_b_path)
            if os.path.exists(dummy_c_path): os.remove(dummy_c_path)
            self.logger.info("Cleaned up dummy images.")
            
        return results

    def create_mobile_model(self, model_path: str, config_path: str, output_path: str):
        """
        Creates a mobile-optimized model using TorchScript and saves it.
        This function is intended to be called separately, not as part of a benchmark.

        Args:
            model_path (str): Path to the original PyTorch model checkpoint.
            config_path (str): Path to the training configuration JSON.
            output_path (str): Path to save the optimized mobile model (.ptl file).
        """
        inference = OptimizedInference(model_path, config_path)
        self.logger.info(f"Creating mobile-optimized model from {model_path}...")
        
        # Convert to TorchScript (tracing)
        dummy_b = torch.randn(1, 3, 256, 256).to(inference.device)
        dummy_c = torch.randn(1, 3, 256, 256).to(inference.device)
        
        # Trace the model with dummy inputs
        traced_model = torch.jit.trace(inference.model, (dummy_b, dummy_c))
        self.logger.info("Model traced to TorchScript.")
        
        # Optimize for mobile deployment
        # This function is part of torch.utils.mobile_optimizer
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile
            optimized_model = optimize_for_mobile(traced_model)
            self.logger.info("Model optimized for mobile.")
        except ImportError:
            self.logger.error("torch.utils.mobile_optimizer not found. Please ensure PyTorch Mobile is installed.")
            optimized_model = traced_model # Fallback to just traced model
        
        # Save the optimized model for Lite Interpreter
        optimized_model._save_for_lite_interpreter(output_path)
        self.logger.info(f"Mobile-optimized model saved to: {output_path}")
        
        # Get and print model size
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        self.logger.info(f"Mobile model size: {size_mb:.2f} MB")

# =====================================================================================
# Clinical Deployment Utilities
# =====================================================================================

class ClinicalDeployment:
    """
    Provides utilities for clinical deployment, including confidence checks
    and automated report generation.
    """
    
    def __init__(self, model_path: str, config_path: str):
        """
        Initializes the ClinicalDeployment handler.

        Args:
            model_path (str): Path to the model checkpoint.
            config_path (str): Path to the training configuration JSON.
        """
        self.inference = OptimizedInference(model_path, config_path)
        self.prediction_history: List[Dict[str, Any]] = [] # Stores a log of all predictions
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def predict_with_confidence_check(self, b_mode_path: Union[str, Path, Image.Image], 
                                      ceus_path: Union[str, Path, Image.Image],
                                      confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Performs prediction and adds a clinical decision based on a confidence threshold.
        If confidence is below the threshold, it flags the prediction for expert review.

        Args:
            b_mode_path (Union[str, Path, Image.Image]): Path to B-mode image or PIL Image.
            ceus_path (Union[str, Path, Image.Image]): Path to CEUS image or PIL Image.
            confidence_threshold (float): The minimum confidence required for a definitive AI decision.

        Returns:
            Dict[str, Any]: The prediction result dictionary with added 'clinical_decision' and 'alert' fields.
        """
        result = self.inference.predict_single(b_mode_path, ceus_path)
        
        # Add confidence check for clinical decision making
        if result['confidence'] < confidence_threshold:
            result['clinical_decision'] = 'UNCERTAIN - Requires Expert Review'
            result['alert'] = True
            self.logger.warning(f"Low confidence prediction ({result['confidence']:.1%}). "
                                f"Flagged for expert review for {b_mode_path}, {ceus_path}.")
        else:
            result['clinical_decision'] = result['prediction']
            result['alert'] = False
            self.logger.info(f"High confidence prediction ({result['confidence']:.1%}): {result['prediction']}.")
        
        # Log the prediction for audit/history
        self.prediction_history.append({
            'timestamp': time.time(),
            'b_mode_image_path': str(b_mode_path), # Store as string for logging
            'ceus_image_path': str(ceus_path),
            'prediction_result': result # Store the full result dict
        })
        
        return result
    
    def generate_clinical_report(self, patient_id: str, result: Dict[str, Any]) -> str:
        """
        Generates a formatted clinical report string based on the prediction result.

        Args:
            patient_id (str): Identifier for the patient.
            result (Dict[str, Any]): The prediction result dictionary (from predict_with_confidence_check).

        Returns:
            str: A multi-line string representing the clinical report.
        """
        report = f"""
LIVER ULTRASOUND ANALYSIS REPORT
================================
Patient ID: {patient_id}
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

ANALYSIS RESULTS:
-----------------
AI Prediction: {result['prediction']}
Confidence Level: {result['confidence']:.1%}

Probability Scores:
- Non-HCC: {result['probabilities'].get('Non-HCC', 'N/A'):.1%}
- HCC: {result['probabilities'].get('HCC', 'N/A'):.1%}

CLINICAL RECOMMENDATION:
------------------------
{result.get('clinical_decision', 'N/A')}

{"⚠️ ALERT: Low confidence prediction. Expert review strongly recommended." if result.get('alert', False) else ""}

DISCLAIMER:
-----------
This AI-assisted analysis is intended to support clinical decision-making
and should not be used as the sole basis for diagnosis. Always correlate
with clinical findings and other diagnostic tests.

Analysis Time: {result.get('inference_time', 'N/A'):.3f} seconds
Model Version: Enhanced Dual-Path CNN v2.0 (Placeholder)
"""
        self.logger.info(f"Generated clinical report for Patient ID: {patient_id}")
        return report
    
    def export_predictions_log(self, output_path: str):
        """
        Exports the history of all predictions made by this ClinicalDeployment instance to a CSV file.

        Args:
            output_path (str): Path to save the CSV log file.
        """
        df_data = []
        for pred_entry in self.prediction_history:
            # Flatten the nested dictionary for CSV export
            df_data.append({
                'timestamp': datetime.fromtimestamp(pred_entry['timestamp']).isoformat(),
                'b_mode_image': pred_entry['b_mode_image_path'],
                'ceus_image': pred_entry['ceus_image_path'],
                'prediction': pred_entry['prediction_result']['prediction'],
                'confidence': pred_entry['prediction_result']['confidence'],
                'non_hcc_probability': pred_entry['prediction_result']['probabilities'].get('Non-HCC', np.nan),
                'hcc_probability': pred_entry['prediction_result']['probabilities'].get('HCC', np.nan),
                'clinical_decision': pred_entry['prediction_result'].get('clinical_decision', ''),
                'alert_flag': pred_entry['prediction_result'].get('alert', False)
            })
        
        df = pd.DataFrame(df_data)
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        self.logger.info(f"Prediction log exported to: {output_path}")

# =====================================================================================
# Main Execution Block
# =====================================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Liver Cancer Detection Inference and Deployment Utilities')
    parser.add_argument('--mode', choices=['predict_single', 'predict_batch', 'benchmark', 'gradio', 'clinical', 'export_onnx', 'create_mobile'],
                        default='predict_single', help='Operation mode: '
                        'predict_single (for one image pair), '
                        'predict_batch (for multiple image pairs), '
                        'benchmark (profile performance), '
                        'gradio (launch web UI), '
                        'clinical (clinical report generation), '
                        'export_onnx (export model to ONNX), '
                        'create_mobile (export model to TorchScript for mobile).')
    
    # Common arguments
    parser.add_argument('--model_path', required=True, help='Path to the PyTorch model checkpoint (.pth file).')
    parser.add_argument('--config_path', required=True, help='Path to the training configuration JSON file.')
    
    # Arguments for 'predict_single' and 'clinical' modes
    parser.add_argument('--b_mode_image', help='Path to the B-mode image file.')
    parser.add_argument('--ceus_image', help='Path to the CEUS image file.')
    parser.add_argument('--patient_id', default='UNKNOWN_PATIENT', help='Patient ID for clinical reports.')

    # Arguments for 'predict_batch' mode
    parser.add_argument('--image_pairs_json', help='Path to a JSON file containing a list of [b_mode_path, ceus_path] pairs for batch prediction.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for batch prediction.')

    # Arguments for 'benchmark' mode
    parser.add_argument('--num_runs', type=int, default=100, help='Number of runs for benchmarking/profiling.')
    parser.add_argument('--trace_output', default="inference_trace.json", help='Output path for profiler trace.')

    # Arguments for 'export_onnx' and 'create_mobile' modes
    parser.add_argument('--output_path', help='Output path for exported models (ONNX, mobile) or clinical log.')
    
    args = parser.parse_args()
    
    # --- Execute based on mode ---

    if args.mode == 'predict_single':
        if not args.b_mode_image or not args.ceus_image:
            parser.error("--b_mode_image and --ceus_image are required for 'predict_single' mode.")
        inference_handler = OptimizedInference(args.model_path, args.config_path)
        result = inference_handler.predict_single(args.b_mode_image, args.ceus_image)
        print(json.dumps(result, indent=2))
        print("\nInference Stats (from single run):")
        print(json.dumps(inference_handler.get_inference_stats(), indent=2))

    elif args.mode == 'predict_batch':
        if not args.image_pairs_json:
            parser.error("--image_pairs_json is required for 'predict_batch' mode.")
        try:
            with open(args.image_pairs_json, 'r') as f:
                image_pairs_list = json.load(f)
        except FileNotFoundError:
            parser.error(f"Image pairs JSON file not found: {args.image_pairs_json}")
        except json.JSONDecodeError:
            parser.error(f"Invalid JSON in image pairs file: {args.image_pairs_json}")

        inference_handler = OptimizedInference(args.model_path, args.config_path)
        results = inference_handler.predict_batch(image_pairs_list, args.batch_size)
        print(json.dumps(results, indent=2))
        print("\nBatch Inference Stats:")
        print(json.dumps(inference_handler.get_inference_stats(), indent=2))

    elif args.mode == 'benchmark':
        profiler = ModelProfiler(args.model_path, args.config_path)
        
        # Run detailed profiler first
        prof_results = profiler.profile_inference(num_runs=args.num_runs, trace_output_path=args.trace_output)
        
        # Then run backend benchmarks
        backend_benchmarks = profiler.benchmark_backends(num_runs=args.num_runs)
        print("\n--- Backend Benchmark Results ---")
        for backend, time_ms in backend_benchmarks.items():
            print(f"{backend}: {time_ms:.2f} ms/inference")
            
    elif args.mode == 'gradio':
        # Ensure example images exist for Gradio interface
        # You might want to create these dummy files or replace with actual paths
        # For demonstration, let's create simple dummy images if they don't exist
        if not Path("example_b_mode_1.png").exists():
            Image.new('RGB', (256, 256), color = 'red').save("example_b_mode_1.png")
            Image.new('RGB', (256, 256), color = 'blue').save("example_ceus_1.png")
        if not Path("example_b_mode_2.png").exists():
            Image.new('RGB', (256, 256), color = 'green').save("example_b_mode_2.png")
            Image.new('RGB', (256, 256), color = 'yellow').save("example_ceus_2.png")

        gradio_interface = GradioInterface(args.model_path, args.config_path)
        gradio_interface.launch()
        
    elif args.mode == 'clinical':
        if not args.b_mode_image or not args.ceus_image:
            parser.error("--b_mode_image and --ceus_image are required for 'clinical' mode.")
        
        clinical_deployment = ClinicalDeployment(args.model_path, args.config_path)
        result = clinical_deployment.predict_with_confidence_check(args.b_mode_image, args.ceus_image)
        report = clinical_deployment.generate_clinical_report(args.patient_id, result)
        print(report)
        
        if args.output_path:
            # Save the report to a text file
            report_file_path = Path(args.output_path)
            report_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file_path, 'w') as f:
                f.write(report)
            print(f"Clinical report saved to: {report_file_path}")
            
            # Also export the prediction history to CSV
            log_csv_path = report_file_path.parent / f"clinical_predictions_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            clinical_deployment.export_predictions_log(str(log_csv_path))

    elif args.mode == 'export_onnx':
        if not args.output_path:
            parser.error("--output_path is required for 'export_onnx' mode.")
        inference_handler = OptimizedInference(args.model_path, args.config_path)
        inference_handler.export_to_onnx(args.output_path)
        
    elif args.mode == 'create_mobile':
        if not args.output_path:
            parser.error("--output_path is required for 'create_mobile' mode.")
        profiler_for_mobile = ModelProfiler(args.model_path, args.config_path) # Use profiler's method
        profiler_for_mobile.create_mobile_model(args.model_path, args.config_path, args.output_path)
