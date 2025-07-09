import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import json # Import json for loading the raw_data file

def plot_shape_counts(df_clean: pd.DataFrame, output_dir: str, top_n: int = 10):
    """
    Generates and saves a bar chart of the top N image array shapes.

    Args:
        df_clean (pd.DataFrame): The cleaned DataFrame containing image data.
        output_dir (str): Directory to save the plot.
        top_n (int): Number of top shapes to display.
    """
    # Count each unique shape string
    shape_counts = df_clean['shape'].value_counts()

    # If too many distinct shapes, show top N
    print(f"Top {top_n} shapes:")
    print(shape_counts.head(top_n))

    # Bar chart of top shapes
    plt.figure(figsize=(8, 4))
    shape_counts.head(top_n).plot.bar()
    plt.title(f"Top {top_n} image array shapes")
    plt.xlabel("Shape (F, H, W, C)")
    plt.ylabel("Count")
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, "top_shapes_bar_chart.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close() # Close the plot to free memory

def plot_multiframe_pie_chart(df_clean: pd.DataFrame, output_dir: str):
    """
    Generates and saves a pie chart showing multi-frame vs. single-frame DICOMs.

    Args:
        df_clean (pd.DataFrame): The cleaned DataFrame containing image data.
        output_dir (str): Directory to save the plot.
    """
    # Count multi-frame vs single-frame
    counts_4d = df_clean['is_4d'].value_counts().sort_index()

    # Map integer codes to human-readable labels
    labels_map = {
        0: '3-D single frame (0)',
        1: '4-D multiframe (1)',
        2: 'other (2)'
    }
    labels_for_chart = [labels_map.get(int(idx), f"other ({int(idx)})")
                        for idx in counts_4d.index]

    plt.figure(figsize=(5, 5))
    counts_4d.plot.pie(labels=labels_for_chart, autopct='%1.1f%%')
    plt.ylabel('') # Hide the default 'is_4d' label
    plt.title("Single vs Multiframe DICOMs")
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, "single_vs_multiframe_pie_chart.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close() # Close the plot to free memory

def plot_epochs_before_stop(raw_data_path: str, output_dir: str):
    """
    Generates and saves a bar chart of epochs before stopping for each fold.

    Args:
        raw_data_path (str): Path to the raw_data.json file (or similar structure).
                             This function assumes raw_data contains 'tr_loss' for each fold.
        output_dir (str): Directory to save the plot.
    """
    try:
        # Load raw_data from the provided JSON file
        with open(raw_data_path, 'r') as f:
            raw_data = json.load(f)
        print(f"Loaded raw_data from {raw_data_path}")

        # Compute epochs before stopping (same for train & val)
        folds = sorted(raw_data.keys())
        epochs_before_stop = [len(raw_data[f]['tr_loss']) for f in folds]

        # Single bar chart
        plt.figure(figsize=(6, 4))
        plt.bar(folds, epochs_before_stop)
        plt.title("Epochs before stopping (per fold)")
        plt.xlabel("Fold")
        plt.ylabel("Epochs")
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(output_dir, "epochs_before_stopping_bar_chart.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close() # Close the plot to free memory

    except FileNotFoundError:
        print(f"Error: raw_data.json file not found at {raw_data_path}. Skipping 'epochs_before_stop' plot.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {raw_data_path}. Skipping 'epochs_before_stop' plot.")
    except KeyError as e:
        print(f"Error: Missing key '{e}' in raw_data.json. Ensure 'tr_loss' is present for each fold. Skipping 'epochs_before_stop' plot.")
    except Exception as e:
        print(f"An unexpected error occurred while generating 'epochs_before_stop' plot: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate various plots from a DICOM manifest file.")
    parser.add_argument('--manifest_path', type=str, required=True,
                        help='Path to the DICOM manifest Excel file (e.g., ../exploration/dicom_manifest.xlsx).')
    parser.add_argument('--output_dir', type=str, default='graphs',
                        help='Directory to save the generated graphs. Defaults to "graphs" in the current working directory.')
    parser.add_argument('--top_n_shapes', type=int, default=10,
                        help='Number of top image shapes to display in the bar chart.')
    parser.add_argument('--raw_data_path', type=str, default='raw_data.json', # Default for convenience, but will attempt to load
                        help='Path to the raw_data.json file for the epochs before stopping plot.')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving graphs to: {os.path.abspath(args.output_dir)}")

    # Load and clean data
    try:
        df = pd.read_excel(args.manifest_path)
        df_clean = df[df['ok/error'] == 1].copy()
        print(f"Total rows: {len(df):,}, after drop: {len(df_clean):,}")
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {args.manifest_path}")
        return
    except Exception as e:
        print(f"Error loading or cleaning manifest file: {e}")
        return

    # Generate plots
    plot_shape_counts(df_clean, args.output_dir, args.top_n_shapes)
    plot_multiframe_pie_chart(df_clean, args.output_dir)
    plot_epochs_before_stop(args.raw_data_path, args.output_dir)

if __name__ == "__main__":
    main()
