"""
utils.py
Shared helper utilities for the Speech Emotion Recognition project.

Includes:
  - set_seeds()          : reproducibility
  - save_results_csv()   : persist experiment results
  - load_or_compute()    : cache features to disk (avoids re-extraction)
  - print_gpu_info()     : confirm GPU is available in Colab
  - plot_class_distribution() : EDA bar chart
  - plot_sample_spectrograms(): EDA spectrogram grid
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_seeds(seed=42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def print_gpu_info():
    """Print whether a GPU is available and its name."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"GPU available: {gpus[0].name}")
        else:
            print("No GPU found. Training will run on CPU (slower).")
            print("In Colab: Runtime -> Change runtime type -> GPU")
    except Exception as e:
        print(f"Could not check GPU: {e}")


def save_results_csv(results_list, filepath="results/experiment_results.csv"):
    """
    Append experiment results to a CSV file.

    Parameters
    ----------
    results_list : list of dict
        Each dict has keys like model, accuracy, macro_f1, epochs, etc.
    filepath : str
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    return df


def load_or_compute(cache_path, compute_fn, *args, **kwargs):
    """
    If cache_path exists, load and return it.
    Otherwise, call compute_fn(*args, **kwargs), save to cache_path, and return.

    This avoids re-extracting features every run.

    Parameters
    ----------
    cache_path : str         path to .npz file
    compute_fn : callable    function that returns (X, y) arrays
    *args, **kwargs          passed to compute_fn

    Returns
    -------
    X, y : np.ndarray
    """
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        data = np.load(cache_path)
        return data["X"], data["y"]

    print(f"Cache not found. Computing features...")
    X, y = compute_fn(*args, **kwargs)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"Features cached to {cache_path}")
    return X, y


def plot_class_distribution(df, save_dir="results/plots"):
    """
    Bar chart showing the number of samples per emotion class.

    Parameters
    ----------
    df       : pd.DataFrame  with 'emotion' column
    save_dir : str
    """
    os.makedirs(save_dir, exist_ok=True)

    counts = df["emotion"].value_counts().sort_index()
    labels = [e.replace("ps", "P.Surprise").title() for e in counts.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, counts.values,
                  color=plt.cm.Set2(np.linspace(0, 1, len(counts))),
                  edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=10)

    ax.set_title("Class Distribution - TESS Dataset", fontsize=13, fontweight="bold")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Number of Samples")
    ax.set_ylim([0, max(counts.values) * 1.15])
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Class distribution plot saved to {path}")


def plot_sample_spectrograms(df, n_per_class=2, save_dir="results/plots"):
    """
    Display sample log-mel spectrograms for each emotion class.

    Parameters
    ----------
    df          : pd.DataFrame  with 'path' and 'emotion' columns
    n_per_class : int
    save_dir    : str
    """
    from feature_extraction import load_audio, extract_log_mel_spectrogram

    os.makedirs(save_dir, exist_ok=True)
    emotions = sorted(df["emotion"].unique())
    n_rows = len(emotions)
    n_cols = n_per_class

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))
    fig.suptitle("Sample Log-Mel Spectrograms per Emotion",
                 fontsize=13, fontweight="bold")

    for r, emo in enumerate(emotions):
        samples = df[df["emotion"] == emo].sample(
            n=n_per_class, random_state=42
        )
        for c, (_, row) in enumerate(samples.iterrows()):
            ax = axes[r, c] if n_cols > 1 else axes[r]
            y, sr = load_audio(row["path"])
            spec = extract_log_mel_spectrogram(y, sr)
            ax.imshow(spec[:, :, 0], aspect="auto",
                      origin="lower", cmap="magma")
            if c == 0:
                ax.set_ylabel(emo.title(), fontsize=9, fontweight="bold")
            ax.axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, "sample_spectrograms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Sample spectrograms saved to {path}")


def format_time(seconds):
    """Format seconds into a readable hh:mm:ss string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"