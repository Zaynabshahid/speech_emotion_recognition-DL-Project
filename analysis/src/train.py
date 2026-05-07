"""
train.py
Training loop with callbacks for the Speech Emotion Recognition project.

Provides:
  - train_model()        : main training function with callbacks
  - get_callbacks()      : builds standard callback list
  - plot_history()       : plots training vs validation curves
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def get_callbacks(model_name,
                  checkpoint_dir="results/checkpoints",
                  patience_es=15,
                  patience_lr=7,
                  min_lr=1e-6):
    """
    Build the standard callback list used across all experiments.

    Callbacks:
      - ModelCheckpoint : saves the best model by val_accuracy
      - EarlyStopping   : stops when val_loss does not improve
      - ReduceLROnPlateau : halves LR when val_loss plateaus
      - CSVLogger       : logs all metrics to a CSV file

    Parameters
    ----------
    model_name     : str   used for file names
    checkpoint_dir : str   directory to save model weights
    patience_es    : int   early stopping patience (epochs)
    patience_lr    : int   LR reduction patience (epochs)
    min_lr         : float minimum allowed learning rate

    Returns
    -------
    list of keras callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir, f"{model_name}_best.keras"
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            filename=f"results/logs/{model_name}_history.csv",
            append=False,
        ),
    ]
    return callbacks


def train_model(model,
                X_train, y_train,
                X_val,   y_val,
                model_name="model",
                epochs=80,
                batch_size=32,
                class_weights=None):
    """
    Train a Keras model with standard callbacks.

    Parameters
    ----------
    model       : keras.Model  compiled model
    X_train     : np.ndarray   training features
    y_train     : np.ndarray   training labels
    X_val       : np.ndarray   validation features
    y_val       : np.ndarray   validation labels
    model_name  : str          used for checkpoint/log filenames
    epochs      : int          maximum training epochs
    batch_size  : int
    class_weights : dict or None   {class_index: weight}

    Returns
    -------
    history : keras History object
    """
    callbacks = get_callbacks(model_name)

    print(f"\nTraining {model_name}...")
    print(f"  Train size : {len(X_train)}")
    print(f"  Val size   : {len(X_val)}")
    print(f"  Epochs     : {epochs} (with early stopping)")
    print(f"  Batch size : {batch_size}")
    if class_weights:
        print(f"  Class weights: {class_weights}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    return history


def plot_history(history, model_name, save_dir="results/plots"):
    """
    Plot training vs validation accuracy and loss curves side by side.

    Parameters
    ----------
    history    : keras History object (or a dict with keys acc/loss/val_*)
    model_name : str   used in title and filename
    save_dir   : str   directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)

    if hasattr(history, "history"):
        h = history.history
    else:
        h = history

    epochs = range(1, len(h["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{model_name} - Training History", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(epochs, h["accuracy"],     label="Train Accuracy", linewidth=2)
    axes[0].plot(epochs, h["val_accuracy"], label="Val Accuracy",   linewidth=2,
                 linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Loss
    axes[1].plot(epochs, h["loss"],     label="Train Loss", linewidth=2)
    axes[1].plot(epochs, h["val_loss"], label="Val Loss",   linewidth=2,
                 linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {save_path}")


def compare_histories(histories_dict, metric="val_accuracy", save_dir="results/plots"):
    """
    Overlay multiple model training curves on a single plot.
    Useful for the ablation and comparison sections.

    Parameters
    ----------
    histories_dict : dict  {model_name: history_object_or_dict}
    metric         : str   metric to compare
    save_dir       : str
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for name, hist in histories_dict.items():
        h = hist.history if hasattr(hist, "history") else hist
        if metric in h:
            plt.plot(h[metric], label=name, linewidth=2)

    plt.title(f"Model Comparison - {metric}", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"comparison_{metric}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Comparison plot saved to {save_path}")