"""
evaluate.py
Evaluation metrics, confusion matrix, per-class report,
and error analysis for the Speech Emotion Recognition project.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
)


EMOTION_DISPLAY = [
    "Angry", "Disgust", "Fear", "Happy", "Neutral", "P.Surprise", "Sad"
]


def evaluate_model(model, X_test, y_test, model_name="model", save_dir="results"):
    """
    Full evaluation: accuracy, macro F1, per-class report, confusion matrix.

    Parameters
    ----------
    model      : keras.Model  trained model
    X_test     : np.ndarray
    y_test     : np.ndarray   integer labels
    model_name : str
    save_dir   : str

    Returns
    -------
    dict  with keys: accuracy, macro_f1, per_class_report, y_pred
    """
    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report   = classification_report(y_test, y_pred,
                                      target_names=EMOTION_DISPLAY,
                                      digits=4)

    print(f"\n{'='*55}")
    print(f"  {model_name} - Test Results")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Macro F1  : {macro_f1:.4f}")
    print(f"\n{report}")

    plot_confusion_matrix(y_test, y_pred, model_name,
                          save_dir=os.path.join(save_dir, "plots"))

    return {
        "accuracy":          acc,
        "macro_f1":          macro_f1,
        "per_class_report":  report,
        "y_pred":            y_pred,
        "y_prob":            y_prob,
    }


def plot_confusion_matrix(y_true, y_pred, model_name,
                           labels=None, save_dir="results/plots"):
    """
    Plot a normalized confusion matrix as a heatmap.

    Parameters
    ----------
    y_true     : np.ndarray  true integer labels
    y_pred     : np.ndarray  predicted integer labels
    model_name : str
    labels     : list of str  class display names
    save_dir   : str
    """
    os.makedirs(save_dir, exist_ok=True)
    if labels is None:
        labels = EMOTION_DISPLAY

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{model_name} - Confusion Matrix", fontsize=13, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Normalized (row %)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor="white", ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Confusion matrix saved to {path}")


def plot_per_class_f1(results_dict, save_dir="results/plots"):
    """
    Bar chart comparing per-class F1 scores across multiple models.

    Parameters
    ----------
    results_dict : dict  {model_name: evaluate_model() output dict}
    save_dir     : str
    """
    os.makedirs(save_dir, exist_ok=True)
    from sklearn.metrics import f1_score as sk_f1

    emotion_f1s = {}
    model_names = list(results_dict.keys())

    for name, res in results_dict.items():
        f1s = sk_f1(res["y_true"], res["y_pred"], average=None)
        emotion_f1s[name] = f1s

    n_classes = len(EMOTION_DISPLAY)
    x = np.arange(n_classes)
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, name in enumerate(model_names):
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, emotion_f1s[name], width, label=name, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_DISPLAY)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, "per_class_f1_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Per-class F1 chart saved to {path}")


def plot_model_comparison_bar(summary_rows, save_dir="results/plots"):
    """
    Horizontal bar chart comparing accuracy and macro F1 across models.

    Parameters
    ----------
    summary_rows : list of dict
        Each dict: {"model": str, "accuracy": float, "macro_f1": float}
    save_dir : str
    """
    os.makedirs(save_dir, exist_ok=True)

    names = [r["model"] for r in summary_rows]
    accs  = [r["accuracy"] * 100 for r in summary_rows]
    f1s   = [r["macro_f1"] * 100 for r in summary_rows]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.barh(y - 0.2, accs, height=0.35, label="Accuracy (%)", color="#4C72B0")
    bars2 = ax.barh(y + 0.2, f1s,  height=0.35, label="Macro F1 (%)", color="#DD8452")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Score (%)")
    ax.set_title("Model Comparison - Accuracy and Macro F1", fontweight="bold")
    ax.set_xlim([0, 105])
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    for bar in bars1:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%", va="center", fontsize=9)
    for bar in bars2:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Model comparison chart saved to {path}")


def find_failure_cases(X_test, y_test, y_pred, y_prob,
                        n=6, save_dir="results/plots"):
    """
    Find and display the most confidently wrong predictions (failure cases).

    Parameters
    ----------
    X_test   : np.ndarray  test spectrograms
    y_test   : np.ndarray  true labels
    y_pred   : np.ndarray  predicted labels
    y_prob   : np.ndarray  softmax probabilities
    n        : int         number of failure cases to display
    save_dir : str
    """
    os.makedirs(save_dir, exist_ok=True)

    wrong_mask = y_test != y_pred
    wrong_idx  = np.where(wrong_mask)[0]

    # Sort by confidence of the wrong prediction (most confident first)
    wrong_confs = y_prob[wrong_idx, y_pred[wrong_idx]]
    sorted_order = np.argsort(wrong_confs)[::-1]
    top_fails = wrong_idx[sorted_order[:n]]

    fig, axes = plt.subplots(2, n // 2, figsize=(15, 7))
    fig.suptitle("Most Confident Wrong Predictions (Failure Cases)",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for i, idx in enumerate(top_fails):
        spec = X_test[idx, :, :, 0]
        true_lbl = EMOTION_DISPLAY[y_test[idx]]
        pred_lbl = EMOTION_DISPLAY[y_pred[idx]]
        conf     = wrong_confs[sorted_order[i]] * 100

        axes[i].imshow(spec, aspect="auto", origin="lower", cmap="magma")
        axes[i].set_title(
            f"True: {true_lbl}\nPred: {pred_lbl} ({conf:.1f}%)",
            fontsize=9, color="red" if true_lbl != pred_lbl else "green"
        )
        axes[i].axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, "failure_cases.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Failure cases saved to {path}")