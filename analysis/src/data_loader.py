"""
data_loader.py
Handles all dataset loading, path resolution, label extraction,
and train/val/test splitting for the TESS dataset.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


EMOTION_MAP = {
    "angry":   0,
    "disgust": 1,
    "fear":    2,
    "happy":   3,
    "neutral": 4,
    "ps":      5,
    "sad":     6,
}

EMOTION_LABELS = {v: k for k, v in EMOTION_MAP.items()}

EMOTION_DISPLAY = {
    "angry":   "Angry",
    "disgust": "Disgust",
    "fear":    "Fear",
    "happy":   "Happy",
    "neutral": "Neutral",
    "ps":      "Pleasant Surprise",
    "sad":     "Sad",
}


def load_tess_paths(data_root):
    """
    Walks the TESS directory and returns a DataFrame with columns:
        path  - full path to the .wav file
        label - integer class index
        emotion - string emotion name
        speaker - OAF or YAF (older/younger adult female)

    Parameters
    ----------
    data_root : str
        Path to the root folder of the TESS dataset.
        Expected structure:
            data_root/
                OAF_angry/  *.wav
                OAF_disgust/ *.wav
                ...
                YAF_angry/  *.wav
                ...

    Returns
    -------
    pd.DataFrame
    """
    records = []

    for folder in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue

        folder_lower = folder.lower()

        # Determine speaker
        if folder_lower.startswith("oaf"):
            speaker = "OAF"
        elif folder_lower.startswith("yaf"):
            speaker = "YAF"
        else:
            speaker = "unknown"

        # Extract emotion from folder name
        emotion = None
        for emo_key in EMOTION_MAP:
            if emo_key in folder_lower:
                emotion = emo_key
                break

        if emotion is None:
            print(f"Warning: Could not parse emotion from folder '{folder}', skipping.")
            continue

        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        if not wav_files:
            wav_files = glob.glob(os.path.join(folder_path, "*.WAV"))

        for wav_path in wav_files:
            records.append({
                "path":    wav_path,
                "label":   EMOTION_MAP[emotion],
                "emotion": emotion,
                "speaker": speaker,
            })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            f"No .wav files found under '{data_root}'. "
            "Check that the TESS dataset is extracted correctly."
        )

    print(f"Loaded {len(df)} audio files across {df['emotion'].nunique()} emotions.")
    print(df["emotion"].value_counts().to_string())
    return df


def split_dataset(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Stratified split into train / validation / test sets.

    Parameters
    ----------
    df : pd.DataFrame  (output of load_tess_paths)
    test_size : float  fraction for test set
    val_size  : float  fraction of remaining data for validation
    random_state : int

    Returns
    -------
    df_train, df_val, df_test : three DataFrames
    """
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    # val_size is expressed as a fraction of the original dataset,
    # so we adjust for the already-removed test portion.
    adjusted_val_size = val_size / (1.0 - test_size)

    df_train, df_val = train_test_split(
        df_train_val,
        test_size=adjusted_val_size,
        stratify=df_train_val["label"],
        random_state=random_state,
    )

    print(f"\nDataset split:")
    print(f"  Train : {len(df_train):>5} samples")
    print(f"  Val   : {len(df_val):>5} samples")
    print(f"  Test  : {len(df_test):>5} samples")

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def get_class_weights(df_train):
    """
    Compute class weights inversely proportional to class frequency.
    Useful for handling mild imbalances in the dataset.

    Returns
    -------
    dict  {class_index: weight}
    """
    counts = df_train["label"].value_counts().sort_index()
    n_samples = len(df_train)
    n_classes = len(counts)

    weights = {}
    for label, count in counts.items():
        weights[label] = n_samples / (n_classes * count)

    print("\nClass weights:")
    for label, w in weights.items():
        print(f"  {EMOTION_LABELS[label]:<18} -> {w:.4f}")

    return weights


if __name__ == "__main__":
    # Quick smoke test - update this path to your local TESS root
    DATA_ROOT = "data/TESS Toronto emotional speech set data"
    df = load_tess_paths(DATA_ROOT)
    df_train, df_val, df_test = split_dataset(df)
    weights = get_class_weights(df_train)