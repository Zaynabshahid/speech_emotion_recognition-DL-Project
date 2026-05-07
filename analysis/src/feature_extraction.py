"""
feature_extraction.py
All audio feature extraction logic for the Speech Emotion Recognition project.

Features computed:
  - MFCC (Mel-Frequency Cepstral Coefficients)     -> used for MLP baseline
  - Log-Mel Spectrogram                              -> used as CNN input image
  - Chroma features                                  -> optional supplementary
  - Zero Crossing Rate, RMS Energy                   -> optional for ablation
"""

import numpy as np
import librosa
import librosa.display


# Global audio config kept in one place so every module imports from here.
SAMPLE_RATE    = 22050   # standard librosa default
DURATION       = 3.0     # seconds - TESS clips are ~2.5s, we pad/trim to 3s
N_MFCC         = 40      # number of MFCC coefficients
N_MELS         = 128     # mel filterbanks for spectrogram
HOP_LENGTH     = 512
N_FFT          = 2048
SPEC_HEIGHT    = 128     # final spectrogram image height (pixels)
SPEC_WIDTH     = 128     # final spectrogram image width  (pixels)


def load_audio(path, sr=SAMPLE_RATE, duration=DURATION):
    """
    Load a .wav file, resample to sr, and pad/trim to fixed duration.

    Returns
    -------
    y  : np.ndarray  shape (sr * duration,)
    sr : int
    """
    y, sr_orig = librosa.load(path, sr=sr, mono=True)

    target_length = int(sr * duration)

    if len(y) < target_length:
        # Pad with zeros at the end
        y = np.pad(y, (0, target_length - len(y)), mode="constant")
    else:
        y = y[:target_length]

    return y, sr


def extract_mfcc(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    Extract mean + std of MFCC coefficients over time.
    Returns a flat vector of length 2 * n_mfcc.

    This is the input for the MLP baseline model.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc,  axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])   # shape: (2 * n_mfcc,)


def extract_mfcc_2d(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    Extract the full 2D MFCC matrix (coefficients x time frames).
    Used as input to CNN/LSTM models that need temporal structure.

    Returns
    -------
    np.ndarray  shape (n_mfcc, T)  where T is number of time frames
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    # Delta and delta-delta MFCC for richer representation
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.stack([mfcc, mfcc_delta, mfcc_delta2], axis=0)  # (3, n_mfcc, T)
    return combined


def extract_log_mel_spectrogram(y, sr=SAMPLE_RATE,
                                 n_mels=N_MELS,
                                 target_h=SPEC_HEIGHT,
                                 target_w=SPEC_WIDTH):
    """
    Compute log-mel spectrogram and resize to a fixed (H, W) image.
    This is the primary input for the CNN and CNN-LSTM models.

    Returns
    -------
    np.ndarray  shape (target_h, target_w, 1)   - single channel grayscale
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=n_mels,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

    # Resize to fixed dimensions using simple interpolation
    from PIL import Image
    img = Image.fromarray((log_mel * 255).astype(np.uint8))
    img = img.resize((target_w, target_h), Image.BILINEAR)
    spec = np.array(img, dtype=np.float32) / 255.0
    spec = np.expand_dims(spec, axis=-1)   # (H, W, 1)

    return spec


def extract_chroma(y, sr=SAMPLE_RATE):
    """
    Chroma features capture pitch-class information.
    Returns mean over time frames: shape (12,)
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.mean(chroma, axis=1)


def extract_zcr_rms(y):
    """
    Zero crossing rate and RMS energy - simple prosodic features.
    Returns a 4-element vector: [zcr_mean, zcr_std, rms_mean, rms_std]
    """
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    return np.array([
        np.mean(zcr), np.std(zcr),
        np.mean(rms), np.std(rms),
    ])


def extract_all_flat(path):
    """
    Convenience function: loads audio and returns a single flat
    feature vector combining MFCC + chroma + ZCR/RMS.
    Used by the MLP baseline.

    Returns
    -------
    np.ndarray  shape (2*N_MFCC + 12 + 4,)  = (96,)
    """
    y, sr = load_audio(path)
    mfcc_feats  = extract_mfcc(y, sr)
    chroma_feats = extract_chroma(y, sr)
    zcr_rms_feats = extract_zcr_rms(y)
    return np.concatenate([mfcc_feats, chroma_feats, zcr_rms_feats])


def extract_spectrogram(path):
    """
    Convenience function: loads audio and returns log-mel spectrogram image.
    Used by CNN and CNN-LSTM models.

    Returns
    -------
    np.ndarray  shape (SPEC_HEIGHT, SPEC_WIDTH, 1)
    """
    y, sr = load_audio(path)
    return extract_log_mel_spectrogram(y, sr)


def batch_extract_features(df, feature_fn, desc="Extracting features"):
    """
    Apply a feature extraction function to every row in a DataFrame.

    Parameters
    ----------
    df         : pd.DataFrame with a 'path' column
    feature_fn : callable that takes a file path and returns np.ndarray
    desc       : string label for progress output

    Returns
    -------
    X : np.ndarray  shape (N, *feature_shape)
    y : np.ndarray  shape (N,)  integer labels
    """
    features = []
    labels   = []

    for i, row in df.iterrows():
        if i % 200 == 0:
            print(f"  {desc}: {i}/{len(df)}")
        feat = feature_fn(row["path"])
        features.append(feat)
        labels.append(row["label"])

    X = np.array(features, dtype=np.float32)
    y = np.array(labels,   dtype=np.int32)
    return X, y