"""
augmentation.py
Audio augmentation techniques for the Speech Emotion Recognition project.

Augmentations implemented:
  1. Add Gaussian noise
  2. Time stretching
  3. Pitch shifting
  4. Time shifting (circular shift)
  5. Frequency masking (SpecAugment-style)
  6. Time masking   (SpecAugment-style)

Augmentations 1-4 operate on raw waveforms.
Augmentations 5-6 operate on spectrograms (2D arrays).
"""

import numpy as np
import librosa


def add_noise(y, noise_factor=0.005):
    """
    Add Gaussian white noise to the audio signal.

    Parameters
    ----------
    y            : np.ndarray  raw audio waveform
    noise_factor : float       controls noise amplitude

    Returns
    -------
    np.ndarray  augmented waveform
    """
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def time_stretch(y, rate=None):
    """
    Stretch or compress the audio in time without changing pitch.

    Parameters
    ----------
    y    : np.ndarray  raw audio waveform
    rate : float or None
           > 1.0 speeds up, < 1.0 slows down.
           If None, rate is sampled uniformly from [0.8, 1.2].

    Returns
    -------
    np.ndarray  time-stretched waveform (same length as input after trim/pad)
    """
    if rate is None:
        rate = np.random.uniform(0.8, 1.2)
    y_stretched = librosa.effects.time_stretch(y, rate=rate)

    # Keep the same length
    if len(y_stretched) > len(y):
        return y_stretched[: len(y)]
    else:
        return np.pad(y_stretched, (0, len(y) - len(y_stretched)), mode="constant")


def pitch_shift(y, sr=22050, n_steps=None):
    """
    Shift the pitch of the audio by n_steps semitones.

    Parameters
    ----------
    y       : np.ndarray  raw audio waveform
    sr      : int         sample rate
    n_steps : float or None
              If None, steps are sampled from [-4, 4] semitones.

    Returns
    -------
    np.ndarray  pitch-shifted waveform
    """
    if n_steps is None:
        n_steps = np.random.uniform(-4, 4)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def time_shift(y, shift_max=0.2):
    """
    Circularly shift the waveform by a random amount.
    Simulates the speech starting at a slightly different offset.

    Parameters
    ----------
    y         : np.ndarray  raw audio waveform
    shift_max : float       maximum shift as a fraction of total length

    Returns
    -------
    np.ndarray  shifted waveform (same length)
    """
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)


def frequency_mask(spec, max_width=10):
    """
    Zero out a contiguous range of frequency bins in a spectrogram.
    SpecAugment-style masking applied along the frequency axis.

    Parameters
    ----------
    spec      : np.ndarray  shape (H, W) or (H, W, C)
    max_width : int         maximum number of frequency bins to mask

    Returns
    -------
    np.ndarray  augmented spectrogram (same shape)
    """
    spec = spec.copy()
    h = spec.shape[0]
    f_width = np.random.randint(0, max_width)
    f_start = np.random.randint(0, max(1, h - f_width))
    spec[f_start : f_start + f_width, :] = 0.0
    return spec


def time_mask(spec, max_width=15):
    """
    Zero out a contiguous range of time frames in a spectrogram.
    SpecAugment-style masking applied along the time axis.

    Parameters
    ----------
    spec      : np.ndarray  shape (H, W) or (H, W, C)
    max_width : int         maximum number of time frames to mask

    Returns
    -------
    np.ndarray  augmented spectrogram (same shape)
    """
    spec = spec.copy()
    w = spec.shape[1]
    t_width = np.random.randint(0, max_width)
    t_start = np.random.randint(0, max(1, w - t_width))
    spec[:, t_start : t_start + t_width] = 0.0
    return spec


def augment_waveform(y, sr=22050, p=0.5):
    """
    Apply a random combination of waveform augmentations.
    Each augmentation is applied independently with probability p.

    Parameters
    ----------
    y  : np.ndarray  raw audio waveform
    sr : int         sample rate
    p  : float       probability of applying each augmentation

    Returns
    -------
    np.ndarray  augmented waveform
    """
    if np.random.rand() < p:
        y = add_noise(y)
    if np.random.rand() < p:
        y = time_stretch(y)
    if np.random.rand() < p:
        y = pitch_shift(y, sr=sr)
    if np.random.rand() < p:
        y = time_shift(y)
    return y


def augment_spectrogram(spec, p=0.5):
    """
    Apply SpecAugment-style masking to a log-mel spectrogram.

    Parameters
    ----------
    spec : np.ndarray  shape (H, W, 1) - single-channel spectrogram
    p    : float       probability of applying each mask

    Returns
    -------
    np.ndarray  augmented spectrogram (same shape)
    """
    if np.random.rand() < p:
        spec = frequency_mask(spec)
    if np.random.rand() < p:
        spec = time_mask(spec)
    return spec


def generate_augmented_sample(path, sr=22050, duration=3.0):
    """
    Load a file, apply waveform augmentation, and return the
    augmented log-mel spectrogram.

    Useful for online augmentation during training.

    Returns
    -------
    np.ndarray  shape (SPEC_HEIGHT, SPEC_WIDTH, 1)
    """
    from feature_extraction import (
        load_audio, extract_log_mel_spectrogram, SAMPLE_RATE
    )

    y, sr = load_audio(path, sr=SAMPLE_RATE, duration=duration)
    y_aug = augment_waveform(y, sr=sr, p=0.5)
    spec  = extract_log_mel_spectrogram(y_aug, sr=sr)
    spec  = augment_spectrogram(spec, p=0.3)
    return spec
