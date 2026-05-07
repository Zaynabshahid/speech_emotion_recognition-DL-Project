import os
import sys
import time
from io import BytesIO
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add analysis src folder so we can reuse the existing project modules.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "analysis" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import EMOTION_DISPLAY, EMOTION_LABELS, load_tess_paths, split_dataset, get_class_weights
from evaluate import evaluate_model
from feature_extraction import (SAMPLE_RATE, extract_all_flat, extract_chroma, extract_log_mel_spectrogram,
                                 extract_spectrogram, extract_zcr_rms, extract_mfcc, batch_extract_features, load_audio)
from models import build_cnn, build_cnn_lstm, build_mlp_baseline
from train import train_model
from utils import format_time, load_or_compute

DATA_ROOT_DEFAULT = ROOT_DIR / "TESS Toronto emotional speech set data"
CACHE_DIR = ROOT_DIR / "results" / "cache"
CHECKPOINT_DIR = ROOT_DIR / "results" / "checkpoints"
PLOTS_DIR = ROOT_DIR / "results" / "plots"

EMOTION_ORDER = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]

EMOTION_COLORS = {
    "angry":   "#ff4d6d",
    "disgust": "#c77dff",
    "fear":    "#7b2fff",
    "happy":   "#ffd166",
    "neutral": "#48cae4",
    "ps":      "#06d6a0",
    "sad":     "#4361ee",
}

st.set_page_config(
    page_title="Speech Emotion Studio",
    page_icon="🎙️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
#  GLOBAL STYLES  ·  Dark neural / waveform aesthetic
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">

    <style>
    /* ── ROOT VARIABLES ── */
    :root {
        --bg-base:      #050a14;
        --bg-card:      #0c1525;
        --bg-card2:     #0f1e35;
        --border:       rgba(99,179,255,0.12);
        --border-glow:  rgba(99,179,255,0.35);
        --accent:       #38bdf8;
        --accent2:      #818cf8;
        --accent3:      #34d399;
        --text-primary: #e8f0fe;
        --text-muted:   #7a8ea8;
        --glow-blue:    0 0 24px rgba(56,189,248,0.25);
        --glow-purple:  0 0 24px rgba(129,140,248,0.25);
        --radius:       16px;
        --font-head:    'Syne', sans-serif;
        --font-mono:    'DM Mono', monospace;
        --font-body:    'Inter', sans-serif;
    }

    /* ── BASE RESET ── */
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], section[data-testid="stSidebar"] > div:first-child {
        background: var(--bg-base) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
    }

    /* ── ANIMATED GRID BACKGROUND ── */
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(56,189,248,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(56,189,248,0.04) 1px, transparent 1px);
        background-size: 48px 48px;
        pointer-events: none;
        z-index: 0;
        animation: gridPulse 8s ease-in-out infinite;
    }
    @keyframes gridPulse {
        0%, 100% { opacity: 0.6; }
        50%       { opacity: 1.0; }
    }

    /* ── BLOCK CONTAINER ── */
    .block-container {
        padding: 2.5rem 3rem !important;
        position: relative;
        z-index: 1;
    }

    /* ── HEADER / TITLE ── */
    h1 {
        font-family: var(--font-head) !important;
        font-weight: 800 !important;
        font-size: clamp(2rem, 4vw, 3.2rem) !important;
        letter-spacing: -0.03em !important;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #34d399 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin-bottom: 0.25rem !important;
        animation: shimmer 4s ease-in-out infinite;
        background-size: 200% 200% !important;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 0% 50%; }
        50%       { background-position: 100% 50%; }
    }

    h2, h3 {
        font-family: var(--font-head) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
    }
    h2 { font-size: 1.6rem !important; }
    h3 { font-size: 1.2rem !important; color: var(--accent) !important; }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: var(--bg-card) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div,
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background: var(--bg-base) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: var(--accent) !important;
    }
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"],
    [data-testid="stSidebar"] .stSlider label {
        color: var(--text-muted) !important;
    }

    /* ── SIDEBAR LABEL / HEADER ── */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        background: none !important;
        -webkit-text-fill-color: var(--text-primary) !important;
        color: var(--text-primary) !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
    }

    /* ── METRICS ── */
    [data-testid="metric-container"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1.2rem 1.4rem !important;
        transition: border-color 0.3s, box-shadow 0.3s, transform 0.3s !important;
        position: relative;
        overflow: hidden;
    }
    [data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
        opacity: 0;
        transition: opacity 0.3s;
    }
    [data-testid="metric-container"]:hover {
        border-color: var(--border-glow) !important;
        box-shadow: var(--glow-blue) !important;
        transform: translateY(-3px) !important;
    }
    [data-testid="metric-container"]:hover::before {
        opacity: 1;
    }
    [data-testid="stMetricValue"] {
        font-family: var(--font-head) !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        color: var(--accent) !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: var(--font-mono) !important;
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: var(--text-muted) !important;
    }
    [data-testid="stMetricDelta"] {
        color: var(--accent3) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.78rem !important;
    }

    /* ── BUTTONS ── */
    .stButton > button {
        font-family: var(--font-head) !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.04em !important;
        border-radius: 999px !important;
        border: none !important;
        padding: 0.65rem 1.8rem !important;
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
        color: #fff !important;
        cursor: pointer !important;
        transition: transform 0.2s, box-shadow 0.2s, opacity 0.2s !important;
        box-shadow: 0 4px 18px rgba(56,189,248,0.25) !important;
        position: relative;
        overflow: hidden;
    }
    .stButton > button::after {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.15), transparent);
        opacity: 0;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 30px rgba(56,189,248,0.4) !important;
    }
    .stButton > button:hover::after { opacity: 1; }
    .stButton > button:active { transform: translateY(0) scale(0.98) !important; }

    /* secondary buttons */
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid var(--border-glow) !important;
        color: var(--accent) !important;
        box-shadow: none !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(56,189,248,0.08) !important;
        box-shadow: var(--glow-blue) !important;
    }

    /* ── EXPANDER ── */
    .streamlit-expanderHeader {
        font-family: var(--font-head) !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        color: var(--text-primary) !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1rem 1.4rem !important;
        transition: border-color 0.3s, box-shadow 0.3s !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: var(--border-glow) !important;
        box-shadow: var(--glow-blue) !important;
    }
    .streamlit-expanderContent {
        background: var(--bg-card2) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius) var(--radius) !important;
        padding: 1.4rem !important;
    }

    /* ── INPUTS ── */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: var(--font-mono) !important;
        transition: border-color 0.3s !important;
    }
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
    }
    .stSelectbox label, .stTextInput label,
    .stTextArea label, .stRadio label,
    .stFileUploader label, .stCheckbox label {
        color: var(--text-muted) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }

    /* ── RADIO ── */
    .stRadio > div {
        gap: 0.6rem !important;
    }
    .stRadio > div > label {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 0.55rem 1rem !important;
        transition: border-color 0.25s, background 0.25s !important;
        cursor: pointer !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
        font-size: 0.88rem !important;
        text-transform: none !important;
        letter-spacing: normal !important;
    }
    .stRadio > div > label:hover {
        border-color: var(--accent) !important;
        background: rgba(56,189,248,0.06) !important;
    }

    /* ── FILE UPLOADER ── */
    [data-testid="stFileUploader"] {
        background: var(--bg-card) !important;
        border: 1.5px dashed var(--border) !important;
        border-radius: var(--radius) !important;
        transition: border-color 0.3s, background 0.3s !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
        background: rgba(56,189,248,0.04) !important;
    }
    [data-testid="stFileUploader"] * { color: var(--text-muted) !important; }

    /* ── AUDIO PLAYER ── */
    .stAudio > audio {
        width: 100% !important;
        border-radius: 12px !important;
        filter: invert(1) hue-rotate(180deg) !important;
    }

    /* ── DATAFRAME ── */
    [data-testid="stDataFrame"] {
        border-radius: var(--radius) !important;
        overflow: hidden !important;
        border: 1px solid var(--border) !important;
    }
    [data-testid="stDataFrame"] th {
        background: var(--bg-card2) !important;
        color: var(--accent) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        border-bottom: 1px solid var(--border) !important;
    }
    [data-testid="stDataFrame"] td {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.82rem !important;
        border-bottom: 1px solid rgba(99,179,255,0.05) !important;
    }

    /* ── ALERTS / MESSAGES ── */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: var(--radius) !important;
        border: none !important;
        font-family: var(--font-mono) !important;
        font-size: 0.85rem !important;
    }
    .stSuccess { background: rgba(52,211,153,0.1) !important; color: #34d399 !important; }
    .stInfo    { background: rgba(56,189,248,0.1) !important; color: #38bdf8 !important; }
    .stWarning { background: rgba(251,191,36,0.1) !important; color: #fbbf24 !important; }
    .stError   { background: rgba(248,113,113,0.1) !important; color: #f87171 !important; }

    /* ── SPINNER ── */
    .stSpinner > div > div {
        border-top-color: var(--accent) !important;
    }

    /* ── DIVIDER ── */
    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 2rem 0 !important;
    }

    /* ── CAPTION ── */
    .stCaption, .stMarkdown small {
        color: var(--text-muted) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.76rem !important;
    }

    /* ── MARKDOWN BODY TEXT ── */
    .stMarkdown p {
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
        line-height: 1.75 !important;
    }
    code, pre {
        background: rgba(56,189,248,0.08) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--accent) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.82rem !important;
    }

    /* ── PROGRESS / SPINNER BAR ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
        border-radius: 999px !important;
    }
    .stProgress > div {
        background: var(--bg-card) !important;
        border-radius: 999px !important;
    }

    /* ── WAVEFORM DECORATION (hero header) ── */
    .hero-wave {
        display: flex;
        align-items: flex-end;
        gap: 3px;
        height: 28px;
        margin-bottom: 0.5rem;
    }
    .hero-wave span {
        display: inline-block;
        width: 4px;
        background: var(--accent);
        border-radius: 2px;
        animation: wave 1.4s ease-in-out infinite;
        opacity: 0.7;
    }
    .hero-wave span:nth-child(1) { height: 8px;  animation-delay: 0.0s; }
    .hero-wave span:nth-child(2) { height: 18px; animation-delay: 0.1s; }
    .hero-wave span:nth-child(3) { height: 28px; animation-delay: 0.2s; background: var(--accent2); }
    .hero-wave span:nth-child(4) { height: 14px; animation-delay: 0.3s; }
    .hero-wave span:nth-child(5) { height: 22px; animation-delay: 0.4s; background: var(--accent3); }
    .hero-wave span:nth-child(6) { height: 10px; animation-delay: 0.5s; }
    .hero-wave span:nth-child(7) { height: 26px; animation-delay: 0.6s; background: var(--accent2); }
    .hero-wave span:nth-child(8) { height: 16px; animation-delay: 0.7s; }
    .hero-wave span:nth-child(9) { height: 6px;  animation-delay: 0.8s; }
    @keyframes wave {
        0%, 100% { transform: scaleY(0.4); opacity: 0.5; }
        50%       { transform: scaleY(1.0); opacity: 1.0; }
    }

    /* ── BADGE CHIPS ── */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border: 1px solid;
        margin-right: 0.4rem;
    }
    .badge-blue   { color: #38bdf8; border-color: rgba(56,189,248,0.4);  background: rgba(56,189,248,0.08);  }
    .badge-purple { color: #818cf8; border-color: rgba(129,140,248,0.4); background: rgba(129,140,248,0.08); }
    .badge-green  { color: #34d399; border-color: rgba(52,211,153,0.4);  background: rgba(52,211,153,0.08);  }

    /* ── SECTION HEADING RULE ── */
    .section-heading {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 1.8rem 0 1.2rem 0;
    }
    .section-heading::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--border-glow), transparent);
    }
    .section-heading span {
        font-family: var(--font-head);
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    /* ── BAR CHART AREA ── */
    [data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
#  CACHED FUNCTIONS (UNCHANGED LOGIC)
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dataset(data_root):
    return load_tess_paths(data_root)


@st.cache_data(show_spinner=False)
def compute_feature_cache(cache_path, df, _feature_fn):
    return load_or_compute(cache_path, batch_extract_features, df, _feature_fn)


def get_feature_cache(df, model_type, data_root):
    os.makedirs(CACHE_DIR, exist_ok=True)
    root_tag = Path(data_root).stem.replace(" ", "_")
    if model_type == "Baseline MLP":
        cache_path = CACHE_DIR / f"features_mlp_{root_tag}.npz"
        feature_fn = extract_all_flat
    else:
        cache_path = CACHE_DIR / f"features_spec_{root_tag}.npz"
        feature_fn = extract_spectrogram

    X, y = compute_feature_cache(str(cache_path), df, feature_fn)
    return X, y


def safe_save_model(model, model_name):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = CHECKPOINT_DIR / f"{model_name}_best.keras"
    model.save(path, overwrite=True, save_format="keras")
    return str(path)


def get_gpu_status():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return f"🟢 GPU · {gpus[0].name}"
        return "🟡 CPU only"
    except Exception as exc:
        return f"⚠️ {exc}"


def load_audio_source(sample_path=None, audio_bytes=None):
    if sample_path is not None:
        return load_audio(sample_path)
    if audio_bytes is not None:
        y, sr = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
        target_length = int(SAMPLE_RATE * 3.0)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode="constant")
        else:
            y = y[:target_length]
        return y, sr
    raise ValueError("No audio source provided.")


def extract_features_from_source(model_type, sample_path=None, audio_bytes=None):
    if sample_path is not None:
        if model_type == "Baseline MLP":
            return extract_all_flat(sample_path)
        return extract_spectrogram(sample_path)

    y, sr = load_audio_source(audio_bytes=audio_bytes)
    if model_type == "Baseline MLP":
        return np.concatenate([extract_mfcc(y, sr), extract_chroma(y, sr), extract_zcr_rms(y)])
    return extract_log_mel_spectrogram(y, sr)


@st.cache_resource(show_spinner=False)
def load_saved_model(model_type):
    file_map = {
        "Baseline MLP":    CHECKPOINT_DIR / "baseline_mlp_best.keras",
        "CNN Spectrogram": CHECKPOINT_DIR / "cnn_spectrogram_best.keras",
        "CNN + LSTM":      CHECKPOINT_DIR / "cnn_lstm_best.keras",
    }
    model_path = file_map.get(model_type)
    if model_path and model_path.exists():
        if model_type == "Baseline MLP":
            model = build_mlp_baseline()
        elif model_type == "CNN Spectrogram":
            model = build_cnn()
        else:
            model = build_cnn_lstm()
        model.load_weights(str(model_path))
        return model
    return None


def build_model(model_type):
    if model_type == "Baseline MLP":
        return build_mlp_baseline()
    if model_type == "CNN Spectrogram":
        return build_cnn()
    if model_type == "CNN + LSTM":
        return build_cnn_lstm()
    raise ValueError("Unknown model type")


def prepare_input(sample, model_type):
    if model_type == "Baseline MLP":
        return sample.reshape(1, -1)
    return sample.reshape(1, *sample.shape)


def predict(model, model_type, data):
    X = prepare_input(data, model_type)
    probs = model.predict(X, verbose=0)[0]
    best = int(np.argmax(probs))
    return best, probs


def plot_spectrogram(y, sr=SAMPLE_RATE, title="Spectrogram"):
    fig, ax = plt.subplots(figsize=(6, 3))
    spec = extract_log_mel_spectrogram(y, sr)[:, :, 0]
    im = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title, color="#e8f0fe", fontsize=10, pad=8)
    ax.set_xlabel("Time frames", color="#7a8ea8", fontsize=8)
    ax.set_ylabel("Mel bands", color="#7a8ea8", fontsize=8)
    ax.tick_params(colors="#7a8ea8", labelsize=7)
    fig.colorbar(im, ax=ax, format="%.2f", label="Norm. power")
    ax.set_facecolor("#0f2138")
    fig.patch.set_facecolor("#112a46")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1a3450")
    ax.spines["bottom"].set_color("#1a3450")
    return fig


def render_emotion_probabilities(probs):
    df = pd.DataFrame(
        {
            "Emotion":     [EMOTION_DISPLAY[emo] for emo in EMOTION_ORDER],
            "Probability": [float(probs[i]) for i in range(len(EMOTION_ORDER))],
        }
    )
    df["Emotion"] = pd.Categorical(df["Emotion"], categories=[EMOTION_DISPLAY[emo] for emo in EMOTION_ORDER])
    return df


def format_seconds(seconds):
    return format_time(int(seconds))


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    # ── HERO HEADER ──────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-wave">
          <span></span><span></span><span></span><span></span>
          <span></span><span></span><span></span><span></span><span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("Speech Emotion Studio")
    st.markdown(
        """
        <p style="color:#7a8ea8;font-size:1rem;font-family:'DM Mono',monospace;margin-top:-0.5rem;margin-bottom:1rem;">
        CS-419 · Deep Learning · TESS Dataset &nbsp;·&nbsp;
        <span class="badge badge-blue">7 Emotions</span>
        <span class="badge badge-purple">3 Architectures</span>
        <span class="badge badge-green">Live Inference</span>
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── SIDEBAR ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<p style='font-family:\"DM Mono\",monospace;font-size:0.7rem;text-transform:uppercase;"
            "letter-spacing:0.12em;color:#38bdf8;margin-bottom:0.3rem;'>⚙ Configuration</p>",
            unsafe_allow_html=True,
        )
        data_root = st.text_input("TESS dataset root path", value=str(DATA_ROOT_DEFAULT))
        model_type = st.selectbox("Architecture", ["Baseline MLP", "CNN Spectrogram", "CNN + LSTM"])
        epochs     = st.slider("Epochs", min_value=4, max_value=40, value=12, step=4)
        batch_size = st.select_slider("Batch size", options=[16, 32, 48, 64], value=32)
        st.markdown("---")
        st.markdown(
            "<p style='font-family:\"DM Mono\",monospace;font-size:0.7rem;text-transform:uppercase;"
            "letter-spacing:0.12em;color:#38bdf8;margin-bottom:0.3rem;'>⚡ Execution</p>",
            unsafe_allow_html=True,
        )
        st.write(get_gpu_status())

    # ── PROJECT OVERVIEW EXPANDER ────────────────────────────
    with st.expander("🔬  Project overview", expanded=True):
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(
                """
                This app uses the **Toronto Emotional Speech Set** to classify seven emotions from short
                audio clips. Explore the dataset, train a model, and run live predictions on new samples.

                **What's included**
                - MFCC + spectrogram feature extraction
                - Baseline MLP on flat features
                - CNN on log-mel spectrograms
                - CNN + LSTM hybrid for temporal modelling
                - Audio preview, spectrogram visualisation, and live predictions
                """
            )
        with col2:
            st.markdown(
                """
                **Emotion palette**

                | Label | Emotion |
                |---|---|
                | `angry` | 😠 Anger |
                | `disgust` | 🤢 Disgust |
                | `fear` | 😨 Fear |
                | `happy` | 😊 Happiness |
                | `neutral` | 😐 Neutral |
                | `ps` | 😲 Pleasant Surprise |
                | `sad` | 😢 Sadness |
                """
            )

    # ── VALIDATE DATASET ─────────────────────────────────────
    if not os.path.isdir(data_root):
        st.error(
            "Dataset root path is not valid. Update the path in the sidebar so it points to "
            "the extracted TESS folder."
        )
        return

    with st.spinner("Loading dataset metadata…"):
        df = load_dataset(data_root)

    train_df, val_df, test_df = split_dataset(df)
    class_weights = get_class_weights(train_df)

    saved_files  = list(CHECKPOINT_DIR.glob("*.keras")) if CHECKPOINT_DIR.exists() else []
    cached_files = list(CACHE_DIR.glob("*.npz"))        if CACHE_DIR.exists()       else []

    # ── STATS CARDS ──────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total clips",    len(df),        delta=f"{len(train_df)} train")
    col2.metric("Train samples",  len(train_df),  delta=f"{len(val_df)} val")
    col3.metric("Test samples",   len(test_df),   delta="7 emotions")
    col4.metric("Selected model", model_type)

    # ── DATASET SNAPSHOT ─────────────────────────────────────
    st.markdown(
        '<div class="section-heading"><span> Dataset snapshot</span></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(
            df.sample(min(len(df), 10), random_state=42)[["emotion", "speaker", "path"]].reset_index(drop=True),
            use_container_width=True,
        )
    with c2:
        fig, ax = plt.subplots(figsize=(6, 3))
        counts     = df["emotion"].value_counts().reindex(EMOTION_ORDER)
        bar_colors = [EMOTION_COLORS.get(e, "#38bdf8") for e in EMOTION_ORDER]
        bars = ax.bar(
            [EMOTION_DISPLAY[e] for e in EMOTION_ORDER],
            counts.values,
            color=bar_colors,
            width=0.6,
            zorder=3,
        )
        ax.set_ylabel("Count", color="#a5b8d6", fontsize=8)
        ax.set_xticklabels([EMOTION_DISPLAY[e] for e in EMOTION_ORDER], rotation=35, ha="right", color="#d2e0ff", fontsize=8)
        for x, v in enumerate(counts.values):
            ax.text(x, v + 2, str(v), ha="center", color="#d2e0ff", fontsize=8)
        ax.set_facecolor("#133a5f")
        fig.patch.set_facecolor("#162f54")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#20507a")
        ax.spines["bottom"].set_color("#20507a")
        ax.tick_params(colors="#a5b8d6", labelsize=7)
        ax.yaxis.label.set_color("#a5b8d6")
        ax.grid(axis="y", color="#1d4063", linestyle="--", alpha=0.55, zorder=0)
        st.pyplot(fig)

    # ── TRAIN A MODEL ─────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-heading"><span> Train a model</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#7a8ea8;font-size:0.9rem;font-family:\"Inter\",sans-serif;'>"
        "Choose an architecture in the sidebar, then launch training. "
        "Feature extraction is cached so repeated runs are fast.</p>",
        unsafe_allow_html=True,
    )

    if st.button("▶  Train model now", type="primary"):
        with st.spinner("Extracting features and training…"):
            X_train, y_train = get_feature_cache(train_df, model_type, data_root)
            X_val,   y_val   = get_feature_cache(val_df,   model_type, data_root)
            model       = build_model(model_type)
            start_time  = time.time()
            history     = train_model(
                model, X_train, y_train, X_val, y_val,
                model_name=model_type.replace(" ", "_").lower(),
                epochs=epochs, batch_size=batch_size,
                class_weights=class_weights,
            )
            duration    = format_seconds(time.time() - start_time)
            model_path  = safe_save_model(model, model_type.replace(" ", "_").lower())

            st.success(f"Training finished in {duration}. Model saved → `{model_path}`")
            st.session_state["latest_model"]      = model
            st.session_state["latest_model_type"] = model_type
            st.session_state["latest_history"]    = history.history

    if st.session_state.get("latest_history"):
        hist = st.session_state["latest_history"]
        st.markdown(
            '<div class="section-heading"><span>📈 Training curves</span></div>',
            unsafe_allow_html=True,
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.patch.set_facecolor("#162f54")
        for ax in axes:
            ax.set_facecolor("#133a5f")
            ax.grid(color="#1d4063", linestyle="--", alpha=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#20507a")
            ax.spines["bottom"].set_color("#20507a")
            ax.tick_params(colors="#a5b8d6", labelsize=8)

        axes[0].plot(hist["accuracy"],     label="Train", linewidth=2.5, color="#38bdf8")
        axes[0].plot(hist["val_accuracy"], label="Val",   linewidth=2.5, color="#818cf8", linestyle="--")
        axes[0].set_title("Accuracy", color="#e8f0fe", fontsize=10, pad=8)
        axes[0].legend(facecolor="#0c1525", edgecolor="#1a2740", labelcolor="#e8f0fe", fontsize=8)

        axes[1].plot(hist["loss"],     label="Train", linewidth=2.5, color="#34d399")
        axes[1].plot(hist["val_loss"], label="Val",   linewidth=2.5, color="#fbbf24", linestyle="--")
        axes[1].set_title("Loss", color="#e8f0fe", fontsize=10, pad=8)
        axes[1].legend(facecolor="#0c1525", edgecolor="#1a2740", labelcolor="#e8f0fe", fontsize=8)

        st.pyplot(fig)

    # ── SAVED RESULTS ────────────────────────────────────────
    if saved_files or cached_files:
        st.markdown("---")
        st.markdown(
            '<div class="section-heading"><span> Saved results</span></div>',
            unsafe_allow_html=True,
        )
        file_cols = st.columns(2)
        with file_cols[0]:
            st.markdown("**Checkpoints**")
            if saved_files:
                for path in saved_files:
                    st.markdown(f"- `{path.name}`")
            else:
                st.write("No saved checkpoints yet.")
        with file_cols[1]:
            st.markdown("**Cached features**")
            if cached_files:
                for path in cached_files:
                    st.markdown(f"- `{path.name}`")
            else:
                st.write("No cached feature files yet.")

        if st.button("Evaluate latest saved model", type="secondary"):
            if saved_files:
                with st.spinner("Loading checkpoint and evaluating…"):
                    model = load_saved_model(model_type)
                    if model is not None:
                        X_test, y_test = get_feature_cache(test_df, model_type, data_root)
                        results = evaluate_model(model, X_test, y_test, model_name=model_type)
                        ec1, ec2 = st.columns(2)
                        ec1.metric("Test accuracy", f"{results['accuracy']:.4f}")
                        ec2.metric("Macro F1",      f"{results['macro_f1']:.4f}")
                        if (ROOT_DIR / "results" / "plots").exists():
                            st.caption("Evaluation plots saved to `results/plots/`")
                    else:
                        st.warning("No saved model available for the selected architecture.")
            else:
                st.warning("No saved checkpoint files were found.")

    # ── AUDIO SAMPLE + PREDICTION ────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-heading"><span> Try an audio sample</span></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        audio_source = st.radio("Choose audio input", ["Dataset sample", "Upload audio file"])
        if audio_source == "Dataset sample":
            emotion_choice = st.selectbox("Select emotion", [EMOTION_DISPLAY[e] for e in EMOTION_ORDER])
            emotion_key    = [k for k, v in EMOTION_DISPLAY.items() if v == emotion_choice][0]
            sample_df      = df[df["emotion"] == emotion_key].sample(1, random_state=42)
            sample_path    = sample_df.iloc[0]["path"]
            st.caption(f"File → `{sample_path}`")
            y, sr = load_audio(sample_path)
            st.audio(sample_path)
        else:
            uploaded_file = st.file_uploader("Upload a WAV / MP3 / FLAC clip", type=["wav", "mp3", "flac"])
            if uploaded_file is not None:
                audio_data  = uploaded_file.read()
                y, sr       = librosa.load(BytesIO(audio_data), sr=SAMPLE_RATE)
                st.audio(audio_data)
                sample_path = None
            else:
                y, sr = None, None

        if y is not None:
            if len(y) < int(SAMPLE_RATE * 3.0):
                y = np.pad(y, (0, int(SAMPLE_RATE * 3.0) - len(y)), mode="constant")
            else:
                y = y[: int(SAMPLE_RATE * 3.0)]
            fig = plot_spectrogram(y, sr, title="Log-mel spectrogram")
            st.pyplot(fig)
            st.caption(f"Sample rate: {sr} Hz  ·  Duration: {len(y)/sr:.2f} s")

    with c2:
        st.markdown("**Prediction**")
        loaded_model = st.session_state.get("latest_model")
        if loaded_model is None:
            loaded_model = load_saved_model(model_type)
            if loaded_model is not None:
                st.success(f"Loaded {model_type} from disk.")
                st.session_state["latest_model"]      = loaded_model
                st.session_state["latest_model_type"] = model_type

        if loaded_model is None:
            st.info("Train a model or reload after training to enable predictions.")
        elif y is not None:
            if audio_source == "Dataset sample":
                sample_features = extract_features_from_source(model_type, sample_path=sample_path)
            else:
                sample_features = extract_features_from_source(model_type, audio_bytes=audio_data)
            pred_label, probs  = predict(loaded_model, model_type, sample_features)
            emotion_name       = EMOTION_DISPLAY[EMOTION_ORDER[pred_label]]
            st.metric("Predicted emotion", emotion_name)
            st.bar_chart(render_emotion_probabilities(probs).set_index("Emotion"))
        else:
            st.write("Upload or select a clip to see predictions.")

    # ── PROJECT DETAILS ──────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-heading"><span> Project details</span></div>',
        unsafe_allow_html=True,
    )
    st.write(
        "This interface uses the existing audio preprocessing and model definitions in `analysis/src/`. "
        "It loads TESS speech files, extracts MFCC / spectrogram features, trains a selected model, "
        "and renders predictions live."
    )
    st.write(
        "For a quick demo choose **Baseline MLP** and train for 4–8 epochs. "
        "For the most expressive temporal model use **CNN + LSTM**."
    )
    st.caption(
        "Dataset root path must point to the extracted TESS folder containing `OAF_*` and `YAF_*` subfolders."
    )


if __name__ == "__main__":
    main()