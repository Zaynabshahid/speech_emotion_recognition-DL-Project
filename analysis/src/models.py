"""
models.py
All model architectures for the Speech Emotion Recognition project.

Models:
  1. build_mlp_baseline    - MLP on flat MFCC features
  2. build_cnn             - CNN on log-mel spectrograms
  3. build_cnn_lstm        - CNN encoder + BiLSTM on spectrograms
  4. build_cnn_attention   - CNN + temporal attention (optional extension)

Each function returns a compiled Keras model.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


NUM_CLASSES = 7


def build_mlp_baseline(input_dim=96,
                        hidden_units=(256, 128, 64),
                        dropout_rate=0.3,
                        l2_reg=1e-4,
                        optimizer="adam",
                        learning_rate=1e-3):
    """
    Phase 1 - MLP Baseline.
    Simple dense network on flat MFCC + chroma + ZCR/RMS features.

    Parameters
    ----------
    input_dim    : int    feature vector length (default 96)
    hidden_units : tuple  number of units in each dense layer
    dropout_rate : float  dropout probability
    l2_reg       : float  L2 weight decay
    optimizer    : str    'adam' or 'sgd'
    learning_rate: float

    Returns
    -------
    keras.Model  compiled
    """
    inp = keras.Input(shape=(input_dim,), name="mfcc_input")
    x = inp

    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"dense_{i+1}"
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    out = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)
    model = keras.Model(inputs=inp, outputs=out, name="MLP_Baseline")

    opt = _build_optimizer(optimizer, learning_rate)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(input_shape=(128, 128, 1),
              base_filters=32,
              dropout_rate=0.4,
              l2_reg=1e-4,
              optimizer="adam",
              learning_rate=1e-3):
    """
    Phase 2 - CNN on log-mel spectrograms.
    Four convolutional blocks with increasing filter counts,
    global average pooling, and a dense head.

    Parameters
    ----------
    input_shape  : tuple  (H, W, C)
    base_filters : int    filters in first conv block (doubled each block)
    dropout_rate : float
    l2_reg       : float
    optimizer    : str
    learning_rate: float

    Returns
    -------
    keras.Model  compiled
    """
    inp = keras.Input(shape=input_shape, name="spectrogram_input")
    x = inp

    for i in range(4):
        filters = base_filters * (2 ** i)   # 32, 64, 128, 256
        x = layers.Conv2D(
            filters, kernel_size=3, padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"conv_{i+1}a"
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}a")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}a")(x)

        x = layers.Conv2D(
            filters, kernel_size=3, padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"conv_{i+1}b"
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}b")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}b")(x)

        x = layers.MaxPooling2D(pool_size=2, name=f"pool_{i+1}")(x)
        x = layers.Dropout(dropout_rate * 0.5, name=f"drop_{i+1}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(l2_reg),
                     name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc1")(x)
    x = layers.Activation("relu", name="relu_fc")(x)
    x = layers.Dropout(dropout_rate, name="dropout_fc")(x)

    out = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)
    model = keras.Model(inputs=inp, outputs=out, name="CNN_Spectrogram")

    opt = _build_optimizer(optimizer, learning_rate)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_lstm(input_shape=(128, 128, 1),
                   cnn_filters=(32, 64, 128),
                   lstm_units=128,
                   dropout_rate=0.4,
                   l2_reg=1e-4,
                   optimizer="adam",
                   learning_rate=1e-3):
    """
    Phase 3 - CNN-LSTM Hybrid.
    CNN extracts local spectral features from each time step;
    BiLSTM models temporal dependencies across the sequence.

    Architecture:
      Spectrogram (H, W, 1)
      -> CNN blocks operating on frequency axis per time step
      -> Reshape to sequence (time_steps, feature_dim)
      -> Bidirectional LSTM
      -> Dense head

    Parameters
    ----------
    input_shape  : tuple  (H, W, 1)
    cnn_filters  : tuple  filters in each CNN block
    lstm_units   : int    units in the LSTM layer
    dropout_rate : float
    l2_reg       : float
    optimizer    : str
    learning_rate: float

    Returns
    -------
    keras.Model  compiled
    """
    inp = keras.Input(shape=input_shape, name="spectrogram_input")
    x = inp

    # CNN feature extractor
    for i, filters in enumerate(cnn_filters):
        x = layers.Conv2D(
            filters, kernel_size=(3, 3), padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"cnn_conv_{i+1}"
        )(x)
        x = layers.BatchNormalization(name=f"cnn_bn_{i+1}")(x)
        x = layers.Activation("relu", name=f"cnn_relu_{i+1}")(x)
        x = layers.MaxPooling2D(pool_size=(2, 1), name=f"cnn_pool_{i+1}")(x)

    # After CNN: shape is (batch, reduced_H, W, last_filters)
    # Treat W (time) as sequence length, collapse H and filters
    sh = x.shape
    seq_len = sh[2]   # time frames
    feat_dim = sh[1] * sh[3]   # frequency * channels

    x = layers.Reshape((seq_len, feat_dim), name="reshape_to_seq")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop_before_lstm")(x)

    # Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False, dropout=0.3,
                    recurrent_dropout=0.1),
        name="bilstm"
    )(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(l2_reg),
                     name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc")(x)
    x = layers.Activation("relu", name="relu_fc")(x)
    x = layers.Dropout(dropout_rate, name="dropout_fc")(x)

    out = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)
    model = keras.Model(inputs=inp, outputs=out, name="CNN_BiLSTM")

    opt = _build_optimizer(optimizer, learning_rate)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_attention(input_shape=(128, 128, 1),
                         cnn_filters=(32, 64, 128),
                         dropout_rate=0.4,
                         optimizer="adam",
                         learning_rate=1e-3):
    """
    Optional extension - CNN + temporal self-attention.
    Can be used in the ablation study to compare attention vs LSTM.

    Returns
    -------
    keras.Model  compiled
    """
    inp = keras.Input(shape=input_shape, name="spectrogram_input")
    x = inp

    for i, filters in enumerate(cnn_filters):
        x = layers.Conv2D(filters, 3, padding="same",
                           name=f"conv_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}")(x)
        x = layers.MaxPooling2D(pool_size=(2, 1), name=f"pool_{i+1}")(x)

    sh = x.shape
    seq_len = sh[2]
    feat_dim = sh[1] * sh[3]
    x = layers.Reshape((seq_len, feat_dim), name="reshape")(x)

    # Multi-head self-attention
    attn_out = layers.MultiHeadAttention(
        num_heads=4, key_dim=feat_dim // 4, name="mha"
    )(x, x)
    x = layers.Add(name="residual")([x, attn_out])
    x = layers.LayerNormalization(name="layer_norm")(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc")(x)
    x = layers.Dropout(dropout_rate, name="drop")(x)

    out = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)
    model = keras.Model(inputs=inp, outputs=out, name="CNN_Attention")

    opt = _build_optimizer(optimizer, learning_rate)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_optimizer(name, lr):
    """Return an optimizer instance given a name string."""
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    elif name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=0.9,
                                     nesterov=True)
    elif name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    elif name == "adamw":
        return keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def model_summary_dict(model):
    """Return a dict with model name and parameter count."""
    total  = model.count_params()
    train  = sum(tf.size(v).numpy() for v in model.trainable_variables)
    return {
        "name":              model.name,
        "total_params":      total,
        "trainable_params":  train,
        "non_trainable":     total - train,
    }