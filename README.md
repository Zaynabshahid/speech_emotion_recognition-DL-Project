# Speech Emotion Recognition
## CS-419 Deep Learning Semester Project

A comprehensive deep learning project for multi-class audio emotion classification using the TESS (Toronto Emotional Speech Set) dataset. This project explores multiple neural network architectures, from baseline MLPs to advanced CNN-LSTM hybrid models, with detailed analysis and visualization.

**Dataset:** TESS (Toronto Emotional Speech Set)  
**Task:** Multi-class audio emotion classification  
**Classes:** angry, disgust, fear, happy, neutral, pleasant surprise, sad  
**Models Implemented:** MLP Baseline, CNN, CNN-LSTM  

---

## Project Structure

```
speech_emotion_recognition/
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   ├── 02_Feature_Extraction.ipynb   # MFCC + Spectrogram extraction
│   ├── 03_Baseline_MLP.ipynb         # Phase 1 - Baseline model
│   ├── 04_CNN_Model.ipynb            # Phase 2 - CNN on spectrograms
│   ├── 05_CNN_LSTM.ipynb             # Phase 3 - CNN + LSTM hybrid
│   ├── 06_Ablation_Study.ipynb       # Phase 4 - Ablation experiments
│   └── 07_GradCAM_Analysis.ipynb     # Phase 5 - Visualization & error analysis
├── src/
│   ├── data_loader.py                # Dataset loading and splitting
│   ├── feature_extraction.py         # MFCC, spectrogram, chroma features
│   ├── augmentation.py               # Audio augmentation techniques
│   ├── models.py                     # All model architectures
│   ├── train.py                      # Training loop with callbacks
│   ├── evaluate.py                   # Evaluation metrics and plots
│   └── utils.py                      # Helper functions
├── results/
│   └── (auto-generated plots, CSVs, saved models)
├── data/
│   └── (TESS dataset - download separately)
├── requirements.txt                  # Python dependencies
├── app.py                            # Streamlit interactive demo
└── README.md                         # This file
```

---

## Emotion Classes

| Label | Description |
|-------|-------------|
| angry | Expressed anger |
| disgust | Expressed disgust |
| fear | Expressed fear |
| happy | Expressed happiness |
| neutral | Neutral speech |
| ps | Pleasant surprise |
| sad | Expressed sadness |

---

## Models Compared

| Model | Input Features | Architecture | Purpose |
|-------|---|---|---|
| MLP Baseline | Raw MFCCs (40 coefficients) | 3-layer dense network | Baseline performance |
| CNN | Log-mel spectrogram (128×128) | 4 conv blocks + dense layers | Spatial feature learning |
| CNN-LSTM | Log-mel spectrogram (128×128) | CNN encoder + Bidirectional LSTM | Temporal + spatial features |

---

## Quick Start

### Prerequisites
- Python 3.7+
- pip package manager
- Virtual environment (recommended)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install librosa numpy pandas matplotlib seaborn scikit-learn tensorflow keras streamlit
```

### Step 2: Download Dataset
Download TESS from Kaggle:
https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

Extract and place in:
```
data/TESS Toronto emotional speech set data/
```

### Step 3: Run Notebooks
Open each notebook in Google Colab or Jupyter and execute top to bottom:
1. **01_EDA.ipynb** - Understand data distribution and characteristics
2. **02_Feature_Extraction.ipynb** - Extract MFCC and spectrogram features
3. **03_Baseline_MLP.ipynb** - Train and evaluate baseline MLP model
4. **04_CNN_Model.ipynb** - Train and evaluate CNN model
5. **05_CNN_LSTM.ipynb** - Train and evaluate CNN-LSTM hybrid model
6. **06_Ablation_Study.ipynb** - Analyze architectural variations
7. **07_GradCAM_Analysis.ipynb** - Visualize model decisions

### Step 4: Run Interactive Demo

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

#### On macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. You can upload audio files or use the dataset selector to test emotion predictions.

---

## Key Features

- **Exploratory Data Analysis:** Visualize audio distributions, emotion class balance, and audio characteristics
- **Feature Engineering:** MFCC, mel-spectrograms, and chroma features with audio augmentation
- **Multiple Architectures:** From simple MLPs to advanced CNN-LSTM models
- **Comprehensive Evaluation:** Accuracy, precision, recall, F1-score, and confusion matrices
- **Ablation Study:** Understand the impact of different architectural components
- **Explainability:** GradCAM visualization to interpret model predictions
- **Interactive Demo:** Streamlit app for real-time emotion classification

---

## Project Workflow

1. **Data Exploration** → Understand emotion distributions and audio characteristics
2. **Feature Extraction** → Convert raw audio to MFCC and spectrogram features
3. **Baseline Modeling** → Establish performance baseline with MLP
4. **Advanced Models** → Explore CNN and CNN-LSTM architectures
5. **Ablation Study** → Analyze impact of architectural choices
6. **Visualization & Analysis** → Use GradCAM to interpret predictions

---

## Dependencies

Key packages used in this project:
- **librosa** - Audio processing and feature extraction
- **tensorflow/keras** - Deep learning framework
- **numpy, pandas** - Numerical and data manipulation
- **matplotlib, seaborn** - Data visualization
- **scikit-learn** - Machine learning utilities
- **streamlit** - Interactive web app framework

Install all with:
```bash
pip install -r requirements.txt
```

---

## Expected Results

- **MLP Baseline:** ~60-70% accuracy
- **CNN Model:** ~75-85% accuracy
- **CNN-LSTM Model:** ~80-90% accuracy

Results may vary based on hyperparameters, data split, and augmentation techniques.

---

## Resources

- **TESS Dataset:** [Toronto Emotional Speech Set on Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- **Librosa Documentation:** https://librosa.org/
- **TensorFlow/Keras:** https://www.tensorflow.org/
- **Streamlit:** https://streamlit.io/

---

## Author

**Zaynab Shahid, Rameen Arshad and Laiba Riaz** 
2026

---

## License

This project is provided as-is for educational purposes.

---

## Contributing

For improvements or bug reports, please open an issue or submit a pull request.
