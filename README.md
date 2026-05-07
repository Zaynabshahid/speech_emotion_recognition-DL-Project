# Speech Emotion Recognition
## CS-419 Deep Learning Semester Project

**Dataset:** TESS (Toronto Emotional Speech Set)  
**Task:** Multi-class audio emotion classification  
**Classes:** angry, disgust, fear, happy, neutral, pleasant surprise, sad

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
├── ui/
│   └── index.html                    # Interactive project presentation UI
└── README.md
```

## How to Run

### Step 1 - Install dependencies
```bash
pip install librosa numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

### Step 2 - Download dataset
Download TESS from Kaggle:
https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

Place the extracted folder as: `data/TESS Toronto emotional speech set data/`

### Step 3 - Run notebooks in order
Open each notebook in Google Colab or Jupyter and run top to bottom.

### Step 4 - Open the UI
Open `ui/index.html` in any browser to see the interactive project dashboard.

### Streamlit demo
1. Create and activate your virtual environment (Windows PowerShell example):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Start the app:
   ```powershell
   streamlit run app.py
   ```
4. If the app cannot find the dataset, update the dataset path in the sidebar to the extracted TESS folder.

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

## Models Compared
| Model | Input | Architecture |
|-------|-------|-------------|
| MLP Baseline | Raw MFCCs (40 coefficients) | 3-layer dense network |
| CNN | Log-mel spectrogram (128x128) | 4 conv blocks + dense |
| CNN-LSTM | Log-mel spectrogram (128x128) | CNN encoder + BiLSTM |
