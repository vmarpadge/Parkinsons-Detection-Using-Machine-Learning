# Parkinson's Disease Detection Using Machine Learning

> ⚠️ **DISCLAIMER**: This project is for **educational and research purposes only**. It is NOT a medical diagnostic tool. Do not use it to make health decisions. Always consult a qualified healthcare professional for medical advice.

Detects Parkinson's disease from voice recordings using an SVM (Support Vector Machine) classifier. The system analyzes phonation features — the sounds produced when pronouncing sustained vowels — since phonation is the most affected component of speech in Parkinson's patients.

## How It Works

1. **Feature Extraction**: Voice recordings (`.wav`) are processed using [PRAAT](https://www.fon.hum.uva.nl/praat/) (via [`parselmouth`](https://github.com/YannickJadworski/Parselmouth)) to extract acoustic features: jitter, shimmer, NHR, HNR, pitch statistics, pulse counts, and period measurements.
2. **Training**: An SVM classifier (RBF kernel, C=10) is trained on the UCI voice dataset (1040 recordings from 56 subjects). Feature scaling is applied via `StandardScaler`, and the scaler is saved alongside the model.
3. **Prediction**: New voice samples are analyzed, features are extracted and scaled using the saved scaler, then classified as Parkinson's-positive or healthy.

## Project Structure

```
├── features.py           # Shared feature extraction module (PRAAT parsing + validation)
├── train.py              # Train the SVM model and save it with the scaler
├── predict.py            # CLI prediction from a .wav file
├── gui.py                # Tkinter GUI for file selection and prediction
├── final2.csv            # Training dataset (1040 samples, 29 columns)
├── requirements.txt      # Pinned Python dependencies
├── LICENSE               # MIT License
└── README.md
```

## Setup

```bash
# Python 3.10+ required
pip install -r requirements.txt
```

## Usage

### 1. Train the model

```bash
python train.py
```

Example output:
```
Cross-validation accuracy: 0.6466 (+/- 0.0169)

Confusion Matrix:
[[60 47]
 [28 73]]

              precision    recall  f1-score   support

     Healthy       0.68      0.56      0.62       107
  Parkinsons       0.61      0.72      0.66       101

    accuracy                           0.64       208

Model and scaler saved to svmclassifier.pkl
```

### 2. Predict from a .wav file (CLI)

```bash
python predict.py path/to/voice_recording.wav
```

```
Healthy — No Parkinson's Detected
```

### 3. Predict using the GUI

```bash
python gui.py
```

Use **File → Open** to select a `.wav` file containing a sustained vowel ('a' or 'o'), then click **Detect**.

## Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/301/) — Parkinson Speech Dataset with Multiple Types of Sound Recordings
- **License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Samples**: 1040 recordings from 56 subjects (sustained vowels 'a' and 'o', 3 times each)
- **Features**: 24 acoustic features including jitter variants, shimmer variants, autocorrelation, noise-to-harmonics ratio, pitch statistics, pulse/period measurements
- **Label**: Binary (1 = Parkinson's, 0 = Healthy) at column index 28

### Citation

If you use this dataset, please cite:

> Sakar, B.E., Isenkul, M.E., Sakar, C.O., Sertbas, A., Gurgen, F., Delil, S., Apaydin, H., Kursun, O. (2013).
> *Collection and Analysis of a Parkinson Speech Dataset with Multiple Types of Sound Recordings.*
> IEEE Journal of Biomedical and Health Informatics, 17(4), 828-834.

## Model Accuracy

The model achieves ~65% accuracy on `final2.csv`. The original project was built for a different dataset (`final1.csv`, 25 columns) which is no longer available and likely achieved higher accuracy. The 10 features were selected via Particle Swarm Optimization (PSO) for that original dataset.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.
