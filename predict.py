"""
predict.py - Predict Parkinson's disease from a .wav voice recording.

Extracts acoustic features using parselmouth (PRAAT), loads the trained
SVM model + scaler from svmclassifier.pkl, and outputs the prediction.

Usage: python predict.py <path_to_wav_file>
"""

import sys
import os
import pickle
import re
import numpy as np

FEATURE_COLS = [5, 23, 22, 13, 1, 7, 2, 4, 8, 3]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_features(wav_path):
    """Extract PRAAT voice features from a .wav file, return selected features as numpy array."""
    try:
        import parselmouth
    except ImportError:
        sys.exit("Error: parselmouth not installed. Run: pip install praat-parselmouth")

    sound = parselmouth.Sound(wav_path)
    pitch = sound.to_pitch()
    pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
    voice_report = parselmouth.praat.call(
        [sound, pitch, pulses], "Voice report", 0.0, 0.0, 75, 600, 1.3, 1.6, 0.03, 0.45
    )

    # Parse all numbers from PRAAT voice report
    # Order: time[0-2], pitch[3-7], pulses[8-9], period[10-13],
    # voicing[14-20], jitter[21-27], shimmer[28-36], harmonicity[37-39]
    n = re.findall(r'-?\d+\.?\d*', voice_report)

    # Build all 24 feature values (indices 0-23 matching test.csv columns 1-24)
    all_features = [
        float(n[21]),                          # 1: Jitter (local)
        float(n[22] + 'E' + n[23]),            # 2: Jitter (local, absolute)
        float(n[24]),                          # 3: Jitter (rap)
        float(n[26]),                          # 4: Jitter (ppq5)
        float(n[27]),                          # 5: Jitter (ddp)
        float(n[28]),                          # 6: Shimmer (local)
        float(n[29]),                          # 7: Shimmer (local, dB)
        float(n[31]),                          # 8: Shimmer (apq3)
        float(n[33]),                          # 9: Shimmer (apq5)
        float(n[35]),                          # 10: Shimmer (apq11)
        float(n[36]),                          # 11: Shimmer (dda)
        float(n[37]),                          # 12: AC
        float(n[38]),                          # 13: NTH
        float(n[39]),                          # 14: HTN
        float(n[3]),                           # 15: Median pitch
        float(n[4]),                           # 16: Mean pitch
        float(n[5]),                           # 17: SD
        float(n[6]),                           # 18: Minimum pitch
        float(n[7]),                           # 19: Maximum Pitch
        float(n[8]),                           # 20: Number of pulses
        float(n[9]),                           # 21: Number of periods
        float(n[10] + 'E' + n[11]),            # 22: Mean period
        float(n[12] + 'E' + n[13]),            # 23: SD of period
    ]

    # Select the 10 training features (indices are 1-based CSV columns,
    # subtract 1 since all_features is 0-based)
    selected = [all_features[c - 1] for c in FEATURE_COLS]
    return np.array(selected).reshape(1, -1)


def predict(wav_path):
    """Run prediction on a .wav file and return the result string."""
    X = extract_features(wav_path)

    with open(os.path.join(BASE_DIR, 'svmclassifier.pkl'), 'rb') as f:
        saved = pickle.load(f)

    X_scaled = saved['scaler'].transform(X)
    prediction = saved['model'].predict(X_scaled)

    if prediction[0] == 1:
        return "Parkinson's Disease Detected"
    return "Healthy — No Parkinson's Detected"


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_wav_file>")
        sys.exit(1)

    result = predict(sys.argv[1])
    print(result)
