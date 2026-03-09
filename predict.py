"""
predict.py - Predict Parkinson's disease from a .wav voice recording.

Usage: python predict.py <path_to_wav_file>
"""

import sys
import os
import pickle

from features import extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'svmclassifier.pkl')


def predict(wav_path: str) -> str:
    """Run prediction on a .wav file and return the result string."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run 'python train.py' first to generate svmclassifier.pkl"
        )

    X = extract_features(wav_path)

    with open(MODEL_PATH, 'rb') as f:
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
