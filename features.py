"""
features.py - Shared feature extraction from PRAAT voice reports.

Parses a .wav file using parselmouth, extracts acoustic features,
and returns the selected feature vector for the SVM classifier.
"""

import os
import re
import numpy as np

# 10 PSO-selected feature columns (0-based CSV column indices, used by both
# train.py via pandas iloc and features.py for extraction).
# Column 0 = Subject ID, columns 1-27 = acoustic features, column 28 = label.
# 5=Jitter(ddp), 23=SD of period, 22=Mean period, 13=NTH,
# 1=Jitter(local), 7=Shimmer(local,dB), 2=Jitter(local,abs),
# 4=Jitter(ppq5), 8=Shimmer(apq3), 3=Jitter(rap)
FEATURE_COLS = [5, 23, 22, 13, 1, 7, 2, 4, 8, 3]

# Minimum number of numeric values expected from a valid PRAAT voice report
_MIN_REPORT_NUMBERS = 40


def extract_features(wav_path: str) -> np.ndarray:
    """
    Extract acoustic features from a .wav file using PRAAT.

    Args:
        wav_path: Path to a .wav file containing sustained vowel phonation.

    Returns:
        numpy array of shape (1, 10) with the selected features.

    Raises:
        FileNotFoundError: If wav_path does not exist.
        ValueError: If the audio cannot be analyzed (too short, no voice detected, etc.)
        ImportError: If parselmouth is not installed.
    """
    try:
        import parselmouth
    except ImportError:
        raise ImportError(
            "parselmouth is required for feature extraction. "
            "Install it with: pip install praat-parselmouth"
        )

    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"Not a valid file: {wav_path}")

    sound = parselmouth.Sound(wav_path)

    # Validate audio duration (PRAAT needs at least ~0.1s for pitch analysis)
    if sound.duration < 0.1:
        raise ValueError(
            f"Audio too short ({sound.duration:.3f}s). "
            "Need at least 0.1 seconds of sustained vowel phonation."
        )

    pitch = sound.to_pitch()
    pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
    voice_report = parselmouth.praat.call(
        [sound, pitch, pulses],
        "Voice report", 0.0, 0.0, 75, 600, 1.3, 1.6, 0.03, 0.45,
    )

    # Parse all numeric values from the PRAAT voice report.
    # Expected order (40 numbers):
    #   time[0-2], pitch[3-7], pulses[8-9], period[10-13],
    #   voicing[14-20], jitter[21-27], shimmer[28-36], harmonicity[37-39]
    nums = re.findall(r'-?\d+\.?\d*', voice_report)

    if len(nums) < _MIN_REPORT_NUMBERS:
        raise ValueError(
            f"PRAAT voice report returned {len(nums)} values (expected >= {_MIN_REPORT_NUMBERS}). "
            "The recording may lack voiced content. Use a sustained vowel ('a' or 'o')."
        )

    # Map PRAAT report numbers to the 23 acoustic features (matching CSV columns 1-23)
    all_features = [
        float(nums[21]),                              #  1: Jitter (local) %
        float(nums[22] + 'E' + nums[23]),             #  2: Jitter (local, absolute) seconds
        float(nums[24]),                              #  3: Jitter (rap) %
        float(nums[26]),                              #  4: Jitter (ppq5) %
        float(nums[27]),                              #  5: Jitter (ddp) %
        float(nums[28]),                              #  6: Shimmer (local) %
        float(nums[29]),                              #  7: Shimmer (local, dB)
        float(nums[31]),                              #  8: Shimmer (apq3) %
        float(nums[33]),                              #  9: Shimmer (apq5) %
        float(nums[35]),                              # 10: Shimmer (apq11) %
        float(nums[36]),                              # 11: Shimmer (dda) %
        float(nums[37]),                              # 12: AC (autocorrelation)
        float(nums[38]),                              # 13: NTH (noise-to-harmonics)
        float(nums[39]),                              # 14: HTN (harmonics-to-noise) dB
        float(nums[3]),                               # 15: Median pitch Hz
        float(nums[4]),                               # 16: Mean pitch Hz
        float(nums[5]),                               # 17: SD Hz
        float(nums[6]),                               # 18: Minimum pitch Hz
        float(nums[7]),                               # 19: Maximum pitch Hz
        float(nums[8]),                               # 20: Number of pulses
        float(nums[9]),                               # 21: Number of periods
        float(nums[10] + 'E' + nums[11]),             # 22: Mean period seconds
        float(nums[12] + 'E' + nums[13]),             # 23: SD of period seconds
    ]

    # Select the 10 training features.
    # all_features[0] corresponds to CSV column 1 (column 0 is Subject ID),
    # so subtract 1 to convert CSV column index → all_features index.
    selected = [all_features[c - 1] for c in FEATURE_COLS]
    return np.array(selected).reshape(1, -1)
