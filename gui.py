"""
gui.py - Tkinter GUI for Parkinson's disease detection from voice recordings.

Requires: parselmouth (pip install praat-parselmouth)
"""

import os
import pickle
import re
import numpy as np
from tkinter import Tk, Button, Menu, filedialog, messagebox

FEATURE_COLS = [5, 23, 22, 13, 1, 7, 2, 4, 8, 3]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_and_predict(wav_path):
    """Extract features from wav_path and return prediction (0=healthy, 1=parkinsons)."""
    import parselmouth

    sound = parselmouth.Sound(wav_path)
    pitch = sound.to_pitch()
    pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
    voice_report = parselmouth.praat.call(
        [sound, pitch, pulses], "Voice report", 0.0, 0.0, 75, 600, 1.3, 1.6, 0.03, 0.45
    )

    n = re.findall(r'-?\d+\.?\d*', voice_report)
    all_features = [
        float(n[21]), float(n[22] + 'E' + n[23]), float(n[24]), float(n[26]),
        float(n[27]), float(n[28]), float(n[29]), float(n[31]),
        float(n[33]), float(n[35]), float(n[36]), float(n[37]),
        float(n[38]), float(n[39]), float(n[3]), float(n[4]),
        float(n[5]), float(n[6]), float(n[7]), float(n[8]),
        float(n[9]), float(n[10] + 'E' + n[11]), float(n[12] + 'E' + n[13]),
    ]
    X = np.array([all_features[c - 1] for c in FEATURE_COLS]).reshape(1, -1)

    with open(os.path.join(BASE_DIR, 'svmclassifier.pkl'), 'rb') as f:
        saved = pickle.load(f)

    X_scaled = saved['scaler'].transform(X)
    return saved['model'].predict(X_scaled)[0]


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Parkinson's Detection")
        self.root.geometry('300x300')
        self.filename = None

        menubar = Menu(root)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.browse_file)
        file_menu.add_command(label="Exit", command=root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(
            label="Help",
            command=lambda: messagebox.showinfo(
                "Help", "1. File → Open to select a .wav voice recording\n"
                        "2. Click Detect to analyze"
            )
        )
        menubar.add_cascade(label="About", menu=help_menu)
        root.config(menu=menubar)

        Button(root, text="Detect", command=self.detect).pack(pady=20)

    def browse_file(self):
        self.filename = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )

    def detect(self):
        if not self.filename:
            messagebox.showwarning("No File", "Please open a .wav file first (File → Open)")
            return
        try:
            result = extract_and_predict(self.filename)
            if result == 1:
                messagebox.showinfo("Result", "Parkinson's Disease Detected")
            else:
                messagebox.showinfo("Result", "Healthy — No Parkinson's Detected")
        except ImportError:
            messagebox.showerror("Error", "parselmouth not installed.\nRun: pip install praat-parselmouth")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{e}")


if __name__ == '__main__':
    root = Tk()
    App(root)
    root.mainloop()
