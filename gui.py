"""
gui.py - Tkinter GUI for Parkinson's disease detection from voice recordings.

Requires: parselmouth (pip install praat-parselmouth)
"""

import os
import pickle
from tkinter import Tk, Button, Label, Menu, filedialog, messagebox

from features import extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DISCLAIMER = (
    "⚠ DISCLAIMER: This tool is for educational and research purposes only.\n"
    "It is NOT a medical diagnostic tool. Do not use it to make health\n"
    "decisions. Always consult a qualified healthcare professional."
)


def run_prediction(wav_path):
    """Extract features and return prediction (0=healthy, 1=parkinsons)."""
    X = extract_features(wav_path)

    model_path = os.path.join(BASE_DIR, 'svmclassifier.pkl')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            "Model not found. Run 'python train.py' first."
        )

    with open(model_path, 'rb') as f:
        saved = pickle.load(f)

    X_scaled = saved['scaler'].transform(X)
    return saved['model'].predict(X_scaled)[0]


class App:
    """Tkinter GUI — file picker + SVM prediction with medical disclaimer."""
    def __init__(self, root):
        self.root = root
        self.root.title("Parkinson's Detection")
        self.root.geometry('400x200')
        self.filename = None

        # Menu
        menubar = Menu(root)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.browse_file)
        file_menu.add_command(label="Exit", command=root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(
            label="Help",
            command=lambda: messagebox.showinfo(
                "Help",
                "1. File → Open to select a .wav voice recording\n"
                "2. Click Detect to analyze\n\n" + DISCLAIMER,
            ),
        )
        menubar.add_cascade(label="About", menu=help_menu)
        root.config(menu=menubar)

        # Disclaimer label
        Label(root, text=DISCLAIMER, fg="red", justify="left",
              wraplength=380, font=("TkDefaultFont", 9)).pack(pady=(10, 5))

        Button(root, text="Detect", command=self.detect).pack(pady=10)

    def browse_file(self):
        self.filename = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )

    def detect(self):
        if not self.filename:
            messagebox.showwarning("No File", "Please open a .wav file first (File → Open)")
            return
        try:
            result = run_prediction(self.filename)
            if result == 1:
                messagebox.showinfo("Result", "Parkinson's Disease Detected\n\n" + DISCLAIMER)
            else:
                messagebox.showinfo("Result", "Healthy — No Parkinson's Detected\n\n" + DISCLAIMER)
        except ImportError as e:
            messagebox.showerror("Missing Dependency", str(e))
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{e}")


if __name__ == '__main__':
    root = Tk()
    App(root)
    root.mainloop()
