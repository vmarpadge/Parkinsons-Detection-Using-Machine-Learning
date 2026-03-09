"""
train.py - Train an SVM classifier for Parkinson's disease detection.

Loads voice feature data from final2.csv (no header, 29 columns),
selects 10 key acoustic features, trains an RBF SVM with StandardScaler,
and saves model + scaler to svmclassifier.pkl.
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

from features import FEATURE_COLS

LABEL_COL = 28  # Binary label: 1 = Parkinson's, 0 = Healthy


def train():
    """Train the SVM model and save it with the scaler."""
    dataset = pd.read_csv('final2.csv', header=None)

    X = dataset.iloc[:, FEATURE_COLS].values
    y = dataset.iloc[:, LABEL_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # RBF kernel with C=10 (tuned for this dataset; higher C = less regularization)
    classifier = SVC(kernel='rbf', C=10, random_state=0)
    classifier.fit(X_train, y_train)

    scores = cross_val_score(classifier, X_train, y_train, cv=5)
    print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    y_pred = classifier.predict(X_test)
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Healthy', 'Parkinsons'])}")

    with open('svmclassifier.pkl', 'wb') as f:
        pickle.dump({
            'model': classifier,
            'scaler': scaler,
            'feature_cols': FEATURE_COLS,
        }, f)

    print("Model and scaler saved to svmclassifier.pkl")


if __name__ == '__main__':
    train()
