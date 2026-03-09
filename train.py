"""
train.py - Train an SVM classifier for Parkinson's disease detection.

Loads voice feature data from final2.csv (no header, 29 columns),
selects 10 key acoustic features, trains an RBF SVM with StandardScaler,
and saves model + scaler to svmclassifier.pkl.

Note: The original project used final1.csv (25 columns) which is no longer
available. final2.csv has 29 columns with the Parkinson label at index 28.
Accuracy is lower (~65%) compared to the original dataset (~85-90%).
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset (no header row; 29 columns, Parkinson label at index 28)
dataset = pd.read_csv('final2.csv', header=None)

# 10 PSO-selected features by column index:
# 5=Jitter(ddp), 23=SD of period, 22=Mean period, 13=NTH,
# 1=Jitter(local), 7=Shimmer(local,dB), 2=Jitter(local,abs),
# 4=Jitter(ppq5), 8=Shimmer(apq3), 3=Jitter(rap)
FEATURE_COLS = [5, 23, 22, 13, 1, 7, 2, 4, 8, 3]
LABEL_COL = 28

X = dataset.iloc[:, FEATURE_COLS].values
y = dataset.iloc[:, LABEL_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = SVC(kernel='rbf', C=10, random_state=0)
classifier.fit(X_train, y_train)

scores = cross_val_score(classifier, X_train, y_train, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

y_pred = classifier.predict(X_test)
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Healthy', 'Parkinsons'])}")

# Save model, scaler, and feature config
with open('svmclassifier.pkl', 'wb') as f:
    pickle.dump({
        'model': classifier,
        'scaler': scaler,
        'feature_cols': FEATURE_COLS,
    }, f)

print("Model and scaler saved to svmclassifier.pkl")
