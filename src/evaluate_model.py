import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split

from src.config import DATA_FILE, MODEL_FILE, PREPROCESSOR_FILE, REPORTS_DIR, TARGET
from src.preprocess import split_features_target

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)
X, y = split_features_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

preprocessor = joblib.load(PREPROCESSOR_FILE)
model = joblib.load(MODEL_FILE)

X_test_t = preprocessor.transform(X_test)
pred = model.predict(X_test_t)
prob = model.predict_proba(X_test_t)[:, 1]

print("Classification Report")
print(classification_report(y_test, pred, zero_division=0))

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=150)
plt.close()

print("Saved evaluation plots to reports/")
