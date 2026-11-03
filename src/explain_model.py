import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.config import (
    DATA_FILE,
    MODEL_FILE,
    PREPROCESSOR_FILE,
    FEATURE_NAMES_FILE,
    REPORTS_DIR,
    TARGET,
)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)
X = df.drop(columns=[TARGET])

preprocessor = joblib.load(PREPROCESSOR_FILE)
model = joblib.load(MODEL_FILE)
feature_names = joblib.load(FEATURE_NAMES_FILE)

X_transformed = preprocessor.transform(X)

# TreeExplainer is ideal when Random Forest is selected.
# For Logistic Regression, SHAP's LinearExplainer is used.
if hasattr(model, "estimators_"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
else:
    explainer = shap.LinearExplainer(model, X_transformed)
    shap_values = explainer.shap_values(X_transformed)

if isinstance(shap_values, list):
    values = shap_values[1]
else:
    values = shap_values

shap.summary_plot(
    values,
    X_transformed,
    feature_names=feature_names,
    show=False,
)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved SHAP global explanation to reports/shap_summary.png")
