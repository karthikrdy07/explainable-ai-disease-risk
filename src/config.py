from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

DATA_FILE = DATA_DIR / "disease_risk.csv"
MODEL_FILE = MODELS_DIR / "model.joblib"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.joblib"

TARGET = "disease_risk"

NUMERIC_FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "glucose",
    "cholesterol",
    "physical_activity_hours",
    "sleep_hours",
]

CATEGORICAL_FEATURES = [
    "gender",
    "smoking",
    "family_history",
]
