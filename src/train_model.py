import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_FILE,
    MODELS_DIR,
    MODEL_FILE,
    PREPROCESSOR_FILE,
    FEATURE_NAMES_FILE,
    TARGET,
)
from src.preprocess import build_preprocessor, split_features_target, get_feature_names


def metrics(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train_t, y_train)
        results[name] = metrics(model, X_test_t, y_test)

    # Select by ROC-AUC, with recall as an important secondary medical-risk metric.
    best_name = max(
        results,
        key=lambda name: (results[name]["roc_auc"], results[name]["recall"])
    )
    best_model = models[best_name]

    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    joblib.dump(get_feature_names(preprocessor), FEATURE_NAMES_FILE)

    with open(MODELS_DIR / "model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nModel comparison:")
    for name, score in results.items():
        print(name)
        for metric_name, value in score.items():
            print(f"  {metric_name}: {value:.4f}")

    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved preprocessor: {PREPROCESSOR_FILE}")


if __name__ == "__main__":
    main()
