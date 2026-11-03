# Explainable AI Disease Risk Prediction

Educational/research prototype that predicts disease risk from health-related features and explains individual predictions using SHAP.

## Important
This project is for education/research only. It is not a medical diagnostic system and must not be used to make clinical decisions.

## Features
- Synthetic, reproducible dataset generation
- Exploratory data analysis
- Preprocessing with scikit-learn
- Logistic Regression and Random Forest models
- Model comparison
- Accuracy, precision, recall, F1 and ROC-AUC
- Confusion matrix and ROC curve
- SHAP global and local explanations
- Streamlit web application
- Saved model and preprocessing pipeline

## Project structure

```text
explainable-ai-disease-risk/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── generate_data.py
│   └── disease_risk.csv              # generated after Step 1
├── models/
│   ├── model.joblib                  # generated after training
│   └── preprocessor.joblib           # generated after training
├── notebooks/
│   └── README.md
├── reports/
│   └── README.md
└── src/
    ├── __init__.py
    ├── config.py
    ├── preprocess.py
    ├── train_model.py
    ├── evaluate_model.py
    └── explain_model.py
```

## Windows setup

Open PowerShell in the project folder:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\activate
```

## Step 1: Generate data

```powershell
python data/generate_data.py
```

## Step 2: Train models

```powershell
python src/train_model.py
```

## Step 3: Evaluate the model

```powershell
python src/evaluate_model.py
```

## Step 4: Generate SHAP explanation

```powershell
python src/explain_model.py
```

## Step 5: Start the web app

```powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Expected workflow

```text
Input data
   ↓
Preprocessing
   ↓
Machine Learning model
   ↓
Disease risk probability
   ↓
SHAP explanation
   ↓
User-friendly result
```

## Dataset
The included generator creates synthetic records. This avoids presenting fabricated clinical observations as real patient data. For a real academic study, replace it with a properly licensed, documented dataset and update the methodology accordingly.
