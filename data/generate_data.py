from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 2500

rng = np.random.default_rng(RANDOM_SEED)

age = rng.integers(18, 81, N_SAMPLES)
bmi = np.clip(rng.normal(27, 5, N_SAMPLES), 16, 45)
systolic_bp = np.clip(rng.normal(128, 18, N_SAMPLES), 90, 200)
glucose = np.clip(rng.normal(105, 30, N_SAMPLES), 60, 250)
cholesterol = np.clip(rng.normal(195, 35, N_SAMPLES), 100, 350)
physical_activity = np.clip(rng.normal(3.5, 2, N_SAMPLES), 0, 10)
sleep_hours = np.clip(rng.normal(7, 1.2, N_SAMPLES), 3.5, 10)
smoking = rng.choice(["No", "Yes"], size=N_SAMPLES, p=[0.72, 0.28])
family_history = rng.choice(["No", "Yes"], size=N_SAMPLES, p=[0.65, 0.35])
gender = rng.choice(["Female", "Male"], size=N_SAMPLES, p=[0.48, 0.52])

# Synthetic risk mechanism for demonstration only.
risk_score = (
    0.035 * (age - 40)
    + 0.11 * (bmi - 25)
    + 0.035 * (systolic_bp - 120)
    + 0.028 * (glucose - 100)
    + 0.012 * (cholesterol - 190)
    - 0.16 * physical_activity
    - 0.10 * (sleep_hours - 7)
    + 0.95 * (smoking == "Yes")
    + 0.70 * (family_history == "Yes")
    + rng.normal(0, 1.0, N_SAMPLES)
)

probability = 1 / (1 + np.exp(-risk_score / 4.0))
disease_risk = rng.binomial(1, probability)

df = pd.DataFrame({
    "age": age,
    "gender": gender,
    "bmi": np.round(bmi, 1),
    "systolic_bp": np.round(systolic_bp, 0).astype(int),
    "glucose": np.round(glucose, 0).astype(int),
    "cholesterol": np.round(cholesterol, 0).astype(int),
    "physical_activity_hours": np.round(physical_activity, 1),
    "sleep_hours": np.round(sleep_hours, 1),
    "smoking": smoking,
    "family_history": family_history,
    "disease_risk": disease_risk,
})

# Add a small amount of missing data to exercise the preprocessing pipeline.
for column in ["bmi", "glucose", "cholesterol"]:
    idx = rng.choice(df.index, size=int(0.01 * N_SAMPLES), replace=False)
    df.loc[idx, column] = np.nan

output = Path(__file__).resolve().parent / "disease_risk.csv"
df.to_csv(output, index=False)

print(f"Generated {len(df)} rows")
print(f"Saved to: {output}")
print(df.head())
