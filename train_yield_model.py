from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "yield_df.csv"
OUTPUT = ROOT / "models" / "yield_model.pkl"

if not DATA.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA}\n"
        "Place yield_df.csv inside the data folder."
    )

df = pd.read_csv(DATA)
df.columns = df.columns.str.strip()

# Match the dataset's actual columns.
required = [
    "Area",
    "Item",
    "Year",
    "hg/ha_yield",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]

missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# Clean numeric fields and target.
numeric = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "hg/ha_yield",
]

for column in numeric:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Area", "Item", "hg/ha_yield"])
df = df[df["hg/ha_yield"] >= 0].copy()

features = [
    "Area",
    "Item",
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]

X = df[features].copy()
y = df["hg/ha_yield"].copy()

categorical = ["Area", "Item"]
numerical = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ),
        (
            "numerical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numerical,
        ),
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Valid rows: {len(df)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print("Training Random Forest yield model...")

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Dataset target is hg/ha. Divide by 10,000 to get tonnes/ha.
mae = mean_absolute_error(y_test, predictions) / 10000
rmse = np.sqrt(mean_squared_error(y_test, predictions)) / 10000
r2 = r2_score(y_test, predictions)

print("\nModel evaluation:")
print(f"MAE:  {mae:.4f} tonnes/ha")
print(f"RMSE: {rmse:.4f} tonnes/ha")
print(f"R2:   {r2:.4f}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": features,
        "target_unit": "hg/ha",
        "display_unit": "tonnes/ha",
    },
    OUTPUT,
)

print(f"\nSaved yield model to: {OUTPUT}")
