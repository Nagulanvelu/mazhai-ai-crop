from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "crop_recommendation.csv"
OUT = ROOT / "models" / "crop_model.pkl"

if not DATA.exists():
    raise FileNotFoundError(
        f"Missing {DATA}. Download a crop recommendation CSV and place it there."
    )

df = pd.read_csv(DATA)
df.columns = df.columns.str.strip().str.lower()

target = next((x for x in ["label", "crop", "crop_name"] if x in df.columns), None)
if target is None:
    raise ValueError(f"Could not find crop label column. Columns: {list(df.columns)}")

aliases = {
    "N": ["n", "nitrogen"],
    "P": ["p", "phosphorus"],
    "K": ["k", "potassium"],
    "temperature": ["temperature", "temp"],
    "humidity": ["humidity"],
    "ph": ["ph", "soil_ph"],
    "rainfall": ["rainfall", "rain"]
}

rename = {}
for standard, options in aliases.items():
    found = next((x for x in options if x in df.columns), None)
    if found is None:
        raise ValueError(f"Missing {standard}. CSV columns: {list(df.columns)}")
    rename[found] = standard

df = df.rename(columns=rename)
features = list(aliases.keys())
X = df[features].apply(pd.to_numeric, errors="coerce")
y = df[target].astype(str)
valid = X.notna().all(axis=1) & y.notna()
X, y = X.loc[valid], y.loc[valid]

if len(X) < 10:
    raise ValueError("Not enough valid rows to train.")

stratify = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(n_estimators=200, random_state=42)
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Test accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, zero_division=0))

OUT.parent.mkdir(exist_ok=True)
joblib.dump(model, OUT)
print("Saved model:", OUT)
