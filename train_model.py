import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "housing.csv")
MODEL_DIR = os.path.join(BASE, "model")
TARGET = "median_house_value"
CATEGORICAL = ["ocean_proximity"]

if not os.path.exists(DATA_PATH):
    raise SystemExit("housing.csv not found. Put it in the same folder as train_model.py.")

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH).dropna()
X = df.drop(TARGET, axis=1)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
    remainder="passthrough",
)
model = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestRegressor(
        n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1
    )),
])

print("Training... (about 1 minute)")
model.fit(X_train, y_train)

pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
r2 = float(r2_score(y_test, pred))
print(f"Test RMSE: {rmse:,.0f}   R2: {r2:.3f}")

joblib.dump(model, os.path.join(MODEL_DIR, "house_model.joblib"), compress=3)
with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
    json.dump(
        {
            "rmse": round(rmse),
            "r2": round(r2, 3),
            "n_rows": int(len(df)),
            "columns": list(X.columns),
        },
        f,
        indent=2,
    )
print("Saved model/house_model.joblib and model/meta.json")