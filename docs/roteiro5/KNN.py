import pandas as pd
import joblib
from pathlib import Path
import numpy as np

DATA_PATH = Path("docs/data/TSLA_ready.csv")
MODEL_PATH = Path("docs/roteiro5/knn_tsla.joblib")

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)

df["Target"] = (df["Change"].shift(-1) > 0).astype(int)
df = df.iloc[:-1].reset_index(drop=True)

feature_cols = ["Volume", "N-Volume", "Z-Volume", "Change", "N-Change", "Z-Change"]
X_all = df[feature_cols]

clf = joblib.load(MODEL_PATH)

x_last = X_all.iloc[[-2]]
pred = clf.predict(x_last)[0]
proba = clf.predict_proba(x_last)[0][int(pred)] if hasattr(clf, "predict_proba") else np.nan

knn = getattr(clf, "named_steps", {}).get("knn", None)
k = getattr(knn, "n_neighbors", None)
weights = getattr(knn, "weights", None)

print("Config do modelo KNN:", {"n_neighbors": k, "weights": weights})
print("\nDemo previsão (penúltimo registro):")
print("Classe prevista:", int(pred), "| Probabilidade:", float(proba))
