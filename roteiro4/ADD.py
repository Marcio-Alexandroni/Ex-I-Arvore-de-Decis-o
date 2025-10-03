import pandas as pd
import joblib
from pathlib import Path
import numpy as np

DATA_PATH = Path("docs/data/TSLA_ready.csv")
MODEL_PATH = Path("docs/roteiro4/tree_tsla.joblib")

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)

feature_cols = ["Volume", "N-Volume", "Z-Volume", "Change", "N-Change", "Z-Change"]
X_all = df[feature_cols]

clf = joblib.load(MODEL_PATH)

print("Importância das variáveis:\n", pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False))

x_last = X_all.iloc[[-2]]
pred = clf.predict(x_last)[0]
proba = clf.predict_proba(x_last)[0][int(pred)] if hasattr(clf, "predict_proba") else np.nan

print("\nDemo previsão (penúltimo registro):")
print("Classe prevista:", int(pred), "| Probabilidade:", float(proba))
