import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

DATA_PATH = Path("docs/data/TSLA_ready.csv")
OUT_DIR   = Path("docs/roteiro5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)

df["Target"] = (df["Change"].shift(-1) > 0).astype(int)
df = df.iloc[:-1].reset_index(drop=True)

feature_cols = ["Volume", "N-Volume", "Z-Volume", "Change", "N-Change", "Z-Change"]
X, y = df[feature_cols], df["Target"]

split_idx = int(len(df) * 0.8)
X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

tscv = TimeSeriesSplit(n_splits=5)
k_values = list(range(1, 32, 2))
cv_means = []
for k in k_values:
    pipe.set_params(knn__n_neighbors=k, knn__weights="uniform")
    scores = cross_val_score(pipe, X_train, y_train, cv=tscv, scoring="accuracy", n_jobs=-1)
    cv_means.append(scores.mean())

plt.figure()
plt.plot(k_values, cv_means, marker="o")
plt.xlabel("k (n_neighbors)")
plt.ylabel("Accuracy (CV temporal)")
plt.title("Curva de Validação — KNN (TimeSeriesSplit)")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "knn_validation_curve.png", dpi=220, transparent=True)
plt.close()
print("OK ->", OUT_DIR / "knn_validation_curve.png")
