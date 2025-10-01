import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
import joblib

DATA_PATH  = Path("docs/data/TSLA_ready.csv")
OUT_DIR    = Path("docs/roteiro5")
MODEL_PATH = OUT_DIR / "knn_tsla.joblib"

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)

df["Target"] = (df["Change"].shift(-1) > 0).astype(int)
df = df.iloc[:-1].reset_index(drop=True)

feature_cols = ["Volume", "N-Volume", "Z-Volume", "Change", "N-Change", "Z-Change"]
X, y = df[feature_cols], df["Target"]

split_idx = int(len(df) * 0.8)
X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

clf = joblib.load(MODEL_PATH)

y_pred = clf.predict(X_test)

plt.figure()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Matriz de Confusão — KNN (Teste)")
plt.tight_layout()
plt.savefig(OUT_DIR / "knn_tsla_confusion_matrix.png", dpi=220, transparent=True)
plt.close()
print("OK ->", OUT_DIR / "knn_tsla_confusion_matrix.png")
