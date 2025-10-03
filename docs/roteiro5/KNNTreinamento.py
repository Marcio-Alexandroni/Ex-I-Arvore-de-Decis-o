# docs/roteiro5/KNNTreinamento.py
import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from pathlib import Path
import json, joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ---------------------------
# Caminhos e saída
# ---------------------------
DATA_PATH = Path("docs/data/TSLA_ready.csv")
OUT_DIR   = Path("docs/roteiro5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Carregar e preparar dados
# ---------------------------
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)

# Target: direção do dia seguinte
df["Target"] = (df["Change"].shift(-1) > 0).astype(int)

# Rolling z-score (sem vazamento): usa somente passado
win = 60
for col, new_col in [("Change", "Z_Change_roll"), ("Volume", "Z_Volume_roll")]:
    roll_mean = df[col].rolling(win, min_periods=win).mean()
    roll_std  = df[col].rolling(win, min_periods=win).std()
    df[new_col] = (df[col] - roll_mean) / roll_std

# Remover linhas iniciais sem janela completa e o último por causa do shift(-1)
df = df.dropna(subset=["Z_Change_roll", "Z_Volume_roll"]).iloc[:-1].reset_index(drop=True)

# Feature set reduzido (evita duplicidade)
feature_cols = ["Z_Change_roll", "Z_Volume_roll"]
X = df[feature_cols].copy()
y = df["Target"].copy()

# ---------------------------
# Split temporal 80/20 (sem embaralhar)
# ---------------------------
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ---------------------------
# Pipeline
# ---------------------------
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# ---------------------------
# Curva de validação (k) — melhor score por k dentre métricas/pesos
# ---------------------------
tscv = TimeSeriesSplit(n_splits=5)
k_values = list(range(3, 32, 2))
cv_best_means = []

for k in k_values:
    best_for_k = -np.inf
    for metric in ["euclidean", "manhattan"]:
        for weights in ["uniform", "distance"]:
            pipe.set_params(knn__n_neighbors=k, knn__metric=metric, knn__weights=weights)
            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                Xt_tr, Xt_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                yt_tr, yt_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                pipe.fit(Xt_tr, yt_tr)
                y_val_pred = pipe.predict(Xt_val)
                scores.append(balanced_accuracy_score(yt_val, y_val_pred))
            mean_score = float(np.mean(scores))
            if mean_score > best_for_k:
                best_for_k = mean_score
    cv_best_means.append(best_for_k)

plt.figure()
plt.plot(k_values, cv_best_means, marker="o")
plt.xlabel("k (n_neighbors)")
plt.ylabel("Balanced Accuracy (CV temporal)")
plt.title("Curva de Validação — KNN (TimeSeriesSplit)")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "knn_validation_curve.png", dpi=220, transparent=True)
plt.close()

# ---------------------------
# GridSearch com métricas e pesos
# ---------------------------
param_grid = {
    "knn__n_neighbors": k_values,
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"],
}
grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=tscv,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True,
)
grid.fit(X_train, y_train)
clf = grid.best_estimator_

# ---------------------------
# Avaliação no hold-out (20% final)
# ---------------------------
y_pred = clf.predict(X_test)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Balanced_Accuracy": balanced_accuracy_score(y_test, y_pred),
    "Precision (1=Alta)": precision_score(y_test, y_pred, zero_division=0),
    "Recall (1=Alta)": recall_score(y_test, y_pred, zero_division=0),
    "F1 (1=Alta)": f1_score(y_test, y_pred, zero_division=0),
    "F1 ponderado": f1_score(y_test, y_pred, average="weighted", zero_division=0),
}
pd.DataFrame([metrics]).to_csv(OUT_DIR / "knn_tsla_metrics.csv", index=False)

cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, index=["Real: 0", "Real: 1"], columns=["Pred: 0", "Pred: 1"]).to_csv(
    OUT_DIR / "knn_tsla_confusion_matrix.csv", index=False
)

plt.figure()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Matriz de Confusão — KNN (Teste)")
plt.tight_layout()
plt.savefig(OUT_DIR / "knn_tsla_confusion_matrix.png", dpi=220, transparent=True)
plt.close()

# ---------------------------
# Persistência
# ---------------------------
(OUT_DIR / "knn_best_params.json").write_text(json.dumps(grid.best_params_), encoding="utf-8")
joblib.dump(clf, OUT_DIR / "knn_tsla.joblib")

# Limpeza final de figuras/backends
import gc
plt.close("all")
gc.collect()

print("Treinamento concluído. Artefatos salvos em:", OUT_DIR.resolve())
print("Melhores parâmetros:", grid.best_params_)
