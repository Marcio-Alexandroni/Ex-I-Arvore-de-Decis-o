# ================================
# Visualizações KNN — Projeto TSLA
# Curva de Validação (k × accuracy)
# Matriz de Confusão (Teste)
# ================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score

# ---------- Config ----------
URL = "https://raw.githubusercontent.com/Marcio-Alexandroni/Ex-I-Arvore-de-Decis-o/refs/heads/main/docs/data/TSLA_ready.csv"
SAVE_DIR = "docs/roteiroKNN"
os.makedirs(SAVE_DIR, exist_ok=True)

RANDOM_STATE = 42
K_VALUES = list(range(1, 32, 2))  # 1..31 ímpares
WEIGHTS = ["uniform", "distance"]
N_SPLITS = 5                      # TimeSeriesSplit

# Se souber o nome da coluna alvo, defina aqui (ex.: "Target")
TARGET = None  # ex.: TARGET = "Target"

# ---------- Load ----------
df = pd.read_csv(URL)

# Inferência simples do alvo (ou fixe manualmente em TARGET acima)
if TARGET is None:
    ignore_like = {'date','time','datetime','timestamp','index','id'}
    cands = [c for c in df.columns
             if c.lower() not in ignore_like and 2 <= df[c].nunique(dropna=True) <= 6]
    pref = ['target','label','classe','class','y','signal','updown']
    lower = {c.lower(): c for c in cands}
    TARGET = next((lower[p] for p in pref if p in lower), cands[0] if cands else None)
    if TARGET is None:
        raise RuntimeError("Defina TARGET manualmente: não encontrei coluna candidata.")

y = df[TARGET].copy()
X = df.drop(columns=[TARGET]).select_dtypes(include=[np.number]).copy()

# Split temporal 80/20 (sem embaralhar)
n = len(df)
cut = int(0.8 * n)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

# Pipeline: imputação + escala + KNN
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# -------------------------------
# 1) Curva de validação (k × accuracy) — CV temporal
# -------------------------------
cv_means = []
for k in K_VALUES:
    pipe.set_params(knn__n_neighbors=k, knn__weights='uniform')
    scores = cross_val_score(pipe, X_train, y_train, cv=tscv, scoring="accuracy", n_jobs=-1)
    cv_means.append(scores.mean())

plt.figure()
plt.plot(K_VALUES, cv_means, marker='o')
plt.xlabel("k (n_neighbors)")
plt.ylabel("Accuracy (CV temporal)")
plt.title("Curva de Validação — KNN (TimeSeriesSplit)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/knn_validation_curve.png", dpi=160, transparent=True)
plt.show()

# -------------------------------
# 2) Matriz de confusão — melhor KNN no conjunto de teste
#    (GridSearch em k × weights usando TimeSeriesSplit)
# -------------------------------
param_grid = {"knn__n_neighbors": K_VALUES, "knn__weights": WEIGHTS}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=tscv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

best = grid.best_estimator_
y_pred = best.predict(X_test)

print("Melhores parâmetros:", grid.best_params_)
print("Accuracy teste:", round(accuracy_score(y_test, y_pred), 4))
print("F1 ponderado teste:", round(f1_score(y_test, y_pred, average='weighted'), 4))

plt.figure()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Matriz de Confusão — KNN (Teste)")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/knn_tsla_confusion_matrix.png", dpi=160, transparent=True)
plt.show()
