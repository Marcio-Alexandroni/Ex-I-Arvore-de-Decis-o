import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
from pathlib import Path
import json

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
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

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

param_grid = {"knn__n_neighbors": k_values, "knn__weights": ["uniform","distance"]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=tscv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
clf = grid.best_estimator_

y_pred = clf.predict(X_test)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision (1=Alta)": precision_score(y_test, y_pred, zero_division=0),
    "Recall (1=Alta)": recall_score(y_test, y_pred, zero_division=0),
    "F1 (1=Alta)": f1_score(y_test, y_pred, zero_division=0),
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

(OUT_DIR / "knn_best_params.json").write_text(json.dumps(grid.best_params_), encoding="utf-8")
joblib.dump(clf, OUT_DIR / "knn_tsla.joblib")

print("Treinamento concluído. Artefatos salvos em:", OUT_DIR)
