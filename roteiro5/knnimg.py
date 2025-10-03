# docs/roteiro5/knnimg.py
import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings, json
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

DATA_PATH  = Path("docs/data/TSLA_ready.csv")
OUT_DIR    = Path("docs/roteiro5")
MODEL_PATH = OUT_DIR / "knn_tsla.joblib"
BEST_PARAMS_PATH = OUT_DIR / "knn_best_params.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== Dados e features (alinhado ao KNNTreinamento.py) ====
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)
df["Target"] = (df["Change"].shift(-1) > 0).astype(int)

win = 60
df["Z_Change_roll"] = (df["Change"] - df["Change"].rolling(win, min_periods=win).mean()) / df["Change"].rolling(win, min_periods=win).std()
df["Z_Volume_roll"] = (df["Volume"] - df["Volume"].rolling(win, min_periods=win).mean()) / df["Volume"].rolling(win, min_periods=win).std()

df = df.dropna(subset=["Z_Change_roll", "Z_Volume_roll"]).iloc[:-1].reset_index(drop=True)

feature_cols = ["Z_Change_roll", "Z_Volume_roll"]
X = df[feature_cols].copy()
y = df["Target"].copy()

# ==== Carrega pipeline treinado, com fallback para reconstruir ====
try:
    pipe = joblib.load(MODEL_PATH)
except Exception:
    split_idx = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    if BEST_PARAMS_PATH.exists():
        best = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
        n_neighbors = int(best.get("knn__n_neighbors", 11))
        weights = best.get("knn__weights", "uniform")
        metric = best.get("knn__metric", "euclidean")
    else:
        n_neighbors, weights, metric = 11, "uniform", "euclidean"
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)),
    ])
    pipe.fit(X_train, y_train)

imputer = pipe.named_steps["imputer"]
scaler  = pipe.named_steps["scaler"]
knn     = pipe.named_steps["knn"]
k       = getattr(knn, "n_neighbors", None)
weights = getattr(knn, "weights", None)
metric  = getattr(knn, "metric", "euclidean")

# ==== 1) Fluxograma do pipeline ====
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.axis("off")

def box(x, y, w, h, text):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.06",
                          linewidth=1.6, edgecolor="black", facecolor="white")
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=11)

def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=12, linewidth=1.4))

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
box(0.05, 0.35, 0.22, 0.30, "SimpleImputer\n(median)")
arrow(0.27, 0.50, 0.34, 0.50)
box(0.34, 0.35, 0.22, 0.30, "StandardScaler")
arrow(0.56, 0.50, 0.63, 0.50)
box(0.63, 0.25, 0.32, 0.50, f"KNN\n(n_neighbors={k}\nweights='{weights}'\nmetric='{metric}')")
ax.text(0.5, 0.94, "Pipeline — KNN (TSLA)", ha="center", va="top", fontsize=13, fontweight="bold")
ax.text(0.5, 0.06, "Entrada: Z_Change_roll, Z_Volume_roll → Saída: Direção (t+1)", ha="center", fontsize=10)
plt.tight_layout()
fig.savefig(OUT_DIR / "knn_pipeline.png", dpi=220, transparent=True); plt.close(fig)

# ==== Pré-processa todo X (2D após scaler) ====
X_proc = scaler.transform(imputer.transform(X))

# ==== 2) k-vizinhos do penúltimo registro ====
idx_query = len(X_proc) - 2
dist, ind = knn.kneighbors(X_proc[[idx_query]], n_neighbors=k, return_distance=True)
d = dist.ravel(); neighbors_idx = ind.ravel(); neighbors_y = y.iloc[neighbors_idx].to_numpy()
order = np.argsort(d); d = d[order]; neighbors_idx = neighbors_idx[order]; neighbors_y = neighbors_y[order]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(range(len(d)), d)
for i, b in enumerate(bars):
    b.set_color("#1f77b4" if neighbors_y[i] == 1 else "#ff7f0e")
ax.set_yticks(range(len(d)))
ax.set_yticklabels([f"viz #{i+1} (idx={int(neighbors_idx[i])})" for i in range(len(d))], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Distância ao ponto consultado")
ax.set_title(f"k-vizinhos do penúltimo registro — k={k}, weights='{weights}', metric='{metric}'")
plt.tight_layout()
fig.savefig(OUT_DIR / "knn_neighbors_example.png", dpi=220, transparent=True); plt.close(fig)

# ==== 3) Regiões de decisão em 2D (usando as 2 features padronizadas) ====
split_idx = int(len(df) * 0.8)
train_idx = np.arange(split_idx)
test_idx  = np.arange(split_idx, len(df))

# Ajusta KNN no espaço 2D processado (sem PCA)
knn_2d = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
knn_2d.fit(X_proc[train_idx], y.iloc[train_idx])

x_min, x_max = X_proc[:,0].min() - .5, X_proc[:,0].max() + .5
y_min, y_max = X_proc[:,1].min() - .5, X_proc[:,1].max() + .5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
grid_pred = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 6))
ax.contourf(xx, yy, grid_pred, alpha=0.25, levels=2)
ax.scatter(X_proc[train_idx,0], X_proc[train_idx,1], s=10, alpha=0.6, label="treino", marker="o")
ax.scatter(X_proc[test_idx,0],  X_proc[test_idx,1],  s=12, alpha=0.8, label="teste",  marker="^")
ax.set_xlabel("Feature 1 (Z_Change_roll padronizado)")
ax.set_ylabel("Feature 2 (Z_Volume_roll padronizado)")
ax.set_title("KNN — Regiões de decisão (2D padronizado)")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "knn_decision_map_pca.png", dpi=220, transparent=True)  # mantemos o mesmo nome de arquivo
plt.close(fig)

# ==== 4) Badge de previsão (penúltimo dia) ====
x_last = X.iloc[[-2]]
pred = pipe.predict(x_last)[0]
proba = pipe.predict_proba(x_last)[0, int(pred)] if hasattr(pipe, "predict_proba") else np.nan
dref = df["Date"].iloc[-2].date()

fig, ax = plt.subplots(figsize=(4.8, 2.3))
ax.axis("off")
txt = (f"KNN — Previsão (penúltimo dia: {dref})\n"
       f"Classe prevista: {'Alta (1)' if pred==1 else 'Queda (0)'}  |  Prob.: {proba:.2f}\n"
       f"Config: n_neighbors={k}, weights='{weights}', metric='{metric}'")
ax.text(0.02, 0.60, txt, fontsize=10, va="center")
ax.text(0.02, 0.22, "Features: Z_Change_roll, Z_Volume_roll", fontsize=8, alpha=0.85)
plt.tight_layout()
fig.savefig(OUT_DIR / "knn_prediction_badge.png", dpi=220, transparent=True)
plt.close(fig)

print("PNG gerados em:", OUT_DIR.resolve())
