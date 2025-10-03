# docs/roteiro5/KNN.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Caminhos
DATA_PATH = Path("docs/data/TSLA_ready.csv")
MODEL_PATH = Path("docs/roteiro5/knn_tsla.joblib")

# Carrega dados
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)

# Target (para referência/consistência com o treino)
df["Target"] = (df["Change"].shift(-1) > 0).astype(int)

# FEATURES: rolling z-score sem vazamento (mesma lógica do KNNTreinamento.py)
win = 60
df["Z_Change_roll"] = (df["Change"] - df["Change"].rolling(win, min_periods=win).mean()) / df["Change"].rolling(win, min_periods=win).std()
df["Z_Volume_roll"] = (df["Volume"] - df["Volume"].rolling(win, min_periods=win).mean()) / df["Volume"].rolling(win, min_periods=win).std()

# Remove linhas sem janela completa e a última (por causa do shift do Target no treino)
df = df.dropna(subset=["Z_Change_roll", "Z_Volume_roll"]).iloc[:-1].reset_index(drop=True)

# Seleciona features consistentes com o modelo treinado
feature_cols = ["Z_Change_roll", "Z_Volume_roll"]
X = df[feature_cols].copy()

# Carrega pipeline treinado e faz uma previsão de demonstração
pipe = joblib.load(MODEL_PATH)

# Usamos o penúltimo registro para manter consistência com as visualizações
x_query = X.iloc[[-2]]
pred = pipe.predict(x_query)[0]
proba = pipe.predict_proba(x_query)[0, int(pred)] if hasattr(pipe, "predict_proba") else np.nan

# Extrai config do KNN
knn = pipe.named_steps.get("knn")
k = getattr(knn, "n_neighbors", None)
weights = getattr(knn, "weights", None)

print("Config do modelo KNN:", {"n_neighbors": k, "weights": weights})
print("Features:", feature_cols)
print("\nPrevisão (penúltimo registro):")
print("Data de referência:", df.loc[len(df)-2, "Date"].date())
print("Classe prevista:", int(pred), "| Probabilidade:", float(proba))
