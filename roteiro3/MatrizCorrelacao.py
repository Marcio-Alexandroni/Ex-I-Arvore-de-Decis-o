import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar dataset final
df = pd.read_csv("docs/data/TSLA_ready.csv", parse_dates=["Date"])

# Selecionar colunas de interesse
cols = ["Volume","Change","N-Volume","Z-Volume","N-Change","Z-Change"]
corr = df[cols].corr()

# Plot heatmap simples
plt.figure(figsize=(7,6))
plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
plt.yticks(range(len(cols)), cols)
plt.colorbar(label="Correlação")
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.savefig("docs/roteiro3/MatrizCorrelacao.png", dpi=200)
plt.close()
