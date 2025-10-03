import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Carregar dataset final
df = pd.read_csv("docs/Data/TSLA_ready.csv", parse_dates=["Date"])

# Features e Target
X = df[["N-Volume","Z-Volume","N-Change","Z-Change"]]
y = (df["Change"] > 0).astype(int)   # 1 = Subiu, 0 = Caiu

# Split train/test
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar árvore de decisão
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(Xtr, ytr)

# Matriz de confusão
cm = confusion_matrix(yte, clf.predict(Xte))
disp = ConfusionMatrixDisplay(cm, display_labels=["Caiu (0)", "Subiu (1)"])
disp.plot(values_format="d")
plt.title("Matriz de Confusão — Decision Tree")
plt.tight_layout()
plt.savefig("docs/roteiro3/MatrizConfusao.png", dpi=200)
plt.close()
