import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
from pathlib import Path

DATA_PATH = Path("docs/data/TSLA_ready.csv")
OUT_DIR   = Path("docs/roteiro4")
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

classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight=class_weight,
    random_state=42
).fit(X_train, y_train)

y_pred = clf.predict(X_test)
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision (1=Alta)": precision_score(y_test, y_pred, zero_division=0),
    "Recall (1=Alta)": recall_score(y_test, y_pred, zero_division=0),
    "F1 (1=Alta)": f1_score(y_test, y_pred, zero_division=0),
}
pd.DataFrame([metrics]).to_csv(OUT_DIR / "tsla_metrics.csv", index=False)

cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, index=["Real: 0", "Real: 1"], columns=["Pred: 0", "Pred: 1"]).to_csv(
    OUT_DIR / "tsla_confusion_matrix.csv", index=False
)

importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
importances.reset_index().rename(columns={"index": "Feature", 0: "Importance"}).to_csv(
    OUT_DIR / "tsla_feature_importances.csv", index=False
)

plt.figure(figsize=(18, 10))
plot_tree(clf, feature_names=feature_cols, class_names=["Queda (0)", "Alta (1)"],
          filled=True, rounded=True, fontsize=9)
plt.title("Árvore de Decisão - Direção do Próximo Dia (TSLA)")
plt.tight_layout()
plt.savefig(OUT_DIR / "tree_tsla.png", dpi=220)
plt.close()

(OUT_DIR / "tsla_tree_rules.txt").write_text(export_text(clf, feature_names=feature_cols, max_depth=4), encoding="utf-8")
joblib.dump(clf, OUT_DIR / "tree_tsla.joblib")

print("Treinamento concluído. Artefatos salvos em:", OUT_DIR)
