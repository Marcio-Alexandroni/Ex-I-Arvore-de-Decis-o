import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Carregar dataset já limpo
df = pd.read_csv("docs/data/TSLA_clean.csv", parse_dates=["Date"])

# Criar coluna Change = variação percentual do fechamento
df["Change"] = df["Close"].pct_change()

# Remover primeira linha com NaN
df = df.dropna().reset_index(drop=True)

# Inicializar escaladores
scaler_minmax = MinMaxScaler()
scaler_std = StandardScaler()

# Normalização e padronização do Volume
df["N-Volume"] = scaler_minmax.fit_transform(df[["Volume"]])
df["Z-Volume"] = scaler_std.fit_transform(df[["Volume"]])

# Normalização e padronização do Change
df["N-Change"] = scaler_minmax.fit_transform(df[["Change"]])
df["Z-Change"] = scaler_std.fit_transform(df[["Change"]])

# Selecionar apenas as colunas pedidas
df_final = df[["Date", "Volume", "N-Volume", "Z-Volume", "Change", "N-Change", "Z-Change"]]

# Salvar dataset final
df_final.to_csv("docs/data/TSLA_ready.csv", index=False)

print("✅ Dataset final criado: docs/data/TSLA_ready.csv")
print(df_final.head())
