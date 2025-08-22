import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Carregar dataset já limpo (ex: TSLA_clean.csv)
df = pd.read_csv("docs/Data/TSLA_clean.csv", parse_dates=["Date"])

# Selecionar apenas colunas numéricas para escalar
num_cols = ['Open','High','Low','Close','Volume']

# --- Normalization (Min-Max Scaling [0,1]) ---
scaler_minmax = MinMaxScaler()
df_minmax = pd.DataFrame(scaler_minmax.fit_transform(df[num_cols]), 
                         columns=[col+"_minmax" for col in num_cols])

# --- Standardization (Z-score) ---
scaler_std = StandardScaler()
df_std = pd.DataFrame(scaler_std.fit_transform(df[num_cols]), 
                      columns=[col+"_std" for col in num_cols])

# Concatenar com o dataset original
df_scaled = pd.concat([df, df_minmax, df_std], axis=1)

# Salvar versão final
df_scaled.to_csv("docs/Data/TSLA_scaled.csv", index=False)

print("✅ Normalização e padronização concluídas.")
print(df_scaled.head())
