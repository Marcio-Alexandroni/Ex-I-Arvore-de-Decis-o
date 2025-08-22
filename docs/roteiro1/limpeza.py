import pandas as pd
from pathlib import Path

# ---------- 1) Localizar o CSV bruto ----------
candidatos = [
    Path("TSLA.csv"),                     # raiz do projeto
    Path("docs/data/TSLA.csv"),                # pasta data/
]
raw_path = next((p for p in candidatos if p.exists()), None)
if raw_path is None:
    raise FileNotFoundError(
        "Não encontrei TSLA.csv nos caminhos candidatos. "
        "Coloque o arquivo em /docs/Data/Kaggle/TSLA.csv ou ./data/TSLA.csv ou ./TSLA.csv."
    )

# ---------- 2) Ler e limpar ----------
df = pd.read_csv(raw_path)

# 2.1 Converter 'Date' para datetime com fallbacks
# Primeiro: tentativa genérica
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Se ainda houver NaT, tentar formatos específicos
if df['Date'].isna().any():
    df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y", errors='coerce')
if df['Date'].isna().any():
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d", errors='coerce')

# Checagem final de datas
if df['Date'].isna().any():
    # mostra algumas amostras problemáticas para debug
    exemplos_ruins = df[df['Date'].isna()].head()
    raise ValueError(f"Falha ao converter 'Date' em algumas linhas. Exemplos:\n{exemplos_ruins}")

# 2.2 Ordenar por data
df = df.sort_values('Date').reset_index(drop=True)

# 2.3 Remover coluna redundante
if 'Adj Close' in df.columns:
    df = df.drop(columns=['Adj Close'])

# 2.4 Remover duplicatas
antes = len(df)
df = df.drop_duplicates()
dups_removed = antes - len(df)

# 2.5 Relatório de nulos
null_report = df.isnull().sum()

# ---------- 3) Salvar versão clean ----------
destino = Path("docs/data/TSLA_clean.csv")
try:
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False)
except Exception:
    # fallback para dentro do projeto
    destino = Path("data/TSLA_clean.csv")
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False)

# ---------- 4) Resumo ----------
summary = {
    "rows_clean": len(df),
    "cols_clean": df.shape[1],
    "date_min": df['Date'].min(),
    "date_max": df['Date'].max(),
    "duplicates_removed": dups_removed,
    "saved_to": str(destino),
}

print("Resumo:", summary)
print("\nNulos por coluna:\n", null_report)
print("\nHead:\n", df.head())
print("\nTail:\n", df.tail())
