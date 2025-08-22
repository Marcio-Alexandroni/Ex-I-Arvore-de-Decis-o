import pandas as pd

# Carregar o dataset TSLA do GitHub
url = 'https://raw.githubusercontent.com/Marcio-Alexandroni/Ex-I-Arvore-de-Decis-o/main/docs/Data/TSLA_scaled.csv'
df = pd.read_csv(url)

# Amostra de 10 linhas
df = df.sample(n=10, random_state=42)

# Exibir algumas linhas
print(df.to_markdown(index=False))
