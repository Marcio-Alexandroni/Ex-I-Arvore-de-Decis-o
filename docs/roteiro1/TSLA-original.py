import pandas as pd

# Load the Titanic dataset
df = pd.read_csv('https://github.com/Marcio-Alexandroni/Ex-I-Arvore-de-Decis-o/blob/main/docs/data/TSLA.csv')
df = df.sample(n=10, random_state=42)

# Display the first few rows of the dataset
print(df.to_markdown(index=False))