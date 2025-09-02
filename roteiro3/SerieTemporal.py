import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("docs/Data/TSLA_ready.csv", parse_dates=["Date"])

# Série temporal de Change
df.plot(x="Date", y="Change", figsize=(10,4))
plt.axhline(0, ls="--")
plt.title("Variação diária (Change)")
plt.tight_layout()
plt.savefig("docs/roteiro3/SerieTemporal.png", dpi=200)
plt.close()

# Série temporal com média móvel 7d
df.assign(Change_MA7 = df["Change"].rolling(7).mean()) \
  .plot(x="Date", y=["Change","Change_MA7"], figsize=(10,4))
plt.axhline(0, ls="--")
plt.title("Change e média móvel 7 dias")
plt.tight_layout()
plt.savefig("docs/roteiro3/SerieTemporal.png", dpi=200)
plt.close()
