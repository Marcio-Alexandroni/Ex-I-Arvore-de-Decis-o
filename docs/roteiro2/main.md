<h1 id="data-cleaning">Data Cleaning</h1>

A limpeza de dados no dataset da Tesla envolveu a conversão da coluna de datas para o formato datetime, a ordenação cronológica das observações, a remoção da coluna redundante Adj Close e a verificação de duplicatas ou valores ausentes, que não foram encontrados. Essas etapas garantem consistência e preparação adequada dos dados para a análise e modelagem.

<h1 id="data-cleaning">Normalização e Padronização</h1>
O dataset foi preparado com as seguintes etapas:

- Criação da coluna Change (variação percentual diária do fechamento).

- Aplicação de Normalização (Min-Max) e Padronização (Z-Score) para Volume e Change.

- Geração do arquivo final TSLA_ready.csv contendo: Date, Volume, N-Volume, Z-Volume, Change, N-Change, Z-Change.


```python exec="on" html="0"
--8<-- "docs/roteiro2/TSLA-N&S.py"
```
/// caption
Sample rows from Dataset the TSLA Stock Data
///
