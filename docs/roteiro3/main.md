# Visualizações do Projeto

Nesta seção apresentamos as principais visualizações geradas a partir do dataset **TSLA_ready.csv**.

---

## Série temporal de Change

A figura mostra a variação diária do preço de fechamento da Tesla em relação ao dia anterior.  
A linha horizontal em zero indica a linha de base para identificar altas e quedas.

![Série temporal de Change](/change_timeseries.png)

---

## Série temporal de Change com Média Móvel (7 dias)

Aqui adicionamos a **média móvel de 7 dias**, que suaviza as oscilações de curto prazo e evidencia tendências mais claras.

![Série temporal de Change com MA7](/SerieTemporal.png)

---

## Matriz de Correlação

A matriz de correlação mostra as relações entre **Volume, Change** e suas versões normalizadas/padronizadas.  
Ela ajuda a avaliar redundâncias e interações entre variáveis antes da modelagem.

![Matriz de Correlação](/img/MatrizCorrelacao.png)

---

## Matriz de Confusão

A matriz de confusão avalia o desempenho do classificador de Árvore de Decisão em prever se a ação **subiu (1)** ou **caiu (0)**.  
Ela evidencia acertos e erros de classificação em cada classe.

![Matriz de Confusão](img/MatrizConfusao.png)
