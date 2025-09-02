# Árvore de Decisão — TSLA (Direção do Próximo Dia)

![Árvore de Decisão TSLA](./roteiro4/tree_tsla.png){ align=center }

> **Configuração**  
> - Alvo: `Target(t+1) = 1` se `Change(t+1) > 0` (alta), senão `0` (queda).  
> - Split temporal: 80% treino / 20% teste.  
> - Hiperparâmetros: `gini`, `max_depth=5`, `min_samples_split=20`, `min_samples_leaf=10`, `class_weight=balanced`.  
> - Features: `Volume`, `N-Volume`, `Z-Volume`, `Change`, `N-Change`, `Z-Change`.

---

## Coeficiente de Gini (impureza)
O **Gini** mede o quão “misturado” um nó está em termos de classes.  
- Fórmula (intuição): \\( \text{Gini} = 1 - \sum p_k^2 \\), onde \\(p_k\\) é a proporção da classe *k* no nó.  
- **Quanto menor o Gini**, mais “puro” o nó (maioria clara de uma classe).  
- O algoritmo escolhe os *splits* que **reduzem** o Gini (ganho de impureza), deixando os nós-filhos mais homogêneos.

---

## Matriz de Confusão (visão rápida)
A matriz de confusão informa **o que acertamos e erramos** no conjunto de teste:

- **Verdadeiro Positivo (TP)**: modelo previu **Alta (1)** e o próximo dia realmente foi **Alta**.  
- **Verdadeiro Negativo (TN)**: modelo previu **Queda (0)** e o próximo dia realmente foi **Queda**.  
- **Falso Positivo (FP)**: previu **Alta**, mas ocorreu **Queda**.  
- **Falso Negativo (FN)**: previu **Queda**, mas ocorreu **Alta**.

> Métricas úteis derivadas:
> - **Accuracy**: proporção total de acertos.  
> - **Precision (classe 1)**: entre as **Altas previstas**, quantas foram Altas de fato?  
> - **Recall (classe 1)**: entre as **Altas reais**, quantas o modelo acertou?  
> - **F1 (classe 1)**: média harmônica de Precision e Recall (robusta a desequilíbrios).

> *(As tabelas `tsla_metrics.csv` e `tsla_confusion_matrix.csv` ficam em `docs/roteiro4/` para referência técnica.)*

---

## Conclusão
A árvore de decisão indica que **a variação do dia corrente** (especialmente `N-Change` e `Z-Change`) e a **intensidade de volume** (`Volume`, `N-Volume`, `Z-Volume`) são determinantes para prever a direção do **dia seguinte**.  
O modelo está **regularizado** (profundidade e mínimos por nó) e **respeita a temporalidade** (split temporal), o que o torna **interpretável** e adequado como **baseline acadêmico**.  

- **Pontos fortes**: regras claras (“se/então”), importâncias de variáveis transparentes e facilidade de comunicação.  
- **Limitações**: por ser um modelo simples, pode não capturar dinâmicas de mercado mais complexas. Em trabalhos futuros, vale comparar com **ensembles** (Random Forest, XGBoost) e validação temporal (*TimeSeriesSplit*).

---
