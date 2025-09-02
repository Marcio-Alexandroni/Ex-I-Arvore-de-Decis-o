# Objetivo

Exploração e Análise do Kaggle Dataset 'TESLA Stock Data'

# Montagem do Roteiro

## Desafio I

### Tarefas Típicas de Pré-processamento

| Tarefa                 | Descrição                                                                 |
|-------------------------|---------------------------------------------------------------------------|
| **Limpeza de Texto**    | Remove caracteres indesejados, stop words e aplica stemming/lematização. |
| **Normalização**        | Padroniza formatos de texto, como datas e moedas.                         |
| **Tokenização**         | Divide o texto em palavras ou subpalavras para facilitar a análise.       |
| **Extração de Atributos** | Converte o texto em características numéricas usando técnicas como TF-IDF ou embeddings. |
| **Aumento de Dados**    | Gera dados sintéticos para aumentar o tamanho e a diversidade do dataset. |

## Plano do Projeto

Este projeto tem como objetivo analisar, categorizar e estruturar um dataset real — neste caso, os dados históricos das ações da Tesla — aplicando conceitos fundamentais de Ciência de Dados. O processo inclui exploração inicial, limpeza, criação de novas variáveis e normalização/padronização, além de visualizações que permitem identificar padrões e compreender melhor o comportamento dos dados.

Como parte desse pipeline, utilizamos a Árvore de Decisão para transformar o problema em uma tarefa de classificação, prevendo se o preço de fechamento diário da ação subiria ou cairia. Essa modelagem exemplifica o uso de algoritmos de Machine Learning dentro de um contexto mais amplo, no qual o foco é demonstrar todo o ciclo de preparação, análise e leitura de dados que sustentam diferentes experimentos e aplicações em aprendizado de máquina.

- Arquivo CSV Originl do TSLA Stock Data:

```python exec="on" html="0"
--8<-- "docs/roteiro1/TSLA-original.py"
```
/// caption
Sample rows from Dataset the TSLA Stock Data
///

## Discussões

- Em Breve

## Conclusão

= Em Breve
