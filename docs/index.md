# Exercício I - Machine Learning
## Exploração Kaggle Dataset - TESLA Stock Data


???+ info inline end "Edição"

    2025.1


## Grupo

1. Márcio Alexandroni da Silva Filho

!!! tip "📂 Sobre o Dataset"

    Fonte: Kaggle (dados históricos de ações da Tesla).

    Período: 6/29/2010 a 3/24/2022

    Colunas típicas:<br>
        Date → Data de negociação<br>
        Open → Preço de abertura<br>
        High → Maior preço no dia<br>
        Low → Menor preço no dia<br>
        Close → Preço de fechamento<br>
        Adj Close → Preço ajustado (leva em conta splits e dividendos)<br>
        Volume → Quantidade de ações negociadas

## Entregas

- [x] Escolha do Dataset
- [x] Limpeza dos Dados
- [x] Normalização e Padronização
- [x] Diagramas Mermaid do Projeto
- [ ] Visualizações
- [ ] Árvore de Decisão
- [ ] Projeto Concluído

## Diagramas

Feito diagramas [Mermaid](https://mermaid.js.org/intro/){:target='_blank'} para a criação dos diagramas de documentação.

### Pipeline do projeto

O pipeline mostra o fluxo de etapas do trabalho, desde o dataset bruto até a avaliação do modelo. Inclui: limpeza dos dados, criação de novas variáveis (Change e escalas), definição do target (Subiu/Caiu), divisão treino/teste, treinamento da árvore de decisão e avaliação do desempenho.

```mermaid
flowchart TD
    A[Dataset TSLA] --> B[Data Cleaning]
    B --> C[Feature Engineering - Change, scales]
    C --> D[Normalization and Standardization]
    D --> E[Target Definition - Up/Down]
    E --> F[Train/Test Split]
    F --> G[Decision Tree]
    G --> H[Evaluation - metrics]

    %% classes
    class A,B,C,D,E,F,H orange
    class G red

    %% class defs (same as your working sample)
    classDef red fill:#f55
    classDef orange fill:#ffa500
```


### Lógica do Target

A lógica do target define como criamos a variável de saída do modelo: se a variação diária de fechamento (Change) for positiva, marcamos como 1 (Subiu); se for negativa ou zero, marcamos como 0 (Caiu).

```mermaid
flowchart TD
    S(Compute Change) --> Q(Change > 0?)
    Q -->|Yes| U(Target = 1 Up)
    Q -->|No| D(Target = 0 Down)

    %% classes
    class S,Q,D,U orange
    class U red

    %% class defs
    classDef red fill:#f55
    classDef orange fill:#ffa500
```





## Códigos

=== "De um arquivo remoto"

    ``` { .yaml .copy .select linenums='1' title="main.yaml" }
    --8<-- "https://raw.githubusercontent.com/hsandmann/documentation.template/refs/heads/main/.github/workflows/main.yaml"
    ```

=== "Anotações no código"

    ``` { .yaml title="compose.yaml" }
    name: app

        db:
            image: postgres:17
            environment:
                POSTGRES_DB: ${POSTGRES_DB:-projeto} # (1)!
                POSTGRES_USER: ${POSTGRES_USER:-projeto}
                POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-projeto}
            ports:
                - 5432:5432 #(2)!
    ```

    1.  Caso a variável de ambiente `POSTGRES_DB` não exista ou seja nula - não seja definida no arquivo `.env` - o valor padrão será `projeto`. Vide [documentação](https://docs.docker.com/reference/compose-file/interpolation/){target='_blank'}.

    2. Aqui é feito um túnel da porta 5432 do container do banco de dados para a porta 5432 do host (no caso localhost). Em um ambiente de produção, essa porta não deve ser exposta, pois ninguém de fora do compose deveria acessar o banco de dados diretamente.


## Exemplo de vídeo

Lorem ipsum dolor sit amet

<iframe width="100%" height="470" src="https://www.youtube.com/embed/3574AYQml8w" allowfullscreen></iframe>


## Referências

[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/reference/){:target='_blank'}