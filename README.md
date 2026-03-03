# 📊 Credit Default Classification

Este projeto consiste no desenvolvimento de um modelo de **Machine Learning** para a previsão de inadimplência (credit default), utilizando técnicas de classificação supervisionada. O objetivo principal é identificar se um cliente se tornará inadimplente com base em variáveis financeiras e comportamentais, auxiliando na gestão de risco e concessão de crédito.

---

## 🎯 Objetivo

Construir um pipeline completo de Machine Learning capaz de:
* Realizar o tratamento e preparação de dados brutos.
* Analisar e tratar o **desbalanceamento** da variável alvo.
* Treinar e otimizar modelos de classificação.
* Avaliar o desempenho com métricas específicas para o setor financeiro.
* Interpretar os resultados sob a ótica de negócio.

---

## 🧠 Contexto de Negócio

Para instituições financeiras, a precisão na previsão de inadimplência é vital para:
1.  **Reduzir perdas financeiras** diretas.
2.  **Ajustar limites de crédito** de forma dinâmica.
3.  **Melhorar políticas de concessão** e captação de clientes.
4.  **Minimizar o risco sistêmico** da carteira de ativos.

> **Definição do Problema:** Trata-se de uma **Classificação Binária**, onde:
> * `0`: Cliente Adimplente (Pagador)
> * `1`: Cliente Inadimplente (Devedor)



---

## 🛠️ Tecnologias Utilizadas

O projeto foi desenvolvido utilizando o ecossistema Python e ferramentas de análise de dados:

* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Visualização:** Matplotlib, Seaborn
* **Banco de Dados:** SQL Server (integração via `pyodbc`)
* **Ambiente:** Jupyter Notebook

---

## 🔎 Etapas do Projeto

### 1️⃣ Extração dos Dados
Conexão com banco de dados **SQL Server** através de consultas estruturadas para importação direta no DataFrame.

### 2️⃣ Análise Exploratória (EDA)
Identificação de valores nulos, análise de estatística descritiva e verificação da distribuição da variável alvo para detectar o grau de desbalanceamento.

### 3️⃣ Pré-processamento
* Divisão entre atributos (X) e rótulos (y).
* Separação em conjuntos de **treino, validação e teste**.
* Uso de **estratificação** para preservar a proporção das classes.
* Normalização e padronização de escalas.

### 4️⃣ Modelagem
Treinamento de algoritmos de classificação (ex: **Random Forest**) e ajuste de hiperparâmetros para busca da melhor performance.

### 5️⃣ Avaliação do Modelo
A performance é medida através de:
* **Recall:** Foco em evitar Falsos Negativos (clientes inadimplentes classificados como bons pagadores).
* **F1-Score:** Equilíbrio entre Precisão e Recall.
* **Matriz de Confusão:** Visualização clara dos erros e acertos.

---

## 📈 Interpretação dos Resultados

O modelo busca equilibrar o **Trade-off** de concessão:
* **Rigor excessivo:** Perda de bons clientes e receita.
* **Aprovação excessiva:** Aumento da taxa de inadimplência e prejuízo.



---

## 🏗️ Estrutura do Repositório

```text
credit-default-classification/
│
├── data/                       # (Opcional) Base de dados ou amostras
├── credit-default-classification.ipynb
├── README.md                   # Documentação do projeto
└── requirements.txt            # Dependências do Python
```

## 🚀 Como Executar o Projeto

### 1️⃣ Clone o repositório

```bash
git clone https://github.com/mm-dantas/credit-default-classification.git
```

### 2️⃣ Acesse o diretório do projeto

```bash
cd credit-default-classification
```

### 3️⃣ Execute o Jupyter Notebook

```bash
jupyter notebook
```
