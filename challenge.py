import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# streamlit_app.py

import hmac


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.



# st.title("Grão Direto Challenge")

# Descrever o projeto brevemente
"""
# Grão Direto IA Challenge

O desafio consiste em criar um modelo de Machine Learning para prever a probabilidade de um cliente realizar uma transação.
"""
"""
# Para isso, buscaremos entender um pouco melhor os dados visualmente.

**Entendendo a variação do CBOT, preço de mercado e Dólar no tempo, ambos a nível de produto.**"""

img1 = plt.imread('images/precosnomtepo.png')
st.image(img1, caption='Preço dos grãos e dolar pelo tempo (informações de mercado)')

"O gráfico acima já se mostrou mais informativo, de modo que é perceptível que as tendências do `Milho` e da `Soja` são extremamente parecidas em gráficos temporais, muito embora a correlação entre `CBOT` e `PRICE` quando comparados os grãos não tenha se mostrado alta."

"Agora vamos analisar a variação do `price` e do `amount` ao longo do tempo, para tentar entender melhor a relação entre essas variáveis."

"**Vamos também dar uma olhada a a variação do preço e compra/venda de grãos no tempo**"

img2 = plt.imread('images/transacoes_no_tempo.png')
st.image(img2, caption='Preço pago e número de sacas no tempo com o dólar (informações de mercado)')

"""Aparentemente, apresenta-se uma mesma tendência para o `milho` e a `soja`, no entanto foi observado um outlier extremo para a soja no mês de Maio.

Esse outlier pode ser potencialmente algum erro nos dados, mas não pode ser descartado que o mercado possa ter oscilado muito. A opção será por mantê-lo.

Caso o modelo apresente resultados ruins, podemos retornar e fazer a remoção do mesmo."""


"**Vamos tentar entender melhor a diferença entre o `Milho` e a `Soja`.**"

img3 = plt.imread('images/milho_soja.png')
st.image(img3, caption='Preço dos grãos e dolar pelo tempo')

"Parece haver duas nuvens separadas de pontos, o que sugere dizer que essas nuvens separadas se dão pela diferença de grãos."

"""**Vamos reunir as informações de mercado e transações por mês e buscar potenciais correlações:**"""
img4 = plt.imread('images/heatmap_.png')
st.image(img4, caption='Heatmap de correlação entre as variáveis')

"""- `Dólar` e `CBOT` têm forte impacto nas variáveis relacionadas a preço e quantidade.
- Preço de mercado e preço de transações são altamente correlacionados, o que é esperado."""

"""**Buscando entender também as variáveis `categóricas`**"""

img5 = plt.imread('images/cat_mercado.png')
st.image(img5, caption='Variáveis categóricas de mercado')

img5 = plt.imread('images/cat_transacoes.png')
st.image(img5, caption='Variáveis categóricas de transações')

"""
`MG, MT e RO` se mostraram como os estado que mais tiveram exportações de `soja` (*`MG - tem a ver com a grão direto?!`* haha).

Ao que parece, o `MT` também apresenta influencia relevante nas vendas de `Milho`.

A companhia `Polaris` também se mostrou como forte exportadora de `soja`, potencialmente como a principal empresa de exportação tanto da soja quanto dos grãos em geral.

O `milho` parece não ser muito vendido, e provavelmente a compra da soja seja mais atrativa para os compradores.

Vamos tentar entender um pouco melhor sobre os vendedores, que parece ser uma variável extremamente importante para a análise atual.
"""
img5 = plt.imread('images/vendedores.png')
st.image(img5, caption='Top 10 vendedores em todo o período')

"Parecem existir vendedores que atuam muito mais fortemente no mercado, e esses 10 representam `11% do total de vendedores`."

with st.expander("Abordagem para a modelagem", True):
  """# Abordagens para a modelagem

  - Serão removidas as colunas `Price (transações)`
  - `Amount (transações)`

  **Pois são colunas dependentes de que haja uma transação, e o que queremos descobrir é se haverá uma transação, então não faz sentido mantê-las, além de potencialmente causar vazamento de dados.**

  Adição das colunas:

  - `Dia do mês`
  - `Dia da semana`
  - `Semana do mês`
  """

with st.expander("Abordagem para a predição", True):
  """
  ## Abordagem para a predição
  **A opção será por utilizar o Vendedor `(Seller ID)` como variável alvo e APENAS os `dados de mercado, aliado às variáveis mencionadas acima` colunas como variáveis independentes.**

  **A métrica de avaliação principal será o `precision_weighted`, pois nos importa mais se o que o modelo diz está correto, buscando evitar Falsos positivos do modelo, pois já teríamos mais certeza do próximo dia, nos antecipando às demandas de mercado e executando as operações e transações corretas.**

  A abordagem é pelo precision `weighted`, pois essa variável nos proporciona uma média ponderada do precision de cada classe, o que é importante para a análise, uma vez que temos dados desbalanceados e um número muito grande de classes.

  A abordagem será de utilizar modelos de Machine Learning Baseados em árvores de decisão, principalmente adaptados a séries temporais, e os motivos são:
  - Modelos de árvore proporcionam probabilidade de classificação
  - Boa capacidade preditiva para multiclasses (número de vendedores)
  - Bom funcionamento com variáveis desbalanceadas.
  """

with st.expander("Modelos utilizados", True):
  """Modelos utilizados nas análises:
  - Random Forest,
  - Extra trees,
  - Árvores de decisão,
  - Dummy Classifier.

  **E também modelos de Machine Learning adaptados a séries temporais (consultar documentação em [sktime](https://www.sktime.net/en/stable/)), como:**
  - `TimeSeriesForestClassifier`
  """

  """### Observação, todos os testes foram feitos com `Time Series Split validation` [sklearn-TimeSeriesSplit](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), respeitando a ordem temporal dos dados."""

with st.expander("Tentativas de modelagem", True):
  """Foram tentadas diversas abordagens para a modelagem, desde:
  - Remover vendedores com menos de 5 vendas e com vendas ausentes nos últimos 30 dias,
  - Remover apenas vendedores com menos de 5 vendas,
  - Utilizar todo o conjunto de vendedores.

  **Utilizar `todo o conjunto de vendedores` foi o que nos apresentou melhores resultados, e por isso foi a abordagem escolhida para a modelagem final.**
  """

with st.expander("Resultados da modelagem", True):
  """O modelo que apresentou o melhor resultado nos dados de teste foi o `TimeSeriesForestClassifier`, com um `precision_weighted` de aproximadamente `10%` ***(Note que embora o valor inicialmente pareça baixo, trata-se de uma predição multiclasse com 2423 diferentes vendedores)***."""

  """
  Para a predição do dia `11-04-2024`, os sellers IDs cujas probabilides foram maiores podem ser vistos pela tabela a seguir:
  """

  