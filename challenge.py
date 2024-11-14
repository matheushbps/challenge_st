import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
# streamlit_app.py

import hmac
# st.set_page_config(layout="wide")

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

# Descrever o projeto brevemente
col1, col2 = st.columns([8, 1])
with col1:
  st.write("# Grão Direto IA Challenge")
with col2:
  st.image('images/gd2.png', width=200)

with st.expander("Análise exploratória e abordagens iniciais", False):
  """
  #### O desafio consiste em criar um modelo de Machine Learning para prever a probabilidade de um cliente realizar uma transação.
  #### Para isso, buscaremos entender um pouco melhor os dados visualmente.

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

with st.expander("Abordagem para a modelagem", False):
  """# Abordagens para a modelagem

  Serão removidas as colunas:
  
  - `Price (transações)`
  - `Amount (transações)`

  **Pois são colunas dependentes de que haja uma transação, e o que queremos descobrir é se haverá uma transação, então não faz sentido mantê-las, além de potencialmente causar vazamento de dados.**

  Adição das colunas:

  - `Dia do mês`
  - `Dia da semana`
  - `Semana do mês`
  """

with st.expander("Ponto importantíssimo sobre a modelagem", False):
   """Percebe-se que os dados de 2024-11-05 já foram dados, que são:
   - `CBOT`,
   - `Preço de mercado`,
   - `Dólar (obtido no código)`.

   
   **No entanto, em um caso real, `esses dados não estariam disponíveis`, e para obtê-los, precisaríamos prevê-los com modelos de regressão, também baseados em séries temporais (ou técnicas mais avançadas)**.
   
   Em posse desses dados preditos, poderíamos alimentar o modelo presente no projeto atual.

   **Perceba que em um mercado de grãos, tais como em mercados de ações, há muitas incertezas e portanto a predição do modelo aqui presente sendo alimentada por predições de regressão de outros modelos potencialmente adicionaria mais incertezas ao modelo final.**
   """

with st.expander("Abordagem para a predição", False):
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

with st.expander("Modelos utilizados", False):
  """Modelos utilizados nas análises:
  - Random Forest,
  - Extra trees,
  - Árvores de decisão,
  - Dummy Classifier.

  **E também modelos de Machine Learning adaptados a séries temporais (consultar documentação em [sktime](https://www.sktime.net/en/stable/)), como:**
  - `TimeSeriesForestClassifier`
  """

  """### Observação, todos os testes foram feitos com `Time Series Split validation` [sklearn-TimeSeriesSplit](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), respeitando a ordem temporal dos dados."""

with st.expander("Tentativas de modelagem", False):
  """Foram tentadas diversas abordagens para a modelagem, desde:
  - Remover vendedores com menos de 5 vendas e com vendas ausentes nos últimos 30 dias,
  - Remover apenas vendedores com menos de 5 vendas,
  - Utilizar todo o conjunto de vendedores.

  **Utilizar `todo o conjunto de vendedores` foi o que nos apresentou melhores resultados, e por isso foi a abordagem escolhida para a modelagem final.**
  """

with st.expander("Resultados da modelagem", False):
  """O modelo que apresentou o melhor resultado nos dados de teste foi o `TimeSeriesForestClassifier`, com um `precision_weighted` de aproximadamente `10%` ***(Note que embora o valor inicialmente pareça baixo, trata-se de uma predição multiclasse com 2423 diferentes vendedores)***."""

  """
  Para a predição do dia `11-04-2024`, os sellers IDs cujas probabilides foram maiores podem ser vistos pela tabela a seguir:
  """

df_probs_14 = pd.read_csv('top_sellers_04_11.csv')
df_probs_15 = pd.read_csv('probabilities_15.csv')
df_probs_15['Probability'] = round(df_probs_15['Probability'].mul(100),2)
df_probs_14['Probability'] = round(df_probs_14['Probability'].mul(100),2)
df_probs_14.rename({'Probability':'Probability (%)'}, axis=1,inplace=True)
df_probs_15.rename({'Probability':'Probability (%)'}, axis=1, inplace=True)

# Add a slider to select the number of top sellers to display



with st.expander("Probabilidades de transações dos vendedores", False):
  num_sellers = st.slider('Selecione o número máximo de vendedores com maior probabilidade de venda para o dia 14-11:', min_value=1, max_value=100, value=10)
  top_sellers = df_probs_14.head(num_sellers)
  st.write(top_sellers)


  f"""Essa predição foi feita pra cada linha do conjunto de teste. No caso, temos {df_probs_14.shape[0]} vendedores que poderiam fazer uma transação no dia 04/11/2024, pois para cada informação de mercado, há a probabilidade de diferentes vendedores realizarem a transação. """

  # Add a slider to select the number of top sellers to display
  num_sellers = st.slider('Selecione o número máximo de vendedores com maior probabilidade de venda para o dia 15-11:', min_value=1, max_value=100, value=10)

  # Display the top sellers based on the selected number
  top_sellers = df_probs_15.head(num_sellers)
  st.write(top_sellers)


  df_results = pd.read_csv('df_results.csv')
  df_results.rename(columns={'Unnamed: 0': 'Variável'}, inplace=True)
  df_sorted = df_results.sort_values(by="mean", ascending=False)
  df_sorted = df_sorted.set_index('Variável')

  """E aqui pode-se analisar o desempenho de cada métrica para o modelo"""
  # Criar um gráfico de barra com Altair
  chart = alt.Chart(df_sorted.reset_index()).mark_bar(color='#008000').encode(
    x='mean:Q',
    y=alt.Y('Variável:N', sort='-x')
  ).properties(
    height=500
  )

with st.expander("Importância das variáveis no modelo", False):
  st.altair_chart(chart, use_container_width=True)

  """Aqui podemos ver as features que mais impactaram na tomada de decisão do modelo (as features que mais reduzem a imperza nos nós das árvores do modelo).

  Reiterando que essas features são as que mais impactaram na decisão do modelo, e não são necessariamente as features que efetivamente determinam o mercado, mas a relação entre os dados que o modelo encontrou foram essas:

  - `Dolar`,
  - `Origin state`,
  - `Origin city`,
  - `Price (mercado)`,
  - `CBOT`.

  Aparentemente o dólar foi uma ótima adição ao modelo, sendo uma das features que mais influenciam e proporcionam informação pro mesmo fazer a predição."""


with st.expander("Conclusões", False):
  """
  # Conclusão e Próximos passos

  O projeto atual se mostrou bastante desafiador por três principais motivos: A predição de séries temporais para classificação, com um número muito grande de classes e que requisitasse probabilidade. Além da limitação de não poder utilizarmos modelos de redes neurais.

  O modelo feito não teve um prediction muito alto, dada a complexidade da predição de diversas classes e a quantidade de dados disponíveis.
  No entanto, alguns pontos poderiam ter sido utilizados de modo a melhorar o modelo:

  - Utilizar modelos de Deep learning, que teriam um potencial exponencialmente maior de capturar padrões nos dados.
  - Tentar mais técnicas de feature engineering, trazendo dados de negócio, como fontes do IBGE, CONAB, e outras fontes de dados do mercado de grãos.
  - Realizar um tuning de hiperparâmetros mais extenso, de modo a descobrir features melhores dos modelos e reduzir o overfitting, aumentando a capacidade de generanalização do modelo.
  - Potencialmente fazer conversões nos dados para buscar uma regressão (pensei nesta abordagem durante o projeto, mas não encontrei alguma técnica plausível).
  - Com mais tempo, fazer uma análise exploratória mais profunda, buscando encontrar padrões potencialmente ainda não encontrados na atual análise.
  - Transformar o modelo numa abordagem binária, onde o Seller ID poderia ser uma categoria (eu fiz essa abordagem, mas isso tornou o dataset muito grande e fiquei sem recursos para o modelo).
  - Com a presença de dados reais, poderíamos ter mais insumo pra alimentar o modelo, também potencialmente contribuindo com a reduçao de overfitting.
  - Testar outros modelos do sktime para predição de séries temporais.
  - Fazer conversões nos dados de modo a utilizar abordagem comum de séries temporais (ARIMA, SARIMA, Seasonal decomposing, suavizaçao exponencial).
  - Fazer o LAG nos dados, oo que potencialmente aumentaria a capacidade de predição de modelos de machine learning com abordagem tradicional.
  - Fazer o agrupamento dos dados com KMeans, de modo a reduzir o número de classes e fazer a predição sobre um grupo de potenciais vendedores (aprendizagem semi supervisionada).
  - Fazer uma abordagem de classificação binária, onde cada seller_id poderia ser um imput e a predição seria se ele faria ou não uma transação, no entanto após tentar realizar essa abordagem, percebeu-se dificuldade em lidar com um grande dataset gerado.

  Enfim... Várias são as ideias, muitas delas não houve tempo ou recursos computacionais para serem executadas, mas no geral acredito que a abordagem tenha seguido bons padrões e atendido à demanda que era a de entregar a probabilidade de potenciais compradores nos dias 04/11 e 05/11/2024.

  """