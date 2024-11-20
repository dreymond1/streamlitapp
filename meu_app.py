import streamlit as st
import pandas as pd
from sentiment_analysis import prever_sentimento  # Importe o arquivo limpo como um módulo

st.set_page_config(page_title="Meu Site Streamlit")

with st.container():
    st.subheader("Meu primeiro site com o Streamlit")
    st.title("Dashboard de Contratos")
    st.write("Informações sobre os contratos fechados pela Hash&Co ao longo de maio")
    st.write("Quer aprender Python? [Clique aqui](https://www.hashtagtreinamentos.com/curso-python)")


@st.cache_data
def carregar_dados():
    tabela = pd.read_csv("resultados.csv")
    return tabela

with st.container():
    st.write("---")
    qtde_dias = st.selectbox("Selecione o período", ["7D", "15D", "21D", "30D"])
    num_dias = int(qtde_dias.replace("D", ""))
    dados = carregar_dados()
    dados = dados[-num_dias:]
    st.area_chart(dados, x="Data", y="Contratos")

# Título da Aplicação
st.title("Análise de Sentimentos com Naive Bayes")

# Entrada de Texto
text_input = st.text_area("Digite um comentário para análise:")

# Botão de Previsão
if st.button("Analisar Sentimento"):
    if text_input.strip():
        sentimento = prever_sentimento(text_input)
        st.success(f"Sentimento previsto: {sentimento}")
    else:
        st.warning("Por favor, insira um texto.")
