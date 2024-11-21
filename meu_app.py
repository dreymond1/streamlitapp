import streamlit as st
import joblib

# Carregar o modelo Naive Bayes
model = joblib.load('modelo_naive_bayes.pkl')

# Carregar o vetorizador
vectorizer = joblib.load('vectorizer.pkl')

# Entrada de Texto
text_input = st.text_area("Digite um comentário para análise:")

# Botão de Previsão
if st.button("Analisar Sentimento"):
    if text_input.strip():
        sentimento = text_input
        sentimento_vec = vectorizer.transform(sentimento)
        sentimento_pred = model.predict(sentimento_vec)
        st.success(f"Sentimento previsto: {sentimento_pred}")
    else:
        st.warning("Por favor, insira um texto.")


