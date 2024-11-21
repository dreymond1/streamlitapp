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
        # Corrigido: Encapsular o texto em uma lista
        sentimento_vec = vectorizer.transform([text_input])  # Passar como lista
        sentimento_pred = model.predict(sentimento_vec)
        st.success(f"Sentimento previsto: {sentimento_pred[0]}")  # Mostrar o resultado como string
    else:
        st.warning("Por favor, insira um texto.")
