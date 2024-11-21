import streamlit as st
import joblib

# Configurações do título da página
st.set_page_config(page_title="Análise de Sentimento", page_icon="🔍", layout="centered")

# Título e descrição do aplicativo
st.title("🔍 Análise de Sentimento de Comentários")
st.write(
    """
    Este aplicativo utiliza Machine Learning para prever o sentimento de um comentário.
    Basta inserir o texto e clicar em **Analisar Sentimento** para ver o resultado.
    """
)

# Separador elegante
st.markdown("---")

# Carregar o modelo Naive Bayes
model = joblib.load('modelo_naive_bayes.pkl')

# Carregar o vetorizador
vectorizer = joblib.load('vectorizer.pkl')

# Entrada de Texto
st.markdown("### ✍️ Digite o comentário para análise:")
text_input = st.text_area(
    "Insira o comentário aqui:", 
    placeholder="Exemplo: O produto é incrível e superou minhas expectativas!"
)

# Botão de Previsão
if st.button("Analisar Sentimento"):
    if text_input.strip():
        # Transformar o texto e prever o sentimento
        sentimento_vec = vectorizer.transform([text_input])  # Passar como lista
        sentimento_pred = model.predict(sentimento_vec)

        # Exibir resultado com formatação
        st.markdown("#### 🎯 Resultado da Análise:")
        if sentimento_pred[0] == "Positivo":  # Ajuste baseado na classificação do modelo
            st.success(f"Sentimento Previsto: **Positivo** 😊")
        elif sentimento_pred[0] == "Negativo":
            st.error(f"Sentimento Previsto: **Negativo** 😠")
        else:
            st.info(f"Sentimento Previsto: **Neutro** 😐")
    else:
        st.warning("⚠️ Por favor, insira um texto para análise.")

# Rodapé ou separador final
st.markdown("---")
st.markdown("**Criado por [Andrey Alves](https://github.com/dreymond1)** 🚀")
