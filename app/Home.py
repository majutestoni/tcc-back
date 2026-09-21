import streamlit as st

st.set_page_config(page_title="TCC — Análise de Corridas", page_icon="🏃", layout="wide")

st.title("🏃 TCC — Análise e Recomendação de Treinos")

st.markdown(
    """
    Vitrine do serviço de ML (`tcc-back-2`) para análise de corridas e recomendação
    de treino.

    Use o menu lateral para navegar. Disponível agora:

    - **Recomendação de Treino** — escolha um atleta e veja a recomendação por
      regras e via Gemini.

    > Fonte de dados: `dataset/raw-data-kaggle.csv` (pré-processado).
    """
)

st.info(
    "Para a recomendação via Gemini, configure `GEMINI_API_KEY` no arquivo `.env` "
    "(ver `.env.example`) e instale `google-genai`."
)
