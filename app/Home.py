import streamlit as st

st.set_page_config(page_title="TCC — Análise de Corridas", page_icon="🏃", layout="wide")

st.title("🏃 TCC — Análise e Recomendação de Treinos")

st.markdown(
    """
    Vitrine do serviço de ML (`tcc-back-2`) para análise de corridas e recomendação
    de treino.

    Use o menu lateral para navegar. Disponível agora:

    - **Visualização do Dataset** — resumo, normalização e gráficos exploratórios.
    - **Comparação de Modelos** — métricas e gráficos de todos os modelos.
    - **Categorização de Corredores — KMeans** — clusters por gênero e fichas.
    - **Atletas** — tabela de atletas (id, gênero, perfil, métricas) com filtros.
    - **Recomendação de Treino** — informe o ID: posição relativa, ficha do
      cluster, distribuição vs meta do cluster, próximo treino e Gemini.
    - **Histórico de Recomendações** — recomendações da IA persistidas.
    - **Validação da recomendação** — *a regra funciona?* Eficácia por
      desfecho (aderência × tempo de 21 km); imitação só como contraste;
      ablação e modelos auxiliares.

    > As telas de Visualização, Comparação e Categorização são portes dos
    > notebooks de `notebooks/`.

    > Fonte de dados da demo: `dataset/raw-data-kaggle.csv` (pré-processada).
    """
)

st.info(
    "As métricas da tela **Comparação de Modelos** usam o protocolo original "
    "(split aleatório por corrida), mantido por paridade com os notebooks. Os "
    "números estatisticamente defensáveis estão em **Validação da recomendação**."
)

st.info(
    "Para a recomendação via Gemini, configure `GEMINI_API_KEY` no arquivo `.env` "
    "(ver `.env.example`) e instale `google-genai`."
)
