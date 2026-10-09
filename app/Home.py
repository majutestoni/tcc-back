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
    - **Recomendação de Treino** — escolha um atleta e veja a recomendação por
      regras, por cluster e via Gemini.
    - **Recomendação por Cluster** — informe o ID do atleta e receba o próximo
      treino com meta/parâmetros do perfil KMeans.
    - **Histórico de Recomendações** — recomendações da IA persistidas.
    - **🔬 Protocolo Experimental** — validação da recomendação (déficit vs meta
      global / do cluster vs baselines, matriz de confusão, amostra de casos) e,
      em seguida, ablação e modelos auxiliares sob particionamento por atleta.
    - **🎯 Desfecho e Meta** — validação externa contra um desfecho real (melhor
      tempo de meia maratona) e a meta de distribuição estimada dos dados em vez
      de afirmada.

    > As telas de Visualização, Comparação e Categorização são portes dos
    > notebooks de `notebooks/`.

    > Fontes de dados: `dataset/raw-data-kaggle.csv` (primária, pré-processada) e
    > a base externa de validação `long-distance-running-dataset`, usada apenas
    > nas duas últimas telas.
    """
)

st.info(
    "As métricas da tela **Comparação de Modelos** usam o protocolo original "
    "(split aleatório por corrida), mantido por paridade com os notebooks. Os "
    "números estatisticamente defensáveis estão em **Protocolo Experimental**."
)

st.info(
    "Para a recomendação via Gemini, configure `GEMINI_API_KEY` no arquivo `.env` "
    "(ver `.env.example`) e instale `google-genai`."
)
