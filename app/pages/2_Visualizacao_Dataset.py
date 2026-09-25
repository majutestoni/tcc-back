import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import streamlit as st

from app.shared import CSV_PADRAO, carregar_df, carregar_df_raw, perfil_multidim
from dataset.analise import (
    moda_distancia_km,
    plotar_coordenadas_paralelas,
    plotar_distribuicao_tipo_treino,
    plotar_histograma_distancia,
    plotar_pace_por_genero,
    plotar_painel_eda,
    plotar_painel_modelos,
    plotar_painel_posicao_atleta,
    resumo_preprocessamento,
    tabela_percentis_atleta,
    tabela_resumo,
)
from dataset.constants import COL_ATHLETE

st.set_page_config(page_title="Visualização do Dataset", page_icon="📊", layout="wide")

st.title("📊 Visualização do Dataset")
st.caption(
    "Exploração do dataset Strava: resumo estatístico, impacto da normalização "
    "e gráficos exploratórios. Portado do notebook `visualizacao_dataset.ipynb`."
)

df = carregar_df()
df_raw = carregar_df_raw()

# --- Resumo estatístico -----------------------------------------------------
st.header("Resumo estatístico")
st.dataframe(tabela_resumo(df).to_pandas(), width="stretch", hide_index=True)

moda_km, moda_qtd = moda_distancia_km(df)
st.metric("Moda de distância", f"{moda_km} km", f"{moda_qtd} corridas")

with st.expander("Amostra dos dados (df.head)"):
    st.dataframe(df.head().to_pandas(), width="stretch")
with st.expander("Estatísticas descritivas (df.describe)"):
    st.dataframe(df.describe().to_pandas(), width="stretch")

# --- Impacto da normalização ------------------------------------------------
st.header("Impacto da normalização")
st.caption("Quantas corridas foram removidas pelo preprocessamento determinístico (ADR-003).")
st.dataframe(
    resumo_preprocessamento(df_raw, df).to_pandas(), width="stretch", hide_index=True
)

# --- Painel exploratório ----------------------------------------------------
st.header("Painel exploratório (6 gráficos)")
st.pyplot(plotar_painel_eda(df))

# --- Relações usadas nos modelos --------------------------------------------
st.header("Relações usadas nos modelos")
st.pyplot(plotar_painel_modelos(df))

# --- Gráficos individuais ---------------------------------------------------
st.header("Destaques do dataset")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plotar_histograma_distancia(df, ax=axes[0])
plotar_distribuicao_tipo_treino(df, ax=axes[1])
plotar_pace_por_genero(df, ax=axes[2])
fig.tight_layout()
st.pyplot(fig)

# --- Posição multidimensional do atleta -------------------------------------
st.header("Posição multidimensional do atleta")
st.markdown(
    """
    Cada atleta é resumido em **6 dimensões** (mediana por feature): distância,
    pace, FC, frequência semanal, nº de corridas e elevação.

    - **Radar (percentil):** quanto o atleta está acima da amostra em cada eixo
      (0–100; maior = melhor). Pace é invertido (pace menor = posição melhor).
    - **Coordenadas paralelas:** cada linha é um atleta; valores normalizados
      entre 0 e 1 por feature.
    """
)

perfil = perfil_multidim()
ids = perfil[COL_ATHLETE].to_list()
athlete_id = st.selectbox("Atleta", ids, index=0)

st.subheader(f"Posição relativa — atleta {athlete_id}")
st.dataframe(
    tabela_percentis_atleta(perfil, athlete_id).to_pandas(),
    width="stretch",
    hide_index=True,
)
st.pyplot(plotar_painel_posicao_atleta(perfil, athlete_id))

st.subheader("Comparar atletas nas coordenadas paralelas")
default_ids = perfil.sample(min(3, perfil.height), seed=42)[COL_ATHLETE].to_list()
ids_comparar = st.multiselect("Atletas para destacar", ids, default=default_ids)
if ids_comparar:
    fig, ax = plt.subplots(figsize=(12, 5))
    plotar_coordenadas_paralelas(perfil, destacar_ids=ids_comparar, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
