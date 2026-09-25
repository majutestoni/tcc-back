import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl
import streamlit as st

from app.shared import ATLETAS_CSV, exportar_planilha_atletas, tabela_atletas

st.set_page_config(page_title="Atletas", page_icon="🏃", layout="wide")
st.title("Atletas")
st.caption(
    "Consulte o ID e o perfil dos atletas. Use o ID na página "
    "**Recomendação de Treino**."
)

tabela = tabela_atletas()

col_f1, col_f2, col_f3 = st.columns(3)
generos = col_f1.multiselect("Gênero", ["M", "F"], default=["M", "F"])
perfis_disponiveis = sorted(tabela["perfil"].unique().to_list())
perfis = col_f2.multiselect("Perfil", perfis_disponiveis, default=perfis_disponiveis)
min_corridas = col_f3.slider(
    "Mínimo de corridas", 0, int(tabela["corridas"].max()), 0
)

filtrada = tabela.filter(
    pl.col("genero").is_in(generos)
    & pl.col("perfil").is_in(perfis)
    & (pl.col("corridas") >= min_corridas)
)

busca = st.text_input("Buscar por ID", value="", placeholder="ex.: 7947798")
if busca.strip():
    if busca.strip().isdigit():
        filtrada = filtrada.filter(pl.col("id") == int(busca.strip()))
    else:
        st.warning("Digite apenas números no campo de busca por ID.")

st.write(f"**{filtrada.height}** atletas")
st.dataframe(
    filtrada.select(
        "id", "genero", "perfil", "corridas", "freq_semana", "dist_km", "pace", "fc_bpm"
    ),
    width="stretch",
    hide_index=True,
)

st.divider()
col_d1, col_d2 = st.columns(2)
col_d1.download_button(
    "Baixar planilha (CSV)",
    data=tabela.write_csv(),
    file_name="atletas.csv",
    mime="text/csv",
)
if col_d2.button(f"Salvar planilha em {ATLETAS_CSV}"):
    caminho = exportar_planilha_atletas()
    st.success(f"Planilha salva em {caminho}")
