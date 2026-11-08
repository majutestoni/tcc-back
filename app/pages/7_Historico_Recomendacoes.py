import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl
import streamlit as st

from gemini.historico import (
    RECOMENDACOES_CSV,
    alertas_para_lista,
    carregar_historico,
)

st.set_page_config(
    page_title="Histórico de Recomendações", page_icon="🗂️", layout="wide"
)
st.title("Histórico de recomendações")
st.caption(
    "Recomendações de treino geradas pela IA (Gemini), salvas em "
    f"`{RECOMENDACOES_CSV}`."
)

historico = carregar_historico()

if historico.height == 0:
    st.info(
        "Nenhuma recomendação salva ainda. Gere uma na página "
        "**Recomendação de Treino**."
    )
    st.stop()

col_f1, col_f2 = st.columns(2)
busca = col_f1.text_input("Buscar por ID do atleta", value="", placeholder="ex.: 5151970")
tipos_disponiveis = sorted(historico["tipo"].unique().to_list())
tipos = col_f2.multiselect("Tipo de treino", tipos_disponiveis, default=tipos_disponiveis)

filtrado = historico.filter(pl.col("tipo").is_in(tipos))
if busca.strip():
    if busca.strip().isdigit():
        filtrado = filtrado.filter(pl.col("athlete_id") == int(busca.strip()))
    else:
        st.warning("Digite apenas números no campo de busca por ID.")

st.write(f"**{filtrado.height}** recomendações")
st.dataframe(
    filtrado.select(
        pl.col("athlete_id").alias("atleta"),
        pl.col("timestamp").alias("quando"),
        "origem",
        "tipo",
        pl.col("distancia_km").alias("dist_km"),
        pl.col("pace_alvo_min_km").alias("pace_alvo"),
        pl.col("tempo_estimado_min").alias("tempo_min"),
        pl.col("descricao"),
    ),
    width="stretch",
    hide_index=True,
)

st.divider()
st.subheader("Detalhe")
for linha in filtrado.reverse().iter_rows(named=True):
    titulo = (
        f"Atleta {linha['athlete_id']} · {linha['tipo']} · "
        f"{linha['distancia_km']} km · {linha['timestamp']}"
    )
    with st.expander(titulo):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tipo", linha["tipo"])
        c2.metric("Distância (km)", linha["distancia_km"])
        c3.metric("Pace alvo (min/km)", linha["pace_alvo_min_km"])
        c4.metric("Tempo estimado (min)", linha["tempo_estimado_min"])
        st.markdown(f"**{linha['descricao']}**")
        st.markdown("**Justificativa**")
        st.write(linha["justificativa"])
        alertas = alertas_para_lista(linha["alertas"])
        if alertas:
            st.markdown("**Alertas**")
            for alerta in alertas:
                st.warning(alerta)

st.divider()
st.download_button(
    "Baixar histórico (CSV)",
    data=historico.write_csv(),
    file_name="recomendacoes.csv",
    mime="text/csv",
)
