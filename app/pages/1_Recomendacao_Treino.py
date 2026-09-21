import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.shared import (
    carregar_df,
    distribuicao_vs_meta,
    grafico_distribuicao,
    listar_atletas,
    recomendacao_regras,
    resumo_atleta,
    tabela_perfil_regras,
)

st.set_page_config(page_title="Recomendação de Treino", page_icon="🏃", layout="wide")
st.title("Recomendação de Treino")

entrada = st.text_input(
    "Informe o ID do atleta", value="", placeholder="ex.: 7947798"
)

if not entrada.strip():
    st.info("Informe o ID de um atleta para ver o perfil e a recomendação.")
    st.stop()

try:
    athlete_id = int(entrada.strip())
except ValueError:
    st.error("ID inválido. Informe um número inteiro.")
    st.stop()

if athlete_id not in listar_atletas():
    st.error(f"Atleta {athlete_id} não encontrado no dataset.")
    st.stop()

df = carregar_df()
perfil_regras = tabela_perfil_regras()

resumo = resumo_atleta(df, perfil_regras, athlete_id)
if not resumo:
    st.warning("Atleta sem dados suficientes após o preprocessamento.")
    st.stop()

st.subheader("Perfil do atleta")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Perfil (regras)", resumo["perfil"])
col1.metric("Gênero", resumo["genero"])
col2.metric("Nº de corridas", resumo["n_corridas"])
col2.metric("Frequência/semana", resumo["frequencia_semana"])
col3.metric("Pace mediano (min/km)", resumo["pace_mediano"])
col3.metric("Distância mediana (km)", resumo["distancia_mediana_km"])
col4.metric("FC mediana (bpm)", resumo["fc_mediana"] if resumo["fc_mediana"] else "—")

st.subheader("Distribuição de treinos vs meta")
atual, meta, deficits, ultimos = distribuicao_vs_meta(df, athlete_id)
col_g, col_d = st.columns([2, 1])
col_g.pyplot(grafico_distribuicao(atual, meta))
col_d.markdown("**Últimos treinos**")
col_d.write(ultimos or "—")
col_d.markdown("**Déficits vs meta**")
col_d.write({tipo: round(valor, 3) for tipo, valor in deficits.items()})

st.subheader("Recomendação por regras")
rec_regras = recomendacao_regras(df, athlete_id)
if rec_regras:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tipo", rec_regras["tipo"])
    c2.metric("Distância (km)", rec_regras["distancia_km"])
    c3.metric("Pace alvo (min/km)", rec_regras["pace_alvo_min_km"])
    c4.metric("Tempo estimado (min)", rec_regras["tempo_estimado_min"])
    st.caption(rec_regras["descricao"])
else:
    st.info("Sem recomendação por regras para este atleta.")

st.divider()
st.subheader("Recomendação via Gemini")

if st.button("Recomendar via Gemini", type="primary"):
    with st.spinner("Consultando Gemini..."):
        try:
            from gemini.recomendacao import recomendar_treino_gemini

            rec = recomendar_treino_gemini(df, athlete_id)
        except RuntimeError as erro:
            st.error(str(erro))
            st.stop()
        except Exception as erro:  # noqa: BLE001
            st.error(f"Erro ao chamar o Gemini: {erro}")
            st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tipo", rec.tipo)
    c2.metric("Distância (km)", rec.distancia_km)
    c3.metric("Pace alvo (min/km)", rec.pace_alvo_min_km)
    c4.metric("Tempo estimado (min)", rec.tempo_estimado_min)
    st.markdown(f"**{rec.descricao}**")
    st.markdown("**Justificativa**")
    st.write(rec.justificativa)
    if rec.alertas:
        st.markdown("**Alertas**")
        for alerta in rec.alertas:
            st.warning(alerta)
