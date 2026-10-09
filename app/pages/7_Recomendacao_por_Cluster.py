import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.shared import (
    carregar_df,
    distribuicao_vs_meta,
    distribuicao_vs_meta_cluster,
    grafico_distribuicao,
    listar_atletas,
    recomendacao_cluster_atleta,
    recomendacao_regras,
    resumo_atleta,
    tabela_perfil_regras,
)
from dataset.constants import SEMANAS_JANELA

st.set_page_config(
    page_title="Recomendação por Cluster", page_icon="🧩", layout="wide"
)
st.title("Recomendação por Cluster")
st.caption(
    "Informe o ID do atleta. A recomendação usa a meta e os parâmetros do "
    "perfil KMeans (`volume_alto` / `volume_moderado`), não a meta global."
)

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

rec = recomendacao_cluster_atleta(athlete_id)
atual, meta_c, deficits, ultimos, info_c = distribuicao_vs_meta_cluster(df, athlete_id)

st.subheader("Perfil do atleta")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Perfil (regras)", resumo["perfil"])
col1.metric(
    "Cluster",
    rec.get("rotulo_cluster", info_c.get("rotulo_cluster", "—")) if rec else "—",
)
col2.metric("Gênero", resumo["genero"])
col2.metric("Nº de corridas", resumo["n_corridas"])
col3.metric("Frequência/semana", resumo["frequencia_semana"])
col3.metric("Pace mediano (min/km)", resumo["pace_mediano"])
col4.metric("Distância mediana (km)", resumo["distancia_mediana_km"])
col4.metric("FC mediana (bpm)", resumo["fc_mediana"] if resumo["fc_mediana"] else "—")

if info_c.get("perfil_cluster"):
    st.caption(
        f"Perfil cluster: `{info_c['perfil_cluster']}` · "
        f"janela das últimas {SEMANAS_JANELA} semanas"
    )

st.subheader("Distribuição de treinos vs meta do cluster")
col_g, col_d = st.columns([2, 1])
col_g.pyplot(grafico_distribuicao(atual, meta_c))
col_d.markdown("**Últimos treinos**")
col_d.write(ultimos or "—")
col_d.markdown("**Déficits vs meta do cluster**")
col_d.write({tipo: round(valor, 3) for tipo, valor in deficits.items()})
col_d.markdown("**Meta do cluster**")
col_d.write({tipo: f"{pct * 100:.0f}%" for tipo, pct in meta_c.items()})

st.subheader("Próximo treino recomendado")
if not rec:
    st.info("Sem recomendação por cluster para este atleta.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tipo", rec["tipo"])
c2.metric("Distância (km)", rec["distancia_km"])
c3.metric("Pace alvo (min/km)", rec["pace_alvo_min_km"])
c4.metric("Tempo estimado (min)", rec["tempo_estimado_min"])
st.caption(rec.get("descricao", ""))
st.success(
    f"**{rec['tipo'].upper()}** · {rec['distancia_km']} km · "
    f"pace {rec['pace_alvo_min_km']} min/km · "
    f"~{rec['tempo_estimado_min']:.0f} min"
)

with st.expander("Comparar com a recomendação global (mesma meta para todos)"):
    atual_g, meta_g, deficits_g, _ = distribuicao_vs_meta(df, athlete_id)
    rec_global = recomendacao_regras(df, athlete_id)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Déficit vs meta global**")
        st.pyplot(grafico_distribuicao(atual_g, meta_g))
        if rec_global:
            st.write(
                {
                    "tipo": rec_global["tipo"],
                    "distancia_km": rec_global["distancia_km"],
                    "pace_alvo_min_km": rec_global["pace_alvo_min_km"],
                    "tempo_estimado_min": rec_global["tempo_estimado_min"],
                }
            )
    with col_b:
        st.markdown("**Por cluster**")
        st.pyplot(grafico_distribuicao(atual, meta_c))
        st.write(
            {
                "tipo": rec["tipo"],
                "distancia_km": rec["distancia_km"],
                "pace_alvo_min_km": rec["pace_alvo_min_km"],
                "tempo_estimado_min": rec["tempo_estimado_min"],
            }
        )
