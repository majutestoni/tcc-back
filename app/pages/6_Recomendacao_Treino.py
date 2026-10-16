import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import polars as pl
import streamlit as st

from app.shared import (
    carregar_df,
    corridas_janela_atleta,
    distribuicao_vs_meta_cluster,
    fichas_atletas,
    grafico_distribuicao,
    grafico_janela_deslizante,
    listar_atletas,
    parametros_por_perfil_cluster,
    perfil_multidim_com_cluster,
    recomendacao_cluster_atleta,
    resumo_atleta,
    tabela_perfil_regras,
)
from dataset.analise import plotar_painel_posicao_atleta, tabela_percentis_atleta
from dataset.categorizar.visualizacao_fichas import plotar_ficha_atleta
from dataset.constants import COL_ATHLETE, SEMANAS_JANELA
from dataset.models.previsao_treino.recomendacao_cluster import explicar_recomendacao
from dataset.preprocessamento import formatar_pace

st.set_page_config(page_title="Recomendação de Treino", page_icon="🏃", layout="wide")
st.title("Recomendação de Treino")
st.caption(
    "Informe o ID do atleta. A recomendação usa a meta e os parâmetros do "
    "perfil KMeans (`volume_alto` / `volume_moderado`)."
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
col3.metric("Pace mediano", formatar_pace(resumo["pace_mediano"]))
col4.metric("Distância mediana (km)", resumo["distancia_mediana_km"])
col4.metric("FC mediana (bpm)", resumo["fc_mediana"] if resumo["fc_mediana"] else "—")

if info_c.get("perfil_cluster"):
    st.caption(
        f"Perfil cluster: `{info_c['perfil_cluster']}` · "
        f"janela das últimas {SEMANAS_JANELA} semanas"
    )

perfil = perfil_multidim_com_cluster()
linha_atleta = perfil.filter(pl.col(COL_ATHLETE) == athlete_id)
tem_cluster = (
    linha_atleta.height > 0
    and linha_atleta["categoria"][0] is not None
)

st.subheader("Posição relativa")
st.caption(
    "Percentil 0–100 (maior = melhor posição). Pace é invertido "
    "(pace menor = posição melhor). Comparado com a **amostra global** e com o "
    "**mesmo cluster** do atleta."
)

aba_global, aba_cluster = st.tabs(["Na amostra global", "No cluster"])

with aba_global:
    st.markdown("**Referência:** todos os atletas da base.")
    st.dataframe(
        tabela_percentis_atleta(perfil, athlete_id).to_pandas(),
        width="stretch",
        hide_index=True,
    )
    st.pyplot(
        plotar_painel_posicao_atleta(
            perfil, athlete_id, escopo="amostra global"
        )
    )

with aba_cluster:
    if not tem_cluster:
        st.info("Atleta sem categoria de cluster para calcular percentis no grupo.")
    else:
        categoria = linha_atleta["categoria"][0]
        rotulo = linha_atleta["rotulo_cluster"][0] or categoria
        perfil_cluster = perfil.filter(pl.col("categoria") == categoria)
        st.markdown(
            f"**Referência:** atletas do cluster `{rotulo}` "
            f"({perfil_cluster.height} atletas)."
        )
        st.dataframe(
            tabela_percentis_atleta(perfil_cluster, athlete_id).to_pandas(),
            width="stretch",
            hide_index=True,
        )
        st.pyplot(
            plotar_painel_posicao_atleta(
                perfil_cluster,
                athlete_id,
                max_fundo=max(perfil_cluster.height, 1),
                escopo=f"cluster · {rotulo}",
            )
        )

st.subheader("Ficha do cluster")
st.caption(
    "Card do atleta na categoria KMeans (cluster × gênero): ranking de pace "
    "dentro do grupo e posição na barra (esquerda = melhor pace)."
)
fichas = fichas_atletas()
if fichas.filter(pl.col(COL_ATHLETE) == athlete_id).height == 0:
    st.info("Atleta sem ficha de cluster (poucos dados para a categorização).")
else:
    _, col_ficha, _ = st.columns([1, 1, 1])
    with col_ficha:
        st.pyplot(plotar_ficha_atleta(fichas, athlete_id, figsize=(4.2, 3.4)))

st.subheader("Janela deslizante — últimas corridas")
st.caption(
    f"O algoritmo **não** olha um número fixo de corridas. Ele usa as "
    f"**últimas {SEMANAS_JANELA} semanas** (a partir da última corrida do "
    f"atleta), conta quantas foram leve/longão/ritmo/intervalado e recomenda "
    f"o tipo com maior déficit vs a meta do cluster. O anti-fadiga olha o "
    f"**último** treino dessa janela."
)
corridas_janela = corridas_janela_atleta(df, athlete_id)
if corridas_janela.height == 0:
    st.warning(
        f"Nenhuma corrida nas últimas {SEMANAS_JANELA} semanas para este atleta."
    )
else:
    st.pyplot(grafico_janela_deslizante(corridas_janela))
    tipos_seq = corridas_janela["tipo"].to_list()
    st.caption(
        f"**{corridas_janela.height} corridas** na janela de "
        f"{SEMANAS_JANELA} semanas · sequência: {' → '.join(tipos_seq)}"
    )
    with st.expander("Tabela das corridas na janela"):
        st.dataframe(corridas_janela.to_pandas(), width="stretch", hide_index=True)

st.subheader("Distribuição de treinos vs meta do cluster")
st.caption(
    "**Déficit** = meta do cluster − % atual na janela. Valor positivo = tipo "
    "em falta; a recomendação prioriza o maior déficit."
)
col_g, col_d = st.columns([2, 1])
col_g.pyplot(grafico_distribuicao(atual, meta_c))
col_d.markdown("**Últimos tipos na janela**")
col_d.write(ultimos or "—")
col_d.markdown("**Déficits vs meta do cluster**")
col_d.write({tipo: round(valor, 3) for tipo, valor in deficits.items()})
col_d.markdown("**Meta do cluster**")
col_d.write({tipo: f"{pct * 100:.0f}%" for tipo, pct in meta_c.items()})

st.subheader("Próximo treino recomendado")
if not rec:
    st.info("Sem recomendação por cluster para este atleta.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tipo", rec["tipo"])
    c2.metric("Distância (km)", rec["distancia_km"])
    c3.metric("Pace alvo", formatar_pace(rec["pace_alvo_min_km"]))
    c4.metric("Tempo estimado (min)", rec["tempo_estimado_min"])
    st.caption(rec.get("descricao", ""))
    st.success(
        f"**{rec['tipo'].upper()}** · {rec['distancia_km']} km · "
        f"pace {formatar_pace(rec['pace_alvo_min_km'])} · "
        f"~{rec['tempo_estimado_min']:.0f} min"
    )

    chave_perfil = info_c.get("perfil_cluster") or rec.get("perfil_cluster")
    params_perfil = parametros_por_perfil_cluster()
    parametros_tipo = None
    if chave_perfil and chave_perfil in params_perfil:
        parametros_tipo = params_perfil[chave_perfil].get(rec["tipo"])

    justificativa = explicar_recomendacao(
        rec=rec,
        atual=atual,
        meta=meta_c,
        deficits=deficits,
        ultimos=ultimos,
        corridas_janela=corridas_janela,
        info_cluster=info_c,
        parametros_tipo=parametros_tipo,
    )
    if justificativa:
        st.markdown("#### Por que esta recomendação?")
        st.markdown(justificativa)

st.divider()
st.subheader("Recomendação via Gemini")

if st.button("Recomendar via Gemini", type="primary"):
    with st.spinner("Consultando Gemini..."):
        try:
            from gemini.recomendacao import recomendar_treino_gemini

            rec_gemini = recomendar_treino_gemini(df, athlete_id)
        except RuntimeError as erro:
            st.error(str(erro))
            st.stop()
        except Exception as erro:  # noqa: BLE001
            st.error(f"Erro ao chamar o Gemini: {erro}")
            st.stop()

    from gemini.historico import salvar_recomendacao

    salvar_recomendacao(athlete_id, rec_gemini, origem="gemini")
    if rec:
        salvar_recomendacao(athlete_id, rec, origem="cluster")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tipo", rec_gemini.tipo)
    c2.metric("Distância (km)", rec_gemini.distancia_km)
    c3.metric("Pace alvo", formatar_pace(rec_gemini.pace_alvo_min_km))
    c4.metric("Tempo estimado (min)", rec_gemini.tempo_estimado_min)
    st.markdown(f"**{rec_gemini.descricao}**")
    st.markdown("**Justificativa**")
    st.write(rec_gemini.justificativa)
    if rec_gemini.alertas:
        st.markdown("**Alertas**")
        for alerta in rec_gemini.alertas:
            st.warning(alerta)
    st.success(
        "Recomendações (Gemini e por cluster) salvas no Histórico de recomendações."
    )
