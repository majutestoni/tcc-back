import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import streamlit as st

from app.shared import CORES_TIPO, executar_modelos
from dataset.constants import COL_ATHLETE
from dataset.models.previsao_treino.model import plotar_recomendacao_atleta
from dataset.models.visualizacao import (
    plotar_matriz_confusao,
    plotar_scatter_regressao,
    tabela_metricas,
)

st.set_page_config(page_title="Comparação de Modelos", page_icon="📈", layout="wide")

st.title("📈 Comparação de Modelos")
st.markdown(
    """
    Executa todos os modelos (previsão de pace, tempo e recomendação de treino),
    com métricas e gráficos comparativos. Portado do notebook `comparacao_modelos.ipynb`.
    """
)

with st.spinner("Treinando modelos (cacheado após a 1ª execução)…"):
    resultados = executar_modelos()

st.success(f"Modelos executados: {', '.join(resultados.keys())}")


def _barras_metricas(nomes, valores, titulo, cor, ylim=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(nomes, valores, color=cor)
    ax.set_title(titulo)
    ax.tick_params(axis="x", rotation=15)
    if ylim:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    return fig


def _detalhes_modelo(res):
    """Reconstrói `exibir_detalhes_modelo` em componentes Streamlit."""
    with st.expander(f"Detalhes — {res.nome}"):
        st.dataframe(
            pl.DataFrame(
                {"metrica": list(res.metricas.keys()),
                 "valor": [round(float(v), 4) for v in res.metricas.values()]}
            ).to_pandas(),
            width="stretch",
            hide_index=True,
        )
        if "coeficientes" in res.detalhes:
            st.markdown("**Coeficientes**")
            st.json(res.detalhes["coeficientes"])
            st.caption(f"Intercepto: {res.detalhes.get('intercepto')}")
        if "importancias" in res.detalhes:
            imp = res.detalhes["importancias"]
            st.markdown("**Importâncias das features**")
            st.dataframe(
                pl.DataFrame({"feature": list(imp.keys()),
                              "importancia": [round(float(v), 4) for v in imp.values()]})
                .sort("importancia", descending=True)
                .to_pandas(),
                width="stretch",
                hide_index=True,
            )
        if "previsao_exemplo" in res.detalhes:
            st.caption(f"Previsão exemplo: {res.detalhes['previsao_exemplo']}")
        if res.y_true_cls is not None and res.class_names is not None:
            from sklearn.metrics import classification_report

            rep = classification_report(
                res.y_true_cls, res.y_pred_cls,
                target_names=res.class_names, digits=3, output_dict=True,
            )
            linhas_rep = []
            for classe, vals in rep.items():
                if isinstance(vals, dict):
                    linhas_rep.append({"classe": classe,
                                       **{k: round(float(v), 3) for k, v in vals.items()}})
                else:
                    linhas_rep.append({"classe": classe, "precision": None, "recall": None,
                                       "f1-score": round(float(vals), 3), "support": None})
            st.markdown("**Relatório de classificação**")
            st.dataframe(pl.DataFrame(linhas_rep).to_pandas(),
                         width="stretch", hide_index=True)


# --- Tabela comparativa -----------------------------------------------------
st.header("Tabela comparativa geral")
st.dataframe(tabela_metricas(list(resultados.values())).to_pandas(), width="stretch", hide_index=True)


# --- Regressões (pace e tempo) ----------------------------------------------
def secao_regressao(chaves, titulo, xlabel, ylabel):
    st.header(titulo)
    modelos = [resultados[c] for c in chaves]

    fig, axes = plt.subplots(1, len(modelos), figsize=(5 * len(modelos), 5))
    if len(modelos) == 1:
        axes = [axes]
    for ax, res in zip(axes, modelos):
        plotar_scatter_regressao(res, xlabel, ylabel, ax=ax)
    fig.suptitle(titulo, fontsize=14)
    fig.tight_layout()
    st.pyplot(fig)

    nomes = [r.nome for r in modelos]
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(_barras_metricas(nomes, [r.metricas.get("mae", 0) for r in modelos], "MAE (menor = melhor)", "steelblue"))
    with col2:
        st.pyplot(_barras_metricas(nomes, [r.metricas.get("r2", 0) for r in modelos], "R² (maior = melhor)", "coral"))

    for res in modelos:
        _detalhes_modelo(res)


secao_regressao(
    ["lr_pace", "rf_pace", "rf_pace_regressor"],
    "Previsão de Pace", "Pace real (min/km)", "Pace previsto (min/km)",
)
secao_regressao(
    ["lr_tempo", "rf_tempo", "xbt_tempo"],
    "Previsão de Tempo", "Tempo real (s)", "Tempo previsto (s)",
)

# --- Classificação (recomendação de treino) ---------------------------------
st.header("Recomendação de treino — Regras vs Random Forest")
treino = [resultados["regras"], resultados["rf_treino"]]

col1, col2 = st.columns(2)
with col1:
    st.pyplot(_barras_metricas(
        [r.nome for r in treino],
        [r.metricas.get("erro_distribuicao_medio", 0) for r in treino],
        "Erro médio de distribuição vs meta (menor = melhor)", "steelblue",
    ))
with col2:
    rf = resultados["rf_treino"]
    if "f1_macro" in rf.metricas:
        st.pyplot(_barras_metricas([rf.nome], [rf.metricas["f1_macro"]], "F1-macro (classificação RF)", "coral", ylim=(0, 1)))

rf = resultados["rf_treino"]
if rf.y_true_cls is not None:
    st.subheader("Matriz de confusão — Random Forest")
    fig, ax = plt.subplots(figsize=(5, 5))
    plotar_matriz_confusao(rf, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)

_detalhes_modelo(rf)

# --- Teste de recomendação (Regras e RF) ------------------------------------
st.header("Teste de recomendação — atual vs meta por atleta")
st.caption(
    "Para cada modelo: parâmetros calibrados, gráfico da distribuição atual vs "
    "meta do 1º atleta e amostra das recomendações."
)
for res in treino:
    if res.recomendacoes is None or res.recomendacoes.height == 0:
        continue
    st.subheader(res.nome)

    if res.parametros_calibrados:
        params = [
            {"tipo": tipo, "distancia_km": p["distancia_km"],
             "pace_offset_min": round(p["pace_offset_min"], 2)}
            for tipo, p in res.parametros_calibrados.items()
        ]
        st.markdown("**Parâmetros calibrados (mediana por tipo)**")
        st.dataframe(pl.DataFrame(params).to_pandas(), width="stretch", hide_index=True)

    erro_medio = res.metricas.get("erro_distribuicao_medio")
    if erro_medio is not None:
        st.metric("Erro médio de distribuição vs meta", round(erro_medio, 3))

    athlete_id = res.recomendacoes[COL_ATHLETE][0]
    plotar_recomendacao_atleta(
        res.df_exibir, res.recomendacoes,
        athlete_id=athlete_id, col_tipo=res.col_tipo or "tipo_treino",
    )
    st.pyplot(plt.gcf())
    plt.close("all")

    st.dataframe(
        res.recomendacoes.select(
            COL_ATHLETE, "tipo", "descricao", "distancia_km",
            "pace_alvo_min_km", "tempo_estimado_min", "erro_distribuicao",
        ).head(5).to_pandas(),
        width="stretch",
        hide_index=True,
    )

# --- Comparação de tipos recomendados ---------------------------------------
st.subheader("Tipos recomendados — Regras vs RF")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, res in zip(axes, treino):
    if res.recomendacoes is None:
        continue
    cont = res.recomendacoes.group_by("tipo").agg(pl.len().alias("qtd")).sort("qtd", descending=True)
    tipos = cont["tipo"].to_list()
    ax.bar(tipos, cont["qtd"].to_list(), color=[CORES_TIPO.get(t, "gray") for t in tipos])
    ax.set_title(res.nome)
    ax.set_ylabel("Nº de atletas")
fig.suptitle("Comparação — tipos recomendados", fontsize=13)
fig.tight_layout()
st.pyplot(fig)

# --- Resumo -----------------------------------------------------------------
st.header("Resumo — melhor modelo por categoria")
linhas = []
for categoria, chave_metrica, menor_melhor in [
    ("pace", "mae", True), ("tempo", "mae", True), ("treino", "erro_distribuicao_medio", True),
]:
    grupo = [r for r in resultados.values() if r.categoria == categoria and chave_metrica in r.metricas]
    if not grupo:
        continue
    melhor = min(grupo, key=lambda r: r.metricas[chave_metrica])
    linhas.append({
        "categoria": categoria,
        "melhor_modelo": melhor.nome,
        chave_metrica: round(melhor.metricas[chave_metrica], 4),
    })
st.dataframe(pl.DataFrame(linhas).to_pandas(), width="stretch", hide_index=True)
