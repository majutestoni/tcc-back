from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st

from dataset.categorizar.definir_perfil_atleta import definir_perfil_atleta
from dataset.categorizar.perfil_atletas import montar_perfil_atletas
from dataset.constants import (
    COL_ATHLETE,
    COL_DISTANCE,
    COL_FREQUENCIA,
    COL_GENDER,
    COL_HR,
    COL_N_CORRIDAS,
    COL_PACE,
    COL_PERFIL_ATLETA,
    COL_TIPO_TREINO,
)
from dataset.models.previsao_treino.model import (
    META,
    calcular_deficit,
    distribuicao_por_atleta,
    filtrar_janela_temporal,
    ultimos_treinos_por_atleta,
)
from dataset.preprocessamento import formatar_pace, preprocessar

CSV_PADRAO = "dataset/raw-data-kaggle.csv"
ATLETAS_CSV = "output/atletas.csv"

CORES_TIPO = {
    "leve": "#4CAF50",
    "longao": "#2196F3",
    "ritmo": "#FF9800",
    "intervalado": "#F44336",
}


@st.cache_data(persist="disk")
def carregar_df(caminho: str = CSV_PADRAO) -> pl.DataFrame:
    return preprocessar(pl.read_csv(caminho, separator=";"))


@st.cache_data(persist="disk")
def listar_atletas(caminho: str = CSV_PADRAO) -> list[int]:
    df = carregar_df(caminho)
    return sorted(df[COL_ATHLETE].unique().to_list())


@st.cache_data(persist="disk")
def tabela_perfil_regras(caminho: str = CSV_PADRAO) -> pl.DataFrame:
    return definir_perfil_atleta(carregar_df(caminho), verbose=False)


def resumo_atleta(
    df: pl.DataFrame, perfil_regras: pl.DataFrame, athlete_id: int
) -> dict:
    perfil = montar_perfil_atletas(df).filter(pl.col(COL_ATHLETE) == athlete_id)
    if perfil.height == 0:
        return {}
    linha = perfil.row(0, named=True)
    linha_regras = perfil_regras.filter(pl.col(COL_ATHLETE) == athlete_id)
    label = linha_regras[COL_PERFIL_ATLETA][0] if linha_regras.height > 0 else "—"
    fc = linha.get(COL_HR)
    return {
        "genero": str(linha[COL_GENDER]).strip().upper(),
        "perfil": label,
        "n_corridas": int(linha[COL_N_CORRIDAS]),
        "frequencia_semana": round(float(linha[COL_FREQUENCIA]), 2),
        "pace_mediano": round(float(linha[COL_PACE]), 2),
        "distancia_mediana_km": round(float(linha[COL_DISTANCE]) / 1000, 2),
        "fc_mediana": round(float(fc), 1) if fc is not None else None,
    }


def distribuicao_vs_meta(
    df: pl.DataFrame, athlete_id: int
) -> tuple[dict, dict, dict, list[str]]:
    df_janela = filtrar_janela_temporal(df)
    distribuicao = distribuicao_por_atleta(df_janela).filter(
        pl.col(COL_ATHLETE) == athlete_id
    )
    atual = {
        item[COL_TIPO_TREINO]: round(item["percentual"], 4)
        for item in distribuicao.iter_rows(named=True)
    }
    deficits = {
        tipo: round(valor, 4) for tipo, valor in calcular_deficit(META, atual).items()
    }
    ultimos = ultimos_treinos_por_atleta(df_janela).get(athlete_id, [])
    return atual, META, deficits, ultimos


def grafico_distribuicao(atual: dict, meta: dict):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    tipos = list(meta.keys())
    valores_atual = [atual.get(t, 0.0) * 100 for t in tipos]
    valores_meta = [meta[t] * 100 for t in tipos]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(tipos))
    largura = 0.35
    ax.bar([i - largura / 2 for i in x], valores_atual, largura, label="Atual (%)",
           color=[CORES_TIPO.get(t, "gray") for t in tipos])
    ax.bar([i + largura / 2 for i in x], valores_meta, largura, label="Meta (%)",
           color="lightgray", edgecolor="gray")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tipos)
    ax.set_ylabel("Percentual (%)")
    ax.set_title("Distribuição atual vs meta (janela recente)")
    ax.legend()
    fig.tight_layout()
    return fig


def _construir_tabela_atletas(
    df: pl.DataFrame, perfil_regras: pl.DataFrame
) -> pl.DataFrame:
    perfil = montar_perfil_atletas(df)
    regras = perfil_regras.select([COL_ATHLETE, COL_PERFIL_ATLETA])
    perfil = perfil.join(regras, on=COL_ATHLETE, how="left")

    tabela = perfil.select(
        pl.col(COL_ATHLETE).alias("id"),
        pl.col(COL_GENDER).cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("genero"),
        pl.col(COL_PERFIL_ATLETA).fill_null("—").alias("perfil"),
        pl.col(COL_N_CORRIDAS).alias("corridas"),
        pl.col(COL_FREQUENCIA).round(2).alias("freq_semana"),
        (pl.col(COL_DISTANCE) / 1000).round(2).alias("dist_km"),
        pl.col(COL_PACE).round(2).alias("pace_min_km"),
        pl.col(COL_HR).round(1).alias("fc_bpm"),
    ).sort("corridas", descending=True)

    return tabela.with_columns(
        pl.Series("pace", [formatar_pace(v) for v in tabela["pace_min_km"].to_list()])
    )


@st.cache_data(persist="disk")
def tabela_atletas(caminho: str = CSV_PADRAO) -> pl.DataFrame:
    return _construir_tabela_atletas(
        carregar_df(caminho), tabela_perfil_regras(caminho)
    )


def exportar_planilha_atletas(
    caminho_saida: str = ATLETAS_CSV, caminho_csv: str = CSV_PADRAO
) -> str:
    tabela = _construir_tabela_atletas(
        carregar_df(caminho_csv), tabela_perfil_regras(caminho_csv)
    )
    Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
    tabela.write_csv(caminho_saida)
    return caminho_saida


def recomendacao_regras(df: pl.DataFrame, athlete_id: int) -> dict:
    from dataset.models.previsao_treino.model import treina_modelo

    resultado = treina_modelo(df, athlete_id=athlete_id, plotar=False, verbose=False)
    if resultado.recomendacoes is None or resultado.recomendacoes.height == 0:
        return {}
    return resultado.recomendacoes.row(0, named=True)


# ---------------------------------------------------------------------------
# Loaders para as páginas portadas dos notebooks (dataset, clusters, modelos).
# Imports pesados (sklearn/xgboost) ficam dentro das funções (ADR-012).
# ---------------------------------------------------------------------------


@st.cache_data(persist="disk")
def carregar_df_raw(caminho: str = CSV_PADRAO) -> pl.DataFrame:
    return pl.read_csv(caminho, separator=";")


@st.cache_data(persist="disk")
def perfil_multidim(caminho: str = CSV_PADRAO) -> pl.DataFrame:
    from dataset.analise import montar_perfil_multidim

    return montar_perfil_multidim(carregar_df(caminho))


@st.cache_data(persist="disk")
def clusters_por_genero(caminho: str = CSV_PADRAO) -> tuple[pl.DataFrame, dict]:
    from dataset.categorizar.kmeans import treina_kmeans_por_genero
    from dataset.categorizar.nomear_clusters import nomear_categorias

    resultado = treina_kmeans_por_genero(
        carregar_df(caminho),
        features=[COL_DISTANCE, COL_PACE],
        n_clusters=2,
        plotar=False,
        verbose=False,
    )
    perfil = nomear_categorias(resultado.df_exibir)
    return perfil, resultado.detalhes


@st.cache_data(persist="disk")
def tabela_clusters(caminho: str = CSV_PADRAO) -> pl.DataFrame:
    from dataset.categorizar.nomear_clusters import tabela_descritiva_clusters

    perfil, _ = clusters_por_genero(caminho)
    return tabela_descritiva_clusters(perfil)


@st.cache_data(persist="disk")
def fichas_atletas(caminho: str = CSV_PADRAO) -> pl.DataFrame:
    from dataset.categorizar.visualizacao_fichas import montar_fichas_atletas

    perfil, _ = clusters_por_genero(caminho)
    return montar_fichas_atletas(perfil)


# Combinações de features testadas para justificar a escolha do KMeans.
COMBINACOES_CLUSTER = {
    "dist_pace": [COL_DISTANCE, COL_PACE],
    "dist_pace_freq": [COL_DISTANCE, COL_PACE, COL_FREQUENCIA],
    "dist_pace_fc_freq": [COL_DISTANCE, COL_PACE, COL_HR, COL_FREQUENCIA],
    "dist_pace_fc_freq_ncorr": [
        COL_DISTANCE,
        COL_PACE,
        COL_HR,
        COL_FREQUENCIA,
        COL_N_CORRIDAS,
    ],
}


@st.cache_data(persist="disk")
def comparacao_features_k(
    caminho: str = CSV_PADRAO,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Grid combinação × k × gênero + sugestão automática (justifica dist_pace, k=2)."""
    from dataset.categorizar.comparar_features import (
        comparar_combinacoes_e_k,
        sugerir_melhor_config,
    )

    grid = comparar_combinacoes_e_k(
        carregar_df(caminho), COMBINACOES_CLUSTER, ks=(2, 3, 4)
    )
    sugestoes = sugerir_melhor_config(grid, min_cluster=5)
    return grid, sugestoes


@st.cache_data(persist="disk")
def grid_k_por_genero(caminho: str = CSV_PADRAO) -> dict[str, pl.DataFrame]:
    """Métricas internas para k=2..7 por gênero (justifica o k=2 final)."""
    from dataset.categorizar.perfil_atletas import montar_perfil_atletas
    from dataset.categorizar.validacao_cluster import avaliar_k

    perfil_base = montar_perfil_atletas(carregar_df(caminho)).with_columns(
        pl.col(COL_GENDER).cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    )
    grids: dict[str, pl.DataFrame] = {}
    for genero in ("M", "F"):
        subset = perfil_base.filter(pl.col(COL_GENDER) == genero)
        grids[genero] = avaliar_k(subset, [COL_DISTANCE, COL_PACE], ks=range(2, 8))
    return grids


@st.cache_data(persist="disk")
def validacao_clusters(caminho: str = CSV_PADRAO) -> dict:
    """Validação interna, estabilidade (ARI) e externa (vs perfil por regras)."""
    from dataset.categorizar.kmeans import treina_kmeans_por_genero
    from dataset.categorizar.nomear_clusters import nomear_categorias
    from dataset.categorizar.validacao_cluster import validar_resultado_kmeans

    df = carregar_df(caminho)
    features = [COL_DISTANCE, COL_PACE]
    resultado = treina_kmeans_por_genero(
        df, features=features, n_clusters=2, plotar=False, verbose=False
    )
    resultado.df_exibir = nomear_categorias(resultado.df_exibir)
    resultado = validar_resultado_kmeans(resultado, df=df, features=features)
    return resultado.detalhes


@st.cache_data(persist="disk")
def silhouette_amostras(caminho: str = CSV_PADRAO) -> pl.DataFrame:
    """Silhouette individual por atleta (identifica atletas na fronteira)."""
    from dataset.categorizar.validacao_cluster import silhouette_por_amostra

    perfil, _ = clusters_por_genero(caminho)
    return silhouette_por_amostra(perfil, features=[COL_DISTANCE, COL_PACE])


@st.cache_data(persist="disk")
def executar_modelos(caminho: str = CSV_PADRAO) -> dict:
    from dataset.models.previsao_pace.linear_regression import treina_linear_regression
    from dataset.models.previsao_pace.random_forest import treina_random_forest
    from dataset.models.previsao_pace.random_forest_regressor import (
        treina_random_forest_regressor,
    )
    from dataset.models.previsao_tempo.random_forest import treina_random_tempo
    from dataset.models.previsao_tempo.regressao_linear import treina_regressao_tempo
    from dataset.models.previsao_tempo.xboost import treina_xboost_tempo
    from dataset.models.previsao_treino.model import treina_modelo
    from dataset.models.previsao_treino.random_forest import treina_rf_recomendacao

    df = carregar_df(caminho)
    return {
        "lr_pace": treina_linear_regression(df, plotar=False, verbose=False),
        "rf_pace": treina_random_forest(df, plotar=False, verbose=False),
        "rf_pace_regressor": treina_random_forest_regressor(df, plotar=False, verbose=False),
        "lr_tempo": treina_regressao_tempo(df, plotar=False, verbose=False),
        "rf_tempo": treina_random_tempo(df, plotar=False, verbose=False),
        "xbt_tempo": treina_xboost_tempo(df, plotar=False, verbose=False),
        "regras": treina_modelo(df, plotar=False, verbose=False),
        "rf_treino": treina_rf_recomendacao(df, plotar=False, verbose=False),
    }
