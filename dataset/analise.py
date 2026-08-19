from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path

from dataset.constants import *

CORES_TIPO = {
    "leve": "#4CAF50",
    "longao": "#2196F3",
    "ritmo": "#FF9800",
    "intervalado": "#F44336",
}
ORDEM_TIPO = ["leve", "longao", "ritmo", "intervalado"]
XLIM_DISTANCIA_KM = (0, 50)


def carregar_dataset(caminho: Path) -> pl.DataFrame:
    return pl.read_csv(caminho, separator=";")


def resumo_dataset(df: pl.DataFrame) -> dict[str, float | int]:
    moda_km, moda_qtd = moda_distancia_km(df)
    return {
        "registros": df.height,
        "atletas": df[COL_ATHLETE].n_unique(),
        "distancia_km_media": round((df[COL_DISTANCE] / 1000).mean(), 2),
        "distancia_km_mediana": round((df[COL_DISTANCE] / 1000).median(), 2),
        "pace_mediano": round(df[COL_PACE].median(), 2),
        "fc_mediana": round(df[COL_HR].median(), 1),
        "moda_distancia_km": moda_km,
        "moda_distancia_qtd": moda_qtd,
    }


def tabela_resumo(df: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame([resumo_dataset(df)])


def resumo_preprocessamento(df_raw: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame([
        {"etapa": "Dataset bruto", "registros": df_raw.height},
        {"etapa": "Após preprocessamento", "registros": df.height},
        {
            "etapa": "Registros removidos",
            "registros": df_raw.height - df.height,
        },
    ])


def moda_distancia_km(df: pl.DataFrame) -> tuple[float, int]:
    dist = (df[COL_DISTANCE] / 1000).round()
    contagem = dist.value_counts().sort("count", descending=True)
    return contagem[0, dist.name], contagem[0, "count"]


def plotar_histograma_distancia(
    df: pl.DataFrame,
    ax: Axes | None = None,
    xlim: tuple[float, float] | None = XLIM_DISTANCIA_KM,
    bins: int | np.ndarray | None = None,
) -> Axes:
    distancia_km = (df[COL_DISTANCE] / 1000).to_numpy()
    distancia_km = distancia_km[distancia_km <= XLIM_DISTANCIA_KM[1]]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if bins is None:
        bins_arr = np.arange(XLIM_DISTANCIA_KM[0], XLIM_DISTANCIA_KM[1] + 1, 1)
    else:
        bins_arr = bins

    ax.hist(
        distancia_km,
        bins=bins_arr,
        color="#2E86AB",
        edgecolor="white",
        alpha=0.85,
    )
    ax.set_xlabel("Distância (km)")
    ax.set_ylabel("Quantidade de corridas")
    ax.set_title("Distribuição da distância (0–50 km)")
    ax.grid(axis="y", alpha=0.1)
    if xlim is not None:
        ax.set_xlim(xlim)

    return ax


def plotar_histograma_pace(df: pl.DataFrame, ax: Axes | None = None) -> Axes:
    pace = df[COL_PACE].to_numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.hist(pace, bins=30, color="#2E86AB", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Pace (min/km)")
    ax.set_ylabel("Quantidade de corridas")
    ax.set_title("Distribuição do pace")
    ax.grid(axis="y", alpha=0.1)

    return ax


def plotar_histograma_fc(df: pl.DataFrame, ax: Axes | None = None) -> Axes:
    fc = df[COL_HR].to_numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.hist(fc, bins=30, color="#E84855", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Frequência cardíaca (bpm)")
    ax.set_ylabel("Quantidade de corridas")
    ax.set_title("Distribuição da FC")
    ax.grid(axis="y", alpha=0.1)

    return ax


def plotar_distribuicao_tipo_treino(df: pl.DataFrame, ax: Axes | None = None) -> Axes:
    if COL_TIPO_TREINO not in df.columns:
        raise ValueError(f"Coluna '{COL_TIPO_TREINO}' não encontrada. Use preprocessar(df) antes.")

    contagem = df.group_by(COL_TIPO_TREINO).agg(pl.len().alias("qtd"))
    qtd_por_tipo = {
        row[COL_TIPO_TREINO]: row["qtd"] for row in contagem.iter_rows(named=True)
    }
    tipos = [t for t in ORDEM_TIPO if t in qtd_por_tipo] + [
        t for t in qtd_por_tipo if t not in ORDEM_TIPO
    ]
    qtds = [qtd_por_tipo[t] for t in tipos]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        tipos,
        qtds,
        color=[CORES_TIPO.get(t, "gray") for t in tipos],
        edgecolor="white",
    )
    ax.set_xlabel("Tipo de treino")
    ax.set_ylabel("Quantidade de corridas")
    ax.set_title("Distribuição por tipo de treino")
    ax.grid(axis="y", alpha=0.1)

    return ax


def plotar_pace_por_genero(df: pl.DataFrame, ax: Axes | None = None) -> Axes:
    df_plot = df.select(COL_GENDER, COL_PACE).drop_nulls()
    df_plot = df_plot.with_columns(
        pl.col(COL_GENDER).cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    )

    generos = ["M", "F"]
    dados = [
        df_plot.filter(pl.col(COL_GENDER) == g)[COL_PACE].to_numpy()
        for g in generos
    ]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(
        dados,
        tick_labels=generos,
        patch_artist=True,
        showfliers=False,
    )
    cores_genero = {"M": "#2E86AB", "F": "#E84855"}
    for patch, genero in zip(bp["boxes"], generos):
        patch.set_facecolor(cores_genero.get(genero, "gray"))
        patch.set_alpha(0.75)

    ax.set_xlabel("Gênero")
    ax.set_ylabel("Pace (min/km)")
    ax.set_title("Pace por gênero")
    ax.grid(axis="y", alpha=0.1)

    return ax


def plotar_corridas_por_atleta(df: pl.DataFrame, ax: Axes | None = None) -> Axes:
    por_atleta = (
        df.group_by(COL_ATHLETE)
        .agg(pl.len().alias("qtd"))
        .sort("qtd")
    )
    qtds = por_atleta["qtd"].to_numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.hist(qtds, bins=20, color="#6A4C93", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Corridas por atleta")
    ax.set_ylabel("Quantidade de atletas")
    ax.set_title("Volume de treinos por atleta")
    ax.grid(axis="y", alpha=0.1)

    return ax


def plotar_pace_vs_distancia(df: pl.DataFrame, ax: Axes | None = None) -> Axes:
    if COL_TIPO_TREINO not in df.columns:
        raise ValueError(f"Coluna '{COL_TIPO_TREINO}' não encontrada. Use preprocessar(df) antes.")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    df_plot = df.filter(pl.col(COL_DISTANCE) <= XLIM_DISTANCIA_KM[1] * 1000)

    for tipo in ORDEM_TIPO:
        subset = df_plot.filter(pl.col(COL_TIPO_TREINO) == tipo)
        if subset.height == 0:
            continue
        ax.scatter(
            subset[COL_DISTANCE] / 1000,
            subset[COL_PACE],
            alpha=0.2,
            s=8,
            c=CORES_TIPO.get(tipo, "gray"),
            label=tipo,
        )

    ax.set_xlabel("Distância (km)")
    ax.set_ylabel("Pace (min/km)")
    ax.set_title("Pace × distância (por tipo)")
    ax.set_xlim(XLIM_DISTANCIA_KM)
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.1)

    return ax


def plotar_painel_eda(df: pl.DataFrame, titulo: str = "Visão geral do dataset") -> Figure:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    plotar_histograma_distancia(df, ax=axes[0, 0])
    plotar_histograma_pace(df, ax=axes[0, 1])
    plotar_histograma_fc(df, ax=axes[0, 2])
    plotar_distribuicao_tipo_treino(df, ax=axes[1, 0])
    plotar_pace_por_genero(df, ax=axes[1, 1])
    plotar_corridas_por_atleta(df, ax=axes[1, 2])

    fig.suptitle(titulo, fontsize=14)
    plt.tight_layout()
    return fig


def plotar_painel_modelos(df: pl.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plotar_pace_vs_distancia(df, ax=axes[0])
    dist_km = (df.filter(pl.col(COL_DISTANCE) <= XLIM_DISTANCIA_KM[1] * 1000)[COL_DISTANCE] / 1000).to_numpy()
    tempo_min = (
        df.filter(pl.col(COL_DISTANCE) <= XLIM_DISTANCIA_KM[1] * 1000)[COL_ELAPSED] / 60
    ).to_numpy()
    axes[1].scatter(dist_km, tempo_min, alpha=0.15, s=8, c="#2E86AB")
    axes[1].set_xlabel("Distância (km)")
    axes[1].set_ylabel("Tempo (min)")
    axes[1].set_title("Tempo × distância")
    axes[1].set_xlim(XLIM_DISTANCIA_KM)
    axes[1].grid(alpha=0.1)

    fig.suptitle("Relações usadas nos modelos de previsão", fontsize=14)
    plt.tight_layout()
    return fig
