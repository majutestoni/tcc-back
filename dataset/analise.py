from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path

from dataset.categorizar.perfil_atletas import montar_perfil_atletas
from dataset.constants import *
from dataset.preprocessamento import formatar_pace

CORES_TIPO = {
    "leve": "#4CAF50",
    "longao": "#2196F3",
    "ritmo": "#FF9800",
    "intervalado": "#F44336",
}
ORDEM_TIPO = ["leve", "longao", "ritmo", "intervalado"]
XLIM_DISTANCIA_KM = (0, 50)

FEATURES_MULTIDIM = [
    {"col": COL_DISTANCE, "label": "Distância mediana", "invert": False},
    {"col": COL_PACE, "label": "Pace mediano", "invert": True},
    {"col": COL_HR, "label": "FC mediana", "invert": False},
    {"col": COL_FREQUENCIA, "label": "Freq. semanal", "invert": False},
    {"col": COL_N_CORRIDAS, "label": "Nº corridas", "invert": False},
    {"col": COL_ELEVATION, "label": "Elevação mediana", "invert": False},
]


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


def montar_perfil_multidim(df: pl.DataFrame) -> pl.DataFrame:
    """Perfil agregado por atleta com features usadas na visualização multidimensional."""
    perfil = montar_perfil_atletas(df)
    if COL_ELEVATION in df.columns:
        elev = df.group_by(COL_ATHLETE).agg(
            pl.col(COL_ELEVATION).median().alias(COL_ELEVATION)
        )
        perfil = perfil.join(elev, on=COL_ATHLETE, how="left")
    return perfil


def _features_disponiveis(perfil: pl.DataFrame) -> list[dict]:
    return [f for f in FEATURES_MULTIDIM if f["col"] in perfil.columns]


def _percentil_rank(valores: np.ndarray, valor: float) -> float:
    if valores.size == 0:
        return 50.0
    return float(np.mean(valores <= valor) * 100.0)


def _formatar_valor_feature(col: str, valor: float) -> str:
    if col == COL_DISTANCE:
        return f"{valor / 1000:.2f} km"
    if col == COL_PACE:
        return formatar_pace(valor)
    if col == COL_ELEVATION:
        return f"{valor:.0f} m"
    if col in (COL_FREQUENCIA, COL_N_CORRIDAS):
        return f"{valor:.2f}" if col == COL_FREQUENCIA else f"{int(valor)}"
    return f"{valor:.1f}"


def tabela_percentis_atleta(perfil: pl.DataFrame, athlete_id: int) -> pl.DataFrame:
    """
    Percentil do atleta em cada feature (0–100; maior = melhor posição relativa).

    O ranking usa exatamente o DataFrame `perfil` passado: amostra global ou
    só os atletas do mesmo cluster.
    """
    subset = perfil.filter(pl.col(COL_ATHLETE) == athlete_id)
    if subset.height == 0:
        raise ValueError(f"Atleta {athlete_id} não encontrado no perfil.")

    row = subset.row(0, named=True)
    linhas: list[dict] = []

    for feat in _features_disponiveis(perfil):
        col = feat["col"]
        valor = row[col]
        if valor is None:
            continue
        valores = perfil[col].drop_nulls().to_numpy()
        percentil = _percentil_rank(valores, float(valor))
        if feat["invert"]:
            percentil = 100.0 - percentil
        linhas.append({
            "feature": feat["label"],
            "valor": _formatar_valor_feature(col, float(valor)),
            "percentil": round(percentil, 1),
            "posicao": f"P{round(percentil):.0f}",
            "n_referencia": int(valores.size),
        })

    return pl.DataFrame(linhas)


def _normalizar_feature(valores: np.ndarray, invert: bool = False) -> np.ndarray:
    vmin = float(np.nanmin(valores))
    vmax = float(np.nanmax(valores))
    if vmax - vmin < 1e-9:
        norm = np.zeros_like(valores, dtype=float)
    else:
        norm = (valores - vmin) / (vmax - vmin)
    return 1.0 - norm if invert else norm


def plotar_radar_percentil(
    perfil: pl.DataFrame,
    athlete_id: int,
    ax: Axes | None = None,
    comparar_media: bool = True,
    cor_atleta: str = "#2E86AB",
) -> Axes:
    """Gráfico radar com percentil do atleta em cada dimensão (100 = topo da amostra)."""
    tabela = tabela_percentis_atleta(perfil, athlete_id)
    if tabela.height == 0:
        raise ValueError("Nenhuma feature disponível para o radar.")

    labels = tabela["feature"].to_list()
    valores = tabela["percentil"].to_numpy()
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    valores_plot = valores.tolist()
    angles_plot = angles.copy()

    valores_plot.append(valores_plot[0])
    angles_plot.append(angles_plot[0])

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7)
    ax.grid(alpha=0.3)

    if comparar_media:
        media = np.full(n, 50.0)
        media_plot = media.tolist() + [media[0]]
        ax.plot(angles_plot, media_plot, color="#AAAAAA", lw=1.5, ls="--", label="Mediana da amostra (P50)")
        ax.fill(angles_plot, media_plot, color="#AAAAAA", alpha=0.08)

    ax.plot(angles_plot, valores_plot, color=cor_atleta, lw=2, marker="o", label=f"Atleta {athlete_id}")
    ax.fill(angles_plot, valores_plot, color=cor_atleta, alpha=0.2)
    ax.set_title(f"Posição relativa — atleta {athlete_id}", fontsize=11, pad=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)

    return ax


def plotar_coordenadas_paralelas(
    perfil: pl.DataFrame,
    destacar_ids: int | list[int] | None = None,
    max_fundo: int = 120,
    seed: int = 42,
    ax: Axes | None = None,
) -> Axes:
    """Coordenadas paralelas: cada linha é um atleta; destaque para IDs escolhidos."""
    features = _features_disponiveis(perfil)
    if not features:
        raise ValueError("Nenhuma feature disponível para coordenadas paralelas.")

    cols = [f["col"] for f in features]
    labels = [f["label"] for f in features]
    core_cols = [c for c in (COL_DISTANCE, COL_PACE, COL_HR, COL_FREQUENCIA, COL_N_CORRIDAS) if c in cols]
    dados = perfil.select([COL_ATHLETE, *cols]).drop_nulls(subset=core_cols)
    if COL_ELEVATION in cols:
        mediana_elev = dados[COL_ELEVATION].median()
        dados = dados.with_columns(pl.col(COL_ELEVATION).fill_null(mediana_elev))
    if dados.height == 0:
        raise ValueError("Perfil sem dados válidos para coordenadas paralelas.")

    if destacar_ids is None:
        ids_destaque: list[int] = []
    elif isinstance(destacar_ids, int):
        ids_destaque = [destacar_ids]
    else:
        ids_destaque = list(destacar_ids)

    mat = np.zeros((dados.height, len(cols)))
    for j, feat in enumerate(features):
        mat[:, j] = _normalizar_feature(
            dados[feat["col"]].to_numpy(),
            invert=feat["invert"],
        )

    athlete_ids = dados[COL_ATHLETE].to_numpy()
    x = np.arange(len(cols))
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    fundo_idx = np.arange(dados.height)
    if dados.height > max_fundo:
        rng = np.random.default_rng(seed)
        fundo_idx = rng.choice(dados.height, size=max_fundo, replace=False)

    for i in fundo_idx:
        if athlete_ids[i] in ids_destaque:
            continue
        ax.plot(x, mat[i], color="#CCCCCC", alpha=0.35, lw=0.8, zorder=1)

    cores_destaque = ["#2E86AB", "#E84855", "#1B7F4E", "#E09F3E", "#6A4C93"]
    for idx, athlete in enumerate(ids_destaque):
        matches = np.where(athlete_ids == athlete)[0]
        if matches.size == 0:
            continue
        i = int(matches[0])
        cor = cores_destaque[idx % len(cores_destaque)]
        ax.plot(
            x,
            mat[i],
            color=cor,
            lw=2.5,
            marker="o",
            markersize=5,
            zorder=3,
            label=f"Atleta {athlete}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Posição normalizada (0 = menor · 1 = maior)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Coordenadas paralelas — perfil multidimensional")
    ax.grid(axis="y", alpha=0.2)
    if ids_destaque:
        ax.legend(loc="upper right", fontsize=8)

    return ax


def plotar_painel_posicao_atleta(
    perfil: pl.DataFrame,
    athlete_id: int,
    max_fundo: int = 120,
    escopo: str = "amostra global",
) -> Figure:
    """
    Painel com radar (percentis) + coordenadas paralelas para um atleta.

    `perfil` é o grupo de referência do ranking: amostra inteira ou só o cluster.
    """
    fig = plt.figure(figsize=(16, 6))
    ax_radar = fig.add_subplot(121, polar=True)
    ax_par = fig.add_subplot(122)

    plotar_radar_percentil(perfil, athlete_id, ax=ax_radar)
    plotar_coordenadas_paralelas(
        perfil, destacar_ids=athlete_id, max_fundo=max_fundo, ax=ax_par
    )

    genero = perfil.filter(pl.col(COL_ATHLETE) == athlete_id)[COL_GENDER][0]
    genero_txt = "Masculino" if str(genero).upper() == "M" else "Feminino"
    fig.suptitle(
        f"Posição relativa ({escopo}) — atleta {athlete_id} ({genero_txt})",
        fontsize=13,
    )
    plt.tight_layout()
    return fig
