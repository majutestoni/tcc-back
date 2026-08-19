from __future__ import annotations

import polars as pl

from dataset.constants import (
    COL_ATHLETE,
    COL_DISTANCE,
    COL_FREQUENCIA,
    COL_GENDER,
    COL_HR,
    COL_N_CORRIDAS,
    COL_NIVEL_DISTANCIA,
    COL_NIVEL_PACE,
    COL_NIVEL_VOLUME,
    COL_PACE,
    COL_PERFIL_ATLETA,
    COL_TIMESTAMP,
    LIMIAR_DIST_CURTA_KM,
    LIMIAR_DIST_LONGA_KM,
    LIMIAR_VOLUME_ALTO,
    LIMIAR_VOLUME_BAIXO,
    MIN_CORRIDAS_PERFIL,
)


def montar_perfil_atletas(df: pl.DataFrame) -> pl.DataFrame:
    """
    Agrega métricas por atleta a partir de distância, pace e timestamp.

    Retorna: athlete, gender, distance (m) mediana, pace mediano,
    FC mediana, frequencia_semana e n_corridas.
    """
    df_ts = df.with_columns(
        pl.col(COL_TIMESTAMP)
        .str.strptime(pl.Datetime, format="%d/%m/%Y %H:%M", strict=False)
        .alias("_ts")
    )

    return (
        df_ts.group_by(COL_ATHLETE)
        .agg(
            pl.col(COL_DISTANCE).median().alias(COL_DISTANCE),
            pl.col(COL_PACE).median().alias(COL_PACE),
            pl.col(COL_HR).median().alias(COL_HR),
            pl.col("_ts").min().alias("_ts_min"),
            pl.col("_ts").max().alias("_ts_max"),
            pl.len().alias(COL_N_CORRIDAS),
            pl.col(COL_GENDER).first().alias(COL_GENDER),
        )
        .with_columns(
            (
                pl.col(COL_N_CORRIDAS)
                / (
                    (pl.col("_ts_max") - pl.col("_ts_min")).dt.total_days() / 7.0
                ).clip(lower_bound=1.0)
            ).alias(COL_FREQUENCIA)
        )
        .drop("_ts_min", "_ts_max")
        .drop_nulls([COL_DISTANCE, COL_PACE, COL_HR, COL_FREQUENCIA])
    )


def categorizar_por_pace(perfil: pl.DataFrame) -> pl.DataFrame:
    """Tercis de pace mediano: rapido | intermediario | recreativo."""
    p33 = perfil[COL_PACE].quantile(0.33)
    p66 = perfil[COL_PACE].quantile(0.66)

    return perfil.with_columns(
        pl.when(pl.col(COL_PACE) <= p33)
        .then(pl.lit("rapido"))
        .when(pl.col(COL_PACE) <= p66)
        .then(pl.lit("intermediario"))
        .otherwise(pl.lit("recreativo"))
        .alias(COL_NIVEL_PACE)
    )


def categorizar_por_volume(perfil: pl.DataFrame) -> pl.DataFrame:
    """Frequência semanal: baixo | moderado | alto."""
    return perfil.with_columns(
        pl.when(pl.col(COL_FREQUENCIA) < LIMIAR_VOLUME_BAIXO)
        .then(pl.lit("baixo"))
        .when(pl.col(COL_FREQUENCIA) < LIMIAR_VOLUME_ALTO)
        .then(pl.lit("moderado"))
        .otherwise(pl.lit("alto"))
        .alias(COL_NIVEL_VOLUME)
    )


def categorizar_por_distancia(perfil: pl.DataFrame) -> pl.DataFrame:
    """Distância mediana: curta | media | longa."""
    dist_km = pl.col(COL_DISTANCE) / 1000
    return perfil.with_columns(
        pl.when(dist_km < LIMIAR_DIST_CURTA_KM)
        .then(pl.lit("curta"))
        .when(dist_km < LIMIAR_DIST_LONGA_KM)
        .then(pl.lit("media"))
        .otherwise(pl.lit("longa"))
        .alias(COL_NIVEL_DISTANCIA)
    )


def definir_perfil_composto(perfil: pl.DataFrame) -> pl.DataFrame:
    """
    Combina os três eixos em um perfil legível:
    avancado | fundista | iniciante | intermediario
    """
    return perfil.with_columns(
        pl.when(
            (pl.col(COL_NIVEL_PACE) == "rapido")
            & (pl.col(COL_NIVEL_VOLUME) == "alto")
        )
        .then(pl.lit("avancado"))
        .when(
            (pl.col(COL_NIVEL_DISTANCIA) == "longa")
            & (pl.col(COL_NIVEL_PACE).is_in(["rapido", "intermediario"]))
        )
        .then(pl.lit("fundista"))
        .when(
            (pl.col(COL_NIVEL_DISTANCIA) == "curta")
            & (pl.col(COL_NIVEL_PACE) == "recreativo")
            & (pl.col(COL_NIVEL_VOLUME) == "baixo")
        )
        .then(pl.lit("iniciante"))
        .otherwise(pl.lit("intermediario"))
        .alias(COL_PERFIL_ATLETA)
    )


def definir_perfil_atleta(
    df: pl.DataFrame,
    min_corridas: int | None = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Categoriza atletas por regras interpretáveis (pace, volume, distância).

    Retorna um DataFrame com uma linha por atleta e as colunas:
    nivel_pace, nivel_volume, nivel_distancia, perfil_atleta.
    """
    if min_corridas is None:
        min_corridas = MIN_CORRIDAS_PERFIL

    perfil = montar_perfil_atletas(df)

    if min_corridas > 0:
        perfil = perfil.filter(pl.col(COL_N_CORRIDAS) >= min_corridas)

    perfil = categorizar_por_pace(perfil)
    perfil = categorizar_por_volume(perfil)
    perfil = categorizar_por_distancia(perfil)
    perfil = definir_perfil_composto(perfil)
    perfil = perfil.sort(COL_PERFIL_ATLETA, COL_PACE)

    if verbose:
        print(f"Atletas categorizados: {perfil.height} (mín. {min_corridas} corridas)")
        print("\nPor perfil:")
        print(
            perfil.group_by(COL_PERFIL_ATLETA)
            .agg(pl.len().alias("atletas"))
            .sort("atletas", descending=True)
        )
        print("\nPor nível de pace:")
        print(
            perfil.group_by(COL_NIVEL_PACE)
            .agg(pl.len().alias("atletas"))
            .sort("atletas", descending=True)
        )
        print("\nPor volume:")
        print(
            perfil.group_by(COL_NIVEL_VOLUME)
            .agg(pl.len().alias("atletas"))
            .sort("atletas", descending=True)
        )
        print("\nPor distância:")
        print(
            perfil.group_by(COL_NIVEL_DISTANCIA)
            .agg(pl.len().alias("atletas"))
            .sort("atletas", descending=True)
        )

    return perfil
