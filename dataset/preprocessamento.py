import polars as pl
from .constants import *


def filtrar_genero_valido(df: pl.DataFrame, col_gender: str = COL_GENDER) -> pl.DataFrame:
    """
    Remove linhas em que gender não é M ou F.
    """
    return df.filter(
        pl.col(col_gender)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_uppercase()
        .is_in(["M", "F"])
    )

def imputar_frequencia_cardiaca(
    df: pl.DataFrame, mediana_treino: float | None = None
) -> pl.DataFrame:
    """
    Converte FC para numérico e preenche ausentes com a mediana.
    Se `mediana_treino` for informada, usa esse valor.
    """

    df = df.with_columns(
        pl.col(COL_HR).cast(pl.Float64, strict=False)
    )

    med = mediana_treino if mediana_treino is not None else df[COL_HR].median()

    if med is None:
        return df

    df = df.with_columns(
        pl.col(COL_HR).fill_null(med)
    )

    return df

def filtrar_corridas_validas(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove corridas com:
    - distância muito pequena
    - pace irreal
    - frequência cardíaca inválida
    """

    df = df.with_columns([
        pl.col(COL_DISTANCE).cast(pl.Float64, strict=False),
        pl.col(COL_ELAPSED).cast(pl.Float64, strict=False),
        pl.col(COL_HR).cast(pl.Float64, strict=False),
    ])

    # Remove distâncias menores que 1 km
    df = df.filter(
        pl.col(COL_DISTANCE) > 1000
    )

    # Cria pace em min/km
    df = df.with_columns(
        (
            (pl.col(COL_ELAPSED) / 60) /
            (pl.col(COL_DISTANCE) / 1000)
        ).alias(COL_PACE)
    )

    # Remove valores irreais
    df = df.filter(
        (pl.col(COL_PACE) > 2) &
        (pl.col(COL_PACE) < 15) &
        (pl.col(COL_HR) > 40) &
        (pl.col(COL_HR) < 220)
    )

    return df

def classificar_tipo_treino(df: pl.DataFrame) -> pl.DataFrame:
    """
    Classifica cada corrida em: leve, longao, ritmo ou intervalado.
    Usa pace relativo ao atleta, distância e FC.
    """
    df = df.with_columns([
        pl.col(COL_PACE).mean().over(COL_ATHLETE).alias("_pace_medio"),
        pl.col(COL_PACE).std().over(COL_ATHLETE).alias("_pace_std"),
        pl.col(COL_DISTANCE).quantile(LIMIAR_DIST_LONGAO).over(COL_ATHLETE).alias("_dist_p75"),
        pl.col(COL_HR).median().over(COL_ATHLETE).alias("_hr_mediana"),
    ])

    df = df.with_columns(
        pl.col("_pace_std").fill_null(0.5)
    )

    df = df.with_columns(
        pl.when(pl.col(COL_DISTANCE) >= pl.col("_dist_p75"))
        .then(pl.lit("longao"))
        .when(
            (pl.col(COL_PACE) < pl.col("_pace_medio") - LIMIAR_PACE_INTERVALADO * pl.col("_pace_std"))
            & (pl.col(COL_HR) >= pl.col("_hr_mediana"))
        )
        .then(pl.lit("intervalado"))
        .when(pl.col(COL_PACE) > pl.col("_pace_medio") + LIMIAR_PACE_LEVE * pl.col("_pace_std"))
        .then(pl.lit("leve"))
        .when(
            (pl.col(COL_PACE) >= pl.col("_pace_medio") - LIMIAR_PACE_RITMO * pl.col("_pace_std"))
            & (pl.col(COL_PACE) <= pl.col("_pace_medio") + LIMIAR_PACE_RITMO * pl.col("_pace_std"))
        )
        .then(pl.lit("ritmo"))
        .otherwise(pl.lit("leve"))
        .alias(COL_TIPO_TREINO)
    )

    return df.drop("_pace_medio", "_pace_std", "_dist_p75", "_hr_mediana")

def formatar_pace(pace_decimal: float, sufixo: str = "/km") -> str:
    """Converte pace decimal (ex.: 6.67) para min:seg (ex.: 6:40/km)."""
    minutos = int(pace_decimal)
    segundos = int(round((pace_decimal - minutos) * 60))
    if segundos == 60:
        minutos += 1
        segundos = 0
    return f"{minutos}:{segundos:02d}{sufixo}"


def formatar_delta_pace(delta_decimal: float) -> str:
    """Formata diferença de pace decimal como +min:seg ou +Ns."""
    sinal = "+" if delta_decimal >= 0 else "-"
    delta = abs(delta_decimal)
    minutos = int(delta)
    segundos = int(round((delta - minutos) * 60))
    if segundos == 60:
        minutos += 1
        segundos = 0
    if minutos == 0:
        return f"{sinal}{segundos}s"
    return f"{sinal}{minutos}:{segundos:02d}"


def preprocessar(df: pl.DataFrame) -> pl.DataFrame:
    df = filtrar_genero_valido(df)
    df = imputar_frequencia_cardiaca(df)
    df = filtrar_corridas_validas(df)
    df = classificar_tipo_treino(df)
    return df
