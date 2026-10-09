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

def filtrar_corridas_validas(df: pl.DataFrame, usar_fc: bool | None = None) -> pl.DataFrame:
    """
    Remove corridas com:
    - distância muito pequena
    - pace irreal
    - frequência cardíaca inválida (apenas quando a fonte tem FC)

    `usar_fc=None` detecta automaticamente. Sem essa checagem, uma fonte sem FC
    (ADR-018) zeraria o dataset: `FC > 40` em nulo resulta em nulo e descarta tudo.
    """
    if usar_fc is None:
        usar_fc = tem_frequencia_cardiaca(df)

    casts = [
        pl.col(COL_DISTANCE).cast(pl.Float64, strict=False),
        pl.col(COL_ELAPSED).cast(pl.Float64, strict=False),
    ]
    if COL_HR in df.columns:
        casts.append(pl.col(COL_HR).cast(pl.Float64, strict=False))

    df = df.with_columns(casts)

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
    condicao = (pl.col(COL_PACE) > 2) & (pl.col(COL_PACE) < 15)
    if usar_fc:
        condicao = condicao & (pl.col(COL_HR) > 40) & (pl.col(COL_HR) < 220)

    return df.filter(condicao)

def tem_frequencia_cardiaca(df: pl.DataFrame) -> bool:
    """
    Indica se a FC carrega informação usável: coluna presente, não toda nula
    e com variação. Fonte sem FC (ADR-018) ou preenchida com constante não passa.
    """
    if COL_HR not in df.columns:
        return False

    fc = df[COL_HR].cast(pl.Float64, strict=False)
    if fc.null_count() == df.height:
        return False

    return (fc.n_unique() > 1) and bool(fc.std() or 0.0)


def classificar_tipo_treino(df: pl.DataFrame, usar_fc: bool | None = None) -> pl.DataFrame:
    """
    Classifica cada corrida em: leve, longao, ritmo ou intervalado.
    Usa pace relativo ao atleta, distância e — quando disponível — FC.

    `usar_fc=None` detecta automaticamente. Em fontes sem FC (ADR-018) a regra de
    `intervalado` cai para o critério de pace apenas.
    """
    if usar_fc is None:
        usar_fc = tem_frequencia_cardiaca(df)

    agregados = [
        pl.col(COL_PACE).mean().over(COL_ATHLETE).alias("_pace_medio"),
        pl.col(COL_PACE).std().over(COL_ATHLETE).alias("_pace_std"),
        pl.col(COL_DISTANCE).quantile(LIMIAR_DIST_LONGAO).over(COL_ATHLETE).alias("_dist_p75"),
    ]
    if usar_fc:
        agregados.append(pl.col(COL_HR).median().over(COL_ATHLETE).alias("_hr_mediana"))

    df = df.with_columns(agregados)

    df = df.with_columns(
        pl.col("_pace_std").fill_null(0.5)
    )

    pace_rapido = (
        pl.col(COL_PACE) < pl.col("_pace_medio") - LIMIAR_PACE_INTERVALADO * pl.col("_pace_std")
    )
    eh_intervalado = (
        pace_rapido & (pl.col(COL_HR) >= pl.col("_hr_mediana")) if usar_fc else pace_rapido
    )

    df = df.with_columns(
        pl.when(pl.col(COL_DISTANCE) >= pl.col("_dist_p75"))
        .then(pl.lit("longao"))
        .when(eh_intervalado)
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

    auxiliares = ["_pace_medio", "_pace_std", "_dist_p75"]
    if usar_fc:
        auxiliares.append("_hr_mediana")
    return df.drop(auxiliares)

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
    """
    Pipeline fixo de ADR-003. A presença de FC é detectada uma única vez e
    propagada, para que fontes com e sem FC (ADR-018) sigam o mesmo caminho.
    """
    usar_fc = tem_frequencia_cardiaca(df)

    df = filtrar_genero_valido(df)
    if usar_fc:
        df = imputar_frequencia_cardiaca(df)
    df = filtrar_corridas_validas(df, usar_fc=usar_fc)
    df = classificar_tipo_treino(df, usar_fc=usar_fc)
    return df
