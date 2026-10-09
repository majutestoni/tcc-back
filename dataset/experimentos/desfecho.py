"""
Extração de desfecho de prova (ADR-018).

Este é o módulo que dá ao projeto o que ele não tinha: uma **variável de
desfecho**. Até aqui a recomendação de treino era avaliada contra a própria
premissa (o erro em relação à `META`), o que é circular. Com um desfecho é
possível perguntar se um padrão de treino se associa a *desempenho melhor*.

Dois desfechos, com papéis diferentes:

- **Meia maratona (21,0975 km) — principal.** O foco do projeto é o corredor
  amador (ADR seção 1), e 21 km é a distância de prova mais praticada nesse
  público. Definido como o **melhor tempo do atleta na distância** no período,
  ou seja um teto de capacidade demonstrada.
- **Maratona (42,195 km) — checagem de robustez.** Mais fraco em relevância
  para o público-alvo, porém mais forte em validade: a coluna `major` da fonte
  externa diz qual maratona o atleta correu, e cruzando com a data oficial da
  prova o esforço é uma **corrida de verdade**, não um treino longo.

Limitação registrada: o melhor esforço de 21 km pode ter sido um treino
intenso e não uma prova. É um proxy de capacidade, não um resultado oficial.
A concordância entre os dois desfechos é o que dá confiança no proxy.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from dataset.constants import (
    COL_ATHLETE,
    COL_DATA_PROVA,
    COL_DISTANCE,
    COL_DISTANCIA_PROVA,
    COL_ELAPSED,
    COL_GENDER,
    COL_PACE,
    COL_PACE_PROVA,
    COL_TEMPO_PROVA,
    COL_TIMESTAMP,
    COL_TIPO_TREINO,
    DISTANCIA_MARATONA_KM,
    DISTANCIA_MEIA_KM,
    MIN_TREINOS_PREPARO,
    SEMANAS_PREPARO,
    TOLERANCIA_PROVA_KM,
)
from dataset.models.previsao_treino.model import META

FORMATO_TIMESTAMP = "%d/%m/%Y %H:%M"
TIPOS = tuple(META.keys())

# Datas oficiais das World Marathon Majors, para o desfecho de maratona.
DATAS_MAJORS = {
    "TOKYO 2019": "2019-03-03",
    "BOSTON 2019": "2019-04-15",
    "LONDON 2019": "2019-04-28",
    "BERLIN 2019": "2019-09-29",
    "CHICAGO 2019": "2019-10-13",
    "NEW YORK 2019": "2019-11-03",
    "TOKYO 2020": "2020-03-01",
    "LONDON 2020": "2020-10-04",
}


def _com_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col(COL_TIMESTAMP)
        .str.strptime(pl.Datetime, format=FORMATO_TIMESTAMP, strict=False)
        .alias("_ts")
    ).drop_nulls("_ts")


def extrair_melhor_esforco(
    df: pl.DataFrame,
    distancia_km: float = DISTANCIA_MEIA_KM,
    tolerancia_km: float = TOLERANCIA_PROVA_KM,
) -> pl.DataFrame:
    """
    Melhor (menor) tempo de cada atleta na distância-alvo.

    Retorna athlete, data_prova, tempo_prova_min, pace_prova_min_km,
    distancia_prova_km e gender.
    """
    minimo = (distancia_km - tolerancia_km) * 1000
    maximo = (distancia_km + tolerancia_km) * 1000

    candidatos = _com_timestamp(df).filter(
        pl.col(COL_DISTANCE).is_between(minimo, maximo)
    )
    if candidatos.height == 0:
        return pl.DataFrame()

    return (
        candidatos.with_columns([
            (pl.col(COL_ELAPSED) / 60).alias(COL_TEMPO_PROVA),
            (pl.col(COL_DISTANCE) / 1000).alias(COL_DISTANCIA_PROVA),
        ])
        # Normaliza o tempo para a distância nominal: 20,6 km e 21,6 km não são
        # comparáveis em tempo bruto, mas são em pace.
        .with_columns(
            (pl.col(COL_TEMPO_PROVA) / pl.col(COL_DISTANCIA_PROVA)).alias(COL_PACE_PROVA)
        )
        .sort([COL_ATHLETE, COL_PACE_PROVA])
        .group_by(COL_ATHLETE, maintain_order=True)
        .first()
        .select([
            COL_ATHLETE,
            COL_GENDER,
            pl.col("_ts").alias(COL_DATA_PROVA),
            COL_TEMPO_PROVA,
            COL_PACE_PROVA,
            COL_DISTANCIA_PROVA,
        ])
    )


def extrair_maratonas_oficiais(
    df: pl.DataFrame,
    col_majors: str = "major",
) -> pl.DataFrame:
    """
    Maratonas verificadas: atleta declara a prova em `major` e tem um esforço
    de ~42 km na data oficial dela. É o desfecho de maior validade disponível.
    """
    if col_majors not in df.columns:
        raise ValueError(
            f"coluna {col_majors!r} ausente; carregue a fonte com manter_extras=True"
        )

    minimo = (DISTANCIA_MARATONA_KM - 2.5) * 1000
    maximo = (DISTANCIA_MARATONA_KM + 2.5) * 1000
    marcado = _com_timestamp(df)

    blocos = []
    for prova, data in DATAS_MAJORS.items():
        bloco = marcado.filter(
            pl.col(col_majors).str.contains(prova, literal=True)
            & (pl.col("_ts").dt.date() == pl.lit(data).str.strptime(pl.Date, "%Y-%m-%d"))
            & pl.col(COL_DISTANCE).is_between(minimo, maximo)
        )
        if bloco.height:
            blocos.append(bloco.with_columns(pl.lit(prova).alias("prova")))

    if not blocos:
        return pl.DataFrame()

    return (
        pl.concat(blocos)
        .with_columns([
            (pl.col(COL_ELAPSED) / 60).alias(COL_TEMPO_PROVA),
            (pl.col(COL_DISTANCE) / 1000).alias(COL_DISTANCIA_PROVA),
        ])
        .with_columns(
            (pl.col(COL_TEMPO_PROVA) / pl.col(COL_DISTANCIA_PROVA)).alias(COL_PACE_PROVA)
        )
        .sort([COL_ATHLETE, COL_PACE_PROVA])
        .group_by(COL_ATHLETE, maintain_order=True)
        .first()
        .select([
            COL_ATHLETE,
            COL_GENDER,
            "prova",
            pl.col("_ts").alias(COL_DATA_PROVA),
            COL_TEMPO_PROVA,
            COL_PACE_PROVA,
            COL_DISTANCIA_PROVA,
        ])
    )


def features_preparo(
    df: pl.DataFrame,
    desfecho: pl.DataFrame,
    semanas: int = SEMANAS_PREPARO,
    min_treinos: int = MIN_TREINOS_PREPARO,
) -> pl.DataFrame:
    """
    Agrega o treino da janela de `semanas` que antecede a prova de cada atleta.

    A prova em si é excluída da janela (senão o desfecho entraria como sua
    própria feature). Esforços iguais ou mais longos que a prova também são
    mantidos: em preparo de 21 km um longão de 21 km é treino legítimo, e
    removê-los penalizaria justamente quem treina mais.
    """
    if desfecho.height == 0:
        return pl.DataFrame()

    janela = pl.duration(days=semanas * 7)

    base = (
        _com_timestamp(df)
        .join(desfecho.select([COL_ATHLETE, COL_DATA_PROVA]), on=COL_ATHLETE, how="inner")
        .filter(
            (pl.col("_ts") < pl.col(COL_DATA_PROVA))
            & (pl.col("_ts") >= pl.col(COL_DATA_PROVA) - janela)
        )
    )

    agregados = (
        base.group_by(COL_ATHLETE)
        .agg([
            pl.len().alias("n_treinos"),
            (pl.col(COL_DISTANCE).sum() / 1000).alias("volume_km"),
            (pl.col(COL_DISTANCE).max() / 1000).alias("longao_max_km"),
            (pl.col(COL_DISTANCE).mean() / 1000).alias("distancia_media_km"),
            pl.col(COL_PACE).median().alias("pace_treino_mediano"),
            pl.col(COL_PACE).std().alias("pace_treino_std"),
            *[
                (pl.col(COL_TIPO_TREINO) == tipo).mean().alias(f"prop_{tipo}")
                for tipo in TIPOS
            ],
        ])
        .filter(pl.col("n_treinos") >= min_treinos)
        .with_columns(
            (pl.col("volume_km") / (semanas * 1.0)).alias("volume_semanal_km"),
            pl.col("pace_treino_std").fill_null(0.0),
        )
    )

    return desfecho.join(agregados, on=COL_ATHLETE, how="inner")


def distancia_a_meta(
    tabela: pl.DataFrame,
    meta: dict[str, float] | None = None,
) -> pl.DataFrame:
    """
    Adiciona `distancia_meta`: distância de variação total (L1/2) entre a
    distribuição de treino do atleta e a meta. 0 = idêntica, 1 = disjunta.
    """
    meta_uso = meta if meta is not None else META
    soma = sum(
        (pl.col(f"prop_{tipo}") - valor).abs() for tipo, valor in meta_uso.items()
    )
    return tabela.with_columns((soma / 2).alias("distancia_meta"))


FEATURES_VOLUME = [
    "volume_semanal_km",
    "n_treinos",
    "longao_max_km",
    "pace_treino_mediano",
]
FEATURES_COMPOSICAO = [f"prop_{tipo}" for tipo in TIPOS]


def montar_tabela_desfecho(
    df: pl.DataFrame,
    distancia_km: float = DISTANCIA_MEIA_KM,
    semanas: int = SEMANAS_PREPARO,
    min_treinos: int = MIN_TREINOS_PREPARO,
) -> pl.DataFrame:
    """Pipeline completo: desfecho + features de preparo + aderência à meta."""
    desfecho = extrair_melhor_esforco(df, distancia_km=distancia_km)
    if desfecho.height == 0:
        return pl.DataFrame()

    tabela = features_preparo(df, desfecho, semanas=semanas, min_treinos=min_treinos)
    if tabela.height == 0:
        return pl.DataFrame()

    return distancia_a_meta(tabela)


def resumo_desfecho(tabela: pl.DataFrame) -> dict[str, float]:
    """Estatísticas descritivas para verificar plausibilidade do desfecho."""
    tempos = tabela[COL_TEMPO_PROVA].to_numpy()
    return {
        "atletas": tabela.height,
        "tempo_mediano_min": float(np.median(tempos)),
        "tempo_p10_min": float(np.percentile(tempos, 10)),
        "tempo_p90_min": float(np.percentile(tempos, 90)),
        "pace_mediano_prova": float(tabela[COL_PACE_PROVA].median()),
        "volume_semanal_mediano_km": float(tabela["volume_semanal_km"].median()),
        "distancia_meta_mediana": float(tabela["distancia_meta"].median()),
    }
