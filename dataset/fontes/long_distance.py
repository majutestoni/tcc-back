"""
Fonte externa: Kaggle mexwell/long-distance-running-dataset (ADR-018).

10,7 M registros de 36.412 atletas em 2019-2020. Usada como conjunto de
validação externa porque, ao contrário da fonte primária, permite extrair um
**desfecho de prova** (ver `dataset/experimentos/desfecho.py`).

Diferenças de schema tratadas aqui:
- `distance` em km  → `distance (m)`
- `duration` em min → `elapsed time (s)`
- `datetime` ISO    → `timestamp` no formato `%d/%m/%Y %H:%M`
- não há FC nem elevação → colunas criadas nulas (o pipeline detecta, ADR-003)
- painel denso: dias sem corrida vêm com `distance = 0` e são descartados

Os arquivos não são versionados (2,3 GB). Baixe com `bash scripts/baixar_long_distance.sh`
ou aponte `DIRETORIO_DADOS_EXTERNOS` para onde eles já estão.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from dataset.constants import (
    COL_ATHLETE,
    COL_DISTANCE,
    COL_ELAPSED,
    COL_ELEVATION,
    COL_FAIXA_IDADE,
    COL_GENDER,
    COL_HR,
    COL_MAJORS,
    COL_PAIS,
    COL_TIMESTAMP,
)
from dataset.fontes.base import FonteDados

ANOS = (2019, 2020)
FREQUENCIAS = ("d", "w", "m", "q")

DIRETORIO_PADRAO = Path.home() / ".cache" / "tcc-back-2" / "dados"

# Colunas cruas da fonte, preservadas para a extração de desfecho.
COLUNAS_EXTRA = [COL_FAIXA_IDADE, COL_PAIS, COL_MAJORS]


def diretorio_dados() -> Path:
    """Diretório dos CSVs externos; sobrescrevível por `DIRETORIO_DADOS_EXTERNOS`."""
    bruto = os.getenv("DIRETORIO_DADOS_EXTERNOS")
    return Path(bruto).expanduser() if bruto else DIRETORIO_PADRAO


def caminho_long_distance(ano: int = 2019, frequencia: str = "d") -> Path:
    """Caminho do CSV `run_ww_<ano>_<frequencia>.csv`."""
    if ano not in ANOS:
        raise ValueError(f"ano deve ser um de {ANOS}, recebido {ano}")
    if frequencia not in FREQUENCIAS:
        raise ValueError(f"frequencia deve ser uma de {FREQUENCIAS}, recebida {frequencia!r}")

    caminho = diretorio_dados() / f"run_ww_{ano}_{frequencia}.csv"
    if not caminho.exists():
        raise FileNotFoundError(
            f"arquivo da fonte externa não encontrado em {caminho}.\n"
            "Rode `bash scripts/baixar_long_distance.sh` ou defina DIRETORIO_DADOS_EXTERNOS."
        )
    return caminho


def scan_long_distance(
    ano: int = 2019,
    frequencia: str = "d",
    manter_extras: bool = True,
) -> pl.LazyFrame:
    """
    LazyFrame no schema canônico, sem materializar o arquivo (até 956 MB).

    Descarta os dias de descanso (`distance = 0`), que são ~65% das linhas do
    painel diário e não representam corridas.
    """
    colunas = [
        pl.col("athlete").alias(COL_ATHLETE),
        pl.col("gender").alias(COL_GENDER),
        pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False)
        .dt.strftime("%d/%m/%Y %H:%M")
        .alias(COL_TIMESTAMP),
        (pl.col("distance") * 1000).alias(COL_DISTANCE),
        (pl.col("duration") * 60).alias(COL_ELAPSED),
        pl.lit(None, dtype=pl.Float64).alias(COL_ELEVATION),
        pl.lit(None, dtype=pl.Float64).alias(COL_HR),
    ]
    if manter_extras:
        colunas += [pl.col(c) for c in COLUNAS_EXTRA]

    return (
        pl.scan_csv(caminho_long_distance(ano, frequencia))
        .filter(pl.col("distance") > 0)
        .select(colunas)
    )


def carregar_long_distance(
    ano: int = 2019,
    frequencia: str = "d",
    max_atletas: int | None = None,
    manter_extras: bool = True,
) -> FonteDados:
    """
    Materializa a fonte externa no schema canônico.

    `max_atletas` limita por ID (os IDs são inteiros densos de 0 a 36.411), o
    que dá uma amostra reprodutível e barata para desenvolvimento.
    """
    lz = scan_long_distance(ano, frequencia, manter_extras=manter_extras)
    if max_atletas is not None:
        lz = lz.filter(pl.col(COL_ATHLETE) < max_atletas)

    return FonteDados(
        nome=f"long-distance-{ano}-{frequencia}",
        df=lz.collect(engine="streaming"),
        tem_fc=False,
        tem_elevacao=False,
    )
