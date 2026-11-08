"""
Particionamento de dados para avaliação (ADR-017).

O protocolo antigo usava `train_test_split` aleatório **por corrida**, o que
coloca o mesmo atleta em treino e teste. Como as features derivadas são
relativas ao atleta (pace relativo, FC relativa) e as corridas de um mesmo
atleta são fortemente correlacionadas, a métrica resultante responde
"quão bem o modelo interpola dentro de atletas que já viu", não
"quão bem generaliza para um atleta novo" — que é o caso de uso real
(SPEC 6.6: um atleta novo conecta o Strava e pede análise).

Aqui a unidade de particionamento é o **atleta**.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from dataset.constants import COL_ATHLETE, COL_TIMESTAMP, N_FOLDS, SEED

FORMATO_TIMESTAMP = "%d/%m/%Y %H:%M"


def grupos_por_atleta(df: pl.DataFrame) -> np.ndarray:
    """Vetor de grupos para CV agrupada: o ID do atleta de cada corrida."""
    return df[COL_ATHLETE].to_numpy()


def folds_por_atleta(
    df: pl.DataFrame,
    n_folds: int = N_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Folds de `GroupKFold` em que nenhum atleta aparece em treino e teste.

    `GroupKFold` é determinístico (não aceita `random_state`), então a
    reprodutibilidade de ADR (seção 4) é preservada sem semente.
    """
    grupos = grupos_por_atleta(df)
    n_atletas = len(np.unique(grupos))
    if n_atletas < n_folds:
        raise ValueError(
            f"{n_atletas} atletas é menos que {n_folds} folds; reduza n_folds"
        )

    indices = np.arange(df.height)
    return list(GroupKFold(n_splits=n_folds).split(indices, groups=grupos))


def _timestamps(df: pl.DataFrame) -> pl.Series:
    return (
        df[COL_TIMESTAMP]
        .str.strptime(pl.Datetime, format=FORMATO_TIMESTAMP, strict=False)
    )


def split_temporal(
    df: pl.DataFrame,
    fracao_teste: float = 0.2,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Corta o dataset numa data única: passado treina, futuro testa.

    Usado quando a pergunta é de previsão no tempo. Não controla atleta — um
    atleta pode aparecer nos dois lados, o que é aceitável aqui porque o
    impedimento é temporal, não de identidade.
    """
    if not 0.0 < fracao_teste < 1.0:
        raise ValueError(f"fracao_teste deve estar em (0, 1), recebido {fracao_teste}")

    ts = _timestamps(df)
    corte = ts.quantile(1.0 - fracao_teste)
    marcado = df.with_columns(ts.alias("_ts"))

    treino = marcado.filter(pl.col("_ts") < corte).drop("_ts")
    teste = marcado.filter(pl.col("_ts") >= corte).drop("_ts")
    return treino, teste


def split_temporal_por_atleta(
    df: pl.DataFrame,
    n_teste: int = 1,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Separa as últimas `n_teste` corridas de **cada** atleta para teste.

    É o protocolo de recomendação (item 2): dado o histórico até t, prever o
    treino em t+1. Mantém todo atleta presente nos dois lados por construção,
    o que é o desejado — a tarefa é sobre o próximo treino de um atleta
    conhecido, não sobre um atleta novo.
    """
    if n_teste < 1:
        raise ValueError(f"n_teste deve ser >= 1, recebido {n_teste}")

    marcado = (
        df.with_columns(_timestamps(df).alias("_ts"))
        .sort([COL_ATHLETE, "_ts"])
        .with_columns(
            (pl.len().over(COL_ATHLETE) - pl.col("_ts").cum_count().over(COL_ATHLETE))
            .alias("_desde_fim")
        )
    )

    treino = marcado.filter(pl.col("_desde_fim") >= n_teste).drop("_ts", "_desde_fim")
    teste = marcado.filter(pl.col("_desde_fim") < n_teste).drop("_ts", "_desde_fim")
    return treino, teste
