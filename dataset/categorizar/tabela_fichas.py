import polars as pl

from dataset.constants import (
    COL_ATHLETE,
    COL_CLUSTER_CORREDOR,
    COL_DISTANCE,
    COL_GENDER,
    COL_N_CORRIDAS,
    COL_PACE,
)
from dataset.preprocessamento import formatar_delta_pace, formatar_pace


def tabela_fichas(fichas: pl.DataFrame) -> pl.DataFrame:
    ordenado = fichas.sort("categoria", COL_PACE)
    return pl.DataFrame(
        {
            "id": ordenado[COL_ATHLETE],
            "nome": ordenado["nome_ficticio"],
            "genero": ordenado[COL_GENDER],
            "categoria": ordenado["categoria"],
            "corridas": ordenado[COL_N_CORRIDAS],
            "cluster": ordenado[COL_CLUSTER_CORREDOR],
            "dist_km": (ordenado[COL_DISTANCE] / 1000).round(2),
            "pace": [formatar_pace(v) for v in ordenado[COL_PACE].to_list()],
            "melhor_categoria": [
                formatar_pace(v) for v in ordenado["pace_melhor_categoria"].to_list()
            ],
            "pior_categoria": [
                formatar_pace(v) for v in ordenado["pace_pior_categoria"].to_list()
            ],
            "delta_melhor": [
                formatar_delta_pace(v) for v in ordenado["dist_pace_melhor"].to_list()
            ],
            "delta_pior": [
                formatar_delta_pace(v) for v in ordenado["dist_pace_pior"].to_list()
            ],
            "rank": ordenado["ranking_na_categoria"],
            "n_categoria": ordenado["atletas_na_categoria"],
        }
    )
