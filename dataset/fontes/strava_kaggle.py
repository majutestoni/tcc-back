"""
Fonte primária: Kaggle olegoaer/running-races-strava (ADR-001, seção 1 do ADR).

Já está no schema canônico — o adaptador existe para uniformizar a interface
com as demais fontes, não para converter nada.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dataset.fontes.base import FonteDados

CAMINHO_PADRAO = Path("dataset/raw-data-kaggle.csv")
SEPARADOR = ";"


def carregar_strava_kaggle(caminho: Path | str = CAMINHO_PADRAO) -> FonteDados:
    """Carrega o CSV primário do projeto (separador `;`)."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"dataset primário não encontrado em {caminho}")

    return FonteDados(
        nome="strava-kaggle",
        df=pl.read_csv(caminho, separator=SEPARADOR),
        tem_fc=True,
        tem_elevacao=True,
    )
