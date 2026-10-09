"""Contrato comum das fontes de dados (ADR-018)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from dataset.constants import (
    COL_ATHLETE,
    COL_DISTANCE,
    COL_ELAPSED,
    COL_ELEVATION,
    COL_GENDER,
    COL_HR,
    COL_TIMESTAMP,
)

# Schema canônico esperado pelo pipeline de ADR-003.
COLUNAS_CANONICAS = [
    COL_ATHLETE,
    COL_GENDER,
    COL_TIMESTAMP,
    COL_DISTANCE,
    COL_ELAPSED,
    COL_ELEVATION,
    COL_HR,
]

FORMATO_TIMESTAMP = "%d/%m/%Y %H:%M"


@dataclass(frozen=True)
class FonteDados:
    """
    Dados de uma fonte já no schema canônico, com metadados sobre o que ela
    oferece. `tem_fc` e `tem_elevacao` dizem aos experimentos quais features e
    quais modelos são aplicáveis àquela fonte.
    """

    nome: str
    df: pl.DataFrame
    tem_fc: bool
    tem_elevacao: bool

    def __post_init__(self) -> None:
        faltando = [c for c in COLUNAS_CANONICAS if c not in self.df.columns]
        if faltando:
            raise ValueError(
                f"fonte {self.nome!r} não está no schema canônico; faltam: {faltando}"
            )

    def resumo(self) -> dict[str, object]:
        return {
            "fonte": self.nome,
            "corridas": self.df.height,
            "atletas": self.df[COL_ATHLETE].n_unique(),
            "tem_fc": self.tem_fc,
            "tem_elevacao": self.tem_elevacao,
        }
