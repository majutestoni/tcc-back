"""
Adaptadores de fonte de dados (ADR-018).

Cada fonte entrega um DataFrame no schema canônico de `dataset/constants.py`,
para que o pipeline de ADR-003 e os modelos rodem sem saber de onde vieram os
dados. Imports lazy seguindo ADR-012.
"""

from __future__ import annotations

__all__ = [
    "FonteDados",
    "carregar_strava_kaggle",
    "carregar_long_distance",
    "caminho_long_distance",
    "COLUNAS_CANONICAS",
]

_MAPA = {
    "FonteDados": "base",
    "COLUNAS_CANONICAS": "base",
    "carregar_strava_kaggle": "strava_kaggle",
    "carregar_long_distance": "long_distance",
    "caminho_long_distance": "long_distance",
}


def __getattr__(nome: str):
    if nome not in _MAPA:
        raise AttributeError(f"module {__name__!r} has no attribute {nome!r}")

    from importlib import import_module

    modulo = import_module(f"{__name__}.{_MAPA[nome]}")
    return getattr(modulo, nome)


def __dir__() -> list[str]:
    return sorted(__all__)
