"""
Protocolo experimental do projeto (ADR-017).

Corrige a avaliação dos modelos: particionamento agrupado por atleta,
intervalos de confiança por bootstrap de atletas e testes pareados com
correção para múltiplas comparações.

Imports lazy (ADR-012) — este pacote arrasta sklearn e scipy.
"""

from __future__ import annotations

__all__ = [
    "EspecModelo",
    "ResultadoAvaliacao",
    "avaliar",
    "avaliar_protocolo_antigo",
    "avaliar_todos",
    "comparar_por_categoria",
    "comparar_todos",
    "correcao_holm",
    "folds_por_atleta",
    "ic_bootstrap",
    "split_temporal",
    "split_temporal_por_atleta",
    "tabela_comparacoes",
    "tabela_comparativa",
    "wilcoxon_pareado",
]

_MAPA = {
    "EspecModelo": "avaliador",
    "ResultadoAvaliacao": "avaliador",
    "avaliar": "avaliador",
    "avaliar_protocolo_antigo": "avaliador",
    "avaliar_todos": "comparacao",
    "comparar_por_categoria": "comparacao",
    "tabela_comparacoes": "comparacao",
    "tabela_comparativa": "comparacao",
    "comparar_todos": "estatistica",
    "correcao_holm": "estatistica",
    "ic_bootstrap": "estatistica",
    "wilcoxon_pareado": "estatistica",
    "folds_por_atleta": "particionamento",
    "split_temporal": "particionamento",
    "split_temporal_por_atleta": "particionamento",
}


def __getattr__(nome: str):
    if nome not in _MAPA:
        raise AttributeError(f"module {__name__!r} has no attribute {nome!r}")

    from importlib import import_module

    modulo = import_module(f"{__name__}.{_MAPA[nome]}")
    return getattr(modulo, nome)


def __dir__() -> list[str]:
    return sorted(__all__)
