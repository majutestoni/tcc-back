from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import polars as pl

from gemini.schemas import RecomendacaoTreino

RECOMENDACOES_CSV = "output/recomendacoes.csv"
FORMATO_TIMESTAMP = "%d/%m/%Y %H:%M"

COLUNAS = [
    "athlete_id",
    "timestamp",
    "origem",
    "tipo",
    "descricao",
    "distancia_km",
    "pace_alvo_min_km",
    "tempo_estimado_min",
    "justificativa",
    "alertas",
]

_SCHEMA = {
    "athlete_id": pl.Int64,
    "timestamp": pl.Utf8,
    "origem": pl.Utf8,
    "tipo": pl.Utf8,
    "descricao": pl.Utf8,
    "distancia_km": pl.Float64,
    "pace_alvo_min_km": pl.Float64,
    "tempo_estimado_min": pl.Float64,
    "justificativa": pl.Utf8,
    "alertas": pl.Utf8,
}


def _campos_recomendacao(recomendacao: RecomendacaoTreino | Mapping) -> dict:
    """Normaliza uma recomendação (Gemini `RecomendacaoTreino` ou dict de regras)."""
    if isinstance(recomendacao, RecomendacaoTreino):
        dados = recomendacao.model_dump()
    else:
        dados = dict(recomendacao)

    alertas = dados.get("alertas") or []
    if isinstance(alertas, str):
        alertas = [alertas]

    return {
        "tipo": dados["tipo"],
        "descricao": dados.get("descricao", ""),
        "distancia_km": float(dados["distancia_km"]),
        "pace_alvo_min_km": float(dados["pace_alvo_min_km"]),
        "tempo_estimado_min": float(dados["tempo_estimado_min"]),
        "justificativa": dados.get("justificativa") or "",
        "alertas": list(alertas),
    }


def _linha_recomendacao(
    athlete_id: int,
    recomendacao: RecomendacaoTreino | Mapping,
    origem: str,
    quando: str,
) -> dict:
    campos = _campos_recomendacao(recomendacao)
    return {
        "athlete_id": int(athlete_id),
        "timestamp": quando,
        "origem": origem,
        "tipo": campos["tipo"],
        "descricao": campos["descricao"],
        "distancia_km": campos["distancia_km"],
        "pace_alvo_min_km": campos["pace_alvo_min_km"],
        "tempo_estimado_min": campos["tempo_estimado_min"],
        "justificativa": campos["justificativa"],
        "alertas": json.dumps(campos["alertas"], ensure_ascii=False),
    }


def carregar_historico(caminho: str = RECOMENDACOES_CSV) -> pl.DataFrame:
    """Lê o histórico de recomendações; retorna tabela vazia se ainda não existir."""
    if not Path(caminho).exists():
        return pl.DataFrame(schema=_SCHEMA)
    return pl.read_csv(caminho, schema_overrides=_SCHEMA)


def salvar_recomendacao(
    athlete_id: int,
    recomendacao: RecomendacaoTreino | Mapping,
    origem: str = "gemini",
    quando: str | None = None,
    caminho: str = RECOMENDACOES_CSV,
) -> str:
    """Anexa uma recomendação ao histórico (`output/recomendacoes.csv`).

    Aceita tanto o `RecomendacaoTreino` do Gemini (com justificativa/alertas)
    quanto o dict da recomendação por regras (justificativa/alertas vazios).
    """
    if quando is None:
        quando = datetime.now().strftime(FORMATO_TIMESTAMP)

    nova = pl.DataFrame(
        [_linha_recomendacao(athlete_id, recomendacao, origem, quando)],
        schema=_SCHEMA,
    )
    historico = pl.concat([carregar_historico(caminho), nova], how="vertical")

    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    historico.write_csv(caminho)
    return caminho


def alertas_para_lista(valor: str | None) -> list[str]:
    """Converte a coluna `alertas` (JSON) de volta para uma lista de strings."""
    if not valor:
        return []
    try:
        dados = json.loads(valor)
    except json.JSONDecodeError:
        return [valor]
    return dados if isinstance(dados, list) else [str(dados)]
