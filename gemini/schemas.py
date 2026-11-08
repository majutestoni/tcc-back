from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TipoTreino = Literal["leve", "longao", "ritmo", "intervalado"]


class FeaturesAtleta(BaseModel):
    """Resumo determinístico do atleta enviado ao Gemini."""

    athlete_id: int
    genero: str | None = None
    perfil_atleta: str | None = None
    n_corridas: int
    frequencia_semana: float
    pace_mediano_min_km: float
    distancia_mediana_km: float
    fc_mediana: float | None = None
    pace_medio_janela_min_km: float | None = None
    distribuicao_atual: dict[str, float]
    meta_distribuicao: dict[str, float]
    deficits: dict[str, float]
    ultimos_treinos: list[str]
    janela_semanas: int


class RecomendacaoTreino(BaseModel):
    """Recomendação estruturada retornada pelo Gemini."""

    tipo: TipoTreino = Field(description="Tipo do próximo treino recomendado")
    descricao: str = Field(description="Descrição curta do treino")
    distancia_km: float = Field(description="Distância alvo em km")
    pace_alvo_min_km: float = Field(description="Pace alvo em min/km")
    tempo_estimado_min: float = Field(description="Tempo estimado em minutos")
    justificativa: str = Field(
        description="Explicação, em português, do porquê deste treino"
    )
    alertas: list[str] = Field(
        default_factory=list, description="Avisos ou cuidados para o atleta"
    )
