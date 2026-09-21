from __future__ import annotations

import polars as pl

from gemini.client import GeminiClient
from gemini.config import GeminiConfig
from gemini.features import montar_features_atleta
from gemini.prompts import SYSTEM_INSTRUCTION, construir_prompt
from gemini.schemas import RecomendacaoTreino


def recomendar_treino_gemini(
    df: pl.DataFrame,
    athlete_id: int,
    config: GeminiConfig | None = None,
) -> RecomendacaoTreino:
    """Recomenda o próximo treino de um atleta específico usando a API do Gemini.

    O DataFrame deve estar preprocessado (ver dataset.preprocessamento.preprocessar).
    """
    features = montar_features_atleta(df, athlete_id)
    cliente = GeminiClient(config)
    prompt = construir_prompt(features)
    return cliente.gerar_estruturado(
        contents=prompt,
        schema=RecomendacaoTreino,
        system_instruction=SYSTEM_INSTRUCTION,
    )


def _main() -> None:
    import sys

    from dataset.preprocessamento import preprocessar

    if len(sys.argv) < 2:
        print("Uso: python -m gemini.recomendacao <athlete_id>")
        raise SystemExit(1)

    athlete_id = int(sys.argv[1])
    df = preprocessar(pl.read_csv("dataset/raw-data-kaggle.csv", separator=";"))
    recomendacao = recomendar_treino_gemini(df, athlete_id)

    print(f"\nRecomendação para o atleta {athlete_id} (via Gemini):")
    print(f"  Tipo: {recomendacao.tipo}")
    print(f"  {recomendacao.descricao}")
    print(f"  Distância: {recomendacao.distancia_km} km")
    print(f"  Pace alvo: {recomendacao.pace_alvo_min_km} min/km")
    print(f"  Tempo estimado: {recomendacao.tempo_estimado_min} min")
    print(f"  Justificativa: {recomendacao.justificativa}")
    if recomendacao.alertas:
        print("  Alertas:")
        for alerta in recomendacao.alertas:
            print(f"    - {alerta}")


if __name__ == "__main__":
    _main()
