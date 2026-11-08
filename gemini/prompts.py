from __future__ import annotations

from gemini.schemas import FeaturesAtleta

SYSTEM_INSTRUCTION = (
    "Você é um treinador de corrida experiente. Recomenda o PRÓXIMO treino de um "
    "corredor amador com base em métricas objetivas do histórico dele. "
    "Priorize equilibrar a distribuição de tipos de treino em relação à meta, "
    "respeite o pace médio real do atleta e evite sobrecarga: não repita treinos "
    "intensos (intervalado ou longao) em sequência imediata. "
    "Os tipos válidos são: leve, longao, ritmo, intervalado. "
    "Baseie a distância e o pace alvo nos números reais do atleta, não em valores "
    "genéricos. Responda sempre em português, de forma objetiva e prática."
)


def construir_prompt(features: FeaturesAtleta) -> str:
    return (
        "Recomende o próximo treino para o atleta a seguir. "
        "Use os déficits em relação à meta, o pace médio da janela recente e os "
        "últimos treinos para decidir tipo, distância e pace alvo.\n\n"
        f"Dados do atleta (JSON):\n{features.model_dump_json(indent=2)}"
    )
