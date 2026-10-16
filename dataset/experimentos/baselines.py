"""
Recomendadores baseline (ADR-017, item 2; ADR-024).

A regra de produto é só o cluster (`déficit vs meta do cluster`). Os baselines
triviais servem de piso de comparação no protocolo de imitação.

- `aleatorio` — piso absoluto (25% por tipo).
- `sempre_leve` — constante; se a regra não superar isto, ela é degenerada.
- `majoritario_global` / `majoritario_atleta` — ignoram contexto fino.
- `distribuicao_atleta` — amostra da distribuição histórica do atleta.
- `repetir_ultimo` — autocorrelação temporal.
- `cluster` — regra de produto (meta do perfil KMeans).
"""

from __future__ import annotations

import numpy as np

from dataset.constants import SEED
from dataset.experimentos.protocolo_recomendacao import (
    TIPOS,
    ContextoAtleta,
    Recomendador,
)
from dataset.models.previsao_treino.model import (
    PESOS_DEFICIT,
    calcular_deficit,
    recomendar_proximo_treino,
)
from dataset.models.previsao_treino.recomendacao_cluster import (
    META_CLUSTER_PADRAO,
    meta_para_perfil,
)

NOME_REGRA_META_CLUSTER = "déficit vs meta do cluster"


def recomendador_aleatorio(seed: int = SEED) -> Recomendador:
    """Uniforme sobre os 4 tipos. Piso de comparação."""
    rng = np.random.default_rng(seed)

    def recomendar(_: ContextoAtleta) -> str:
        return str(rng.choice(TIPOS))

    return recomendar


def recomendador_constante(tipo: str) -> Recomendador:
    """Sempre o mesmo tipo. `tipo='leve'` é o baseline degenerado relevante."""
    if tipo not in TIPOS:
        raise ValueError(f"tipo deve ser um de {TIPOS}, recebido {tipo!r}")

    def recomendar(_: ContextoAtleta) -> str:
        return tipo

    return recomendar


def recomendador_majoritario_global(tipo_majoritario: str) -> Recomendador:
    """A classe mais frequente do conjunto de treino, igual para todo atleta."""
    return recomendador_constante(tipo_majoritario)


def recomendador_majoritario_atleta() -> Recomendador:
    """A classe mais frequente do histórico do próprio atleta."""

    def recomendar(contexto: ContextoAtleta) -> str:
        if contexto.tipos.size == 0:
            return "leve"
        tipos, contagens = np.unique(contexto.tipos, return_counts=True)
        return str(tipos[int(np.argmax(contagens))])

    return recomendar


def recomendador_distribuicao_atleta(seed: int = SEED) -> Recomendador:
    """Amostra da distribuição histórica do atleta na janela temporal."""
    rng = np.random.default_rng(seed)

    def recomendar(contexto: ContextoAtleta) -> str:
        distribuicao = contexto.distribuicao()
        pesos = np.array([distribuicao[t] for t in TIPOS], dtype=float)
        if pesos.sum() <= 0:
            return str(rng.choice(TIPOS))
        return str(rng.choice(TIPOS, p=pesos / pesos.sum()))

    return recomendar


def recomendador_repetir_ultimo() -> Recomendador:
    """Repete o tipo da corrida anterior. Explora autocorrelação."""

    def recomendar(contexto: ContextoAtleta) -> str:
        return contexto.ultimo_tipo or "leve"

    return recomendar


def recomendador_deficit(
    meta: dict[str, float] | None = None,
    pesos: dict[str, float] | None = None,
    usar_antifadiga: bool = True,
) -> Recomendador:
    """
    Regra por déficit vs meta. Sem `meta`, usa `META_CLUSTER_PADRAO`
    (volume_moderado). Usado pela ablação e pelo recomendador de cluster.
    """
    meta_uso = meta if meta is not None else META_CLUSTER_PADRAO
    pesos_uso = pesos if pesos is not None else PESOS_DEFICIT

    def recomendar(contexto: ContextoAtleta) -> str:
        deficits = calcular_deficit(meta_uso, contexto.distribuicao())
        ultimos = [contexto.ultimo_tipo] if contexto.ultimo_tipo else []
        return recomendar_proximo_treino(
            deficits=deficits,
            ultimos_treinos=ultimos,
            pace_medio_atleta=contexto.pace_medio,
            pesos=pesos_uso,
            usar_antifadiga=usar_antifadiga,
        )["tipo"]

    return recomendar


def recomendador_cluster(
    perfil_por_atleta: dict[int, str],
) -> Recomendador:
    """Regra de produto: meta do perfil_cluster (ADR-024)."""

    def recomendar(contexto: ContextoAtleta) -> str:
        chave = perfil_por_atleta.get(contexto.athlete_id)
        return recomendador_deficit(meta=meta_para_perfil(chave))(contexto)

    return recomendar


def baselines_padrao(tipo_majoritario: str = "leve") -> dict[str, Recomendador]:
    """Baselines triviais (sem a regra de produto)."""
    return {
        "aleatorio": recomendador_aleatorio(),
        "sempre_leve": recomendador_constante("leve"),
        f"majoritario_global ({tipo_majoritario})": recomendador_majoritario_global(
            tipo_majoritario
        ),
        "majoritario_atleta": recomendador_majoritario_atleta(),
        "distribuicao_atleta": recomendador_distribuicao_atleta(),
        "repetir_ultimo": recomendador_repetir_ultimo(),
    }


def baselines_com_cluster(
    perfil_por_atleta: dict[int, str],
    tipo_majoritario: str = "leve",
) -> dict[str, Recomendador]:
    """Baselines triviais + regra de produto (meta do cluster)."""
    recs = baselines_padrao(tipo_majoritario=tipo_majoritario)
    recs[NOME_REGRA_META_CLUSTER] = recomendador_cluster(perfil_por_atleta)
    return recs
