"""
Eficácia da recomendação via desfecho de prova (ADR-026).

A origem rolante mede *imitação* do próximo treino no histórico — inadequada
quando o dataset não é um plano padronizado. Aqui a pergunta é outra: atletas
cuja distribuição de preparo fica **mais perto da meta do cluster** têm
**melhor tempo** de meia maratona?
"""

from __future__ import annotations

import numpy as np
import polars as pl

from dataset.constants import COL_TEMPO_PROVA
from dataset.experimentos.desfecho import TIPOS, distancia_a_meta
from dataset.models.previsao_treino.recomendacao_cluster import (
    META_CLUSTER_PADRAO,
    META_POR_PERFIL,
)


def resumo_aderencia_desfecho(
    tabela: pl.DataFrame,
    meta: dict[str, float] | None = None,
    nome_meta: str = "meta_cluster",
) -> dict:
    """
    Correlação (Pearson) entre distância à meta e tempo de prova + tempos
    medianos nos tercis de aderência (menor distância = mais aderente).
    """
    if tabela.height == 0:
        return {"nome_meta": nome_meta, "n": 0}

    meta_uso = meta if meta is not None else META_CLUSTER_PADRAO
    com = distancia_a_meta(tabela, meta=meta_uso)
    d = com["distancia_meta"].to_numpy()
    t = com[COL_TEMPO_PROVA].to_numpy()

    if len(d) < 10 or np.std(d) < 1e-9 or np.std(t) < 1e-9:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(d, t)[0, 1])

    # Tercis de distância à meta: baixo = mais aderente
    q33, q66 = np.quantile(d, [1 / 3, 2 / 3])
    def _med_tempo(mascara: np.ndarray) -> float | None:
        if mascara.sum() == 0:
            return None
        return float(np.median(t[mascara]))

    tempo_alta = _med_tempo(d <= q33)
    tempo_media = _med_tempo((d > q33) & (d <= q66))
    tempo_baixa = _med_tempo(d > q66)

    return {
        "nome_meta": nome_meta,
        "meta": {k: round(v, 4) for k, v in meta_uso.items()},
        "n": int(com.height),
        "corr_distancia_tempo": round(corr, 4) if corr == corr else None,
        "interpretacao": (
            "mais perto da meta ↔ prova mais rápida"
            if corr == corr and corr > 0.05
            else (
                "relação fraca ou nula"
                if corr == corr
                else "correlação indisponível"
            )
        ),
        "tempo_mediano_alta_aderencia_min": (
            round(tempo_alta, 1) if tempo_alta is not None else None
        ),
        "tempo_mediano_media_aderencia_min": (
            round(tempo_media, 1) if tempo_media is not None else None
        ),
        "tempo_mediano_baixa_aderencia_min": (
            round(tempo_baixa, 1) if tempo_baixa is not None else None
        ),
        "distancia_meta_mediana": round(float(np.median(d)), 4),
    }


def comparar_metas_no_desfecho(tabela: pl.DataFrame) -> pl.DataFrame:
    """Compara aderência×tempo para cada meta de perfil_cluster + uniforme."""
    if tabela.height == 0:
        return pl.DataFrame()

    metas = {
        **{f"cluster · {k}": v for k, v in META_POR_PERFIL.items()},
        "uniforme 25%": dict.fromkeys(TIPOS, 0.25),
    }
    linhas = [
        resumo_aderencia_desfecho(tabela, meta=m, nome_meta=nome)
        for nome, m in metas.items()
    ]
    # Remove o dict aninhado para serializar no Streamlit
    for lin in linhas:
        lin.pop("meta", None)
    return pl.DataFrame(linhas).sort(
        "corr_distancia_tempo", descending=True, nulls_last=True
    )


def tabela_metas_vs_regiao(tabela: pl.DataFrame) -> pl.DataFrame:
    """Cada componente das metas de cluster dentro do p5–p95 observado?"""
    if tabela.height == 0:
        return pl.DataFrame()

    linhas = []
    for tipo in TIPOS:
        p5 = float(tabela[f"prop_{tipo}"].quantile(0.05))
        p95 = float(tabela[f"prop_{tipo}"].quantile(0.95))
        med = float(tabela[f"prop_{tipo}"].median())
        for chave, meta in META_POR_PERFIL.items():
            val = meta[tipo]
            linhas.append(
                {
                    "perfil": chave,
                    "tipo": tipo,
                    "meta": round(val, 4),
                    "observado_mediana": round(med, 4),
                    "observado_p5": round(p5, 4),
                    "observado_p95": round(p95, 4),
                    "dentro_da_regiao": bool(p5 <= val <= p95),
                }
            )
    return pl.DataFrame(linhas)
