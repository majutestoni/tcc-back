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

from dataset.constants import COL_ATHLETE, COL_TEMPO_PROVA
from dataset.experimentos.desfecho import TIPOS, distancia_a_meta
from dataset.models.previsao_treino.recomendacao_cluster import (
    META_CLUSTER_PADRAO,
    META_POR_PERFIL,
)

ROTULOS_TERCIL = ("alta", "media", "baixa")


def _com_tercis(
    tabela: pl.DataFrame,
    meta: dict[str, float],
) -> tuple[pl.DataFrame, float, float]:
    """Tabela com distancia_meta + coluna tercil (alta/media/baixa aderência)."""
    com = distancia_a_meta(tabela, meta=meta)
    d = com["distancia_meta"].to_numpy()
    q33, q66 = np.quantile(d, [1 / 3, 2 / 3])
    com = com.with_columns(
        pl.when(pl.col("distancia_meta") <= q33)
        .then(pl.lit("alta"))
        .when(pl.col("distancia_meta") <= q66)
        .then(pl.lit("media"))
        .otherwise(pl.lit("baixa"))
        .alias("tercil_aderencia")
    )
    return com, float(q33), float(q66)


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
    com, q33, q66 = _com_tercis(tabela, meta_uso)
    d = com["distancia_meta"].to_numpy()
    t = com[COL_TEMPO_PROVA].to_numpy()

    if len(d) < 10 or np.std(d) < 1e-9 or np.std(t) < 1e-9:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(d, t)[0, 1])

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
        "q33_distancia_meta": round(q33, 4),
        "q66_distancia_meta": round(q66, 4),
    }


def tempos_por_tercil(
    tabela: pl.DataFrame,
    meta: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Uma linha por tercil: n atletas + tempo mediano (para o gráfico)."""
    if tabela.height == 0:
        return pl.DataFrame()

    meta_uso = meta if meta is not None else META_CLUSTER_PADRAO
    com, _, _ = _com_tercis(tabela, meta_uso)
    linhas = []
    for rotulo in ROTULOS_TERCIL:
        sub = com.filter(pl.col("tercil_aderencia") == rotulo)
        if sub.height == 0:
            continue
        linhas.append(
            {
                "tercil": rotulo,
                "rotulo": {
                    "alta": "Alta aderência",
                    "media": "Média aderência",
                    "baixa": "Baixa aderência",
                }[rotulo],
                "n": sub.height,
                "tempo_mediano_min": round(float(sub[COL_TEMPO_PROVA].median()), 1),
                "distancia_meta_mediana": round(
                    float(sub["distancia_meta"].median()), 4
                ),
            }
        )
    return pl.DataFrame(linhas)


def exemplos_atletas_aderencia(
    tabela: pl.DataFrame,
    meta: dict[str, float] | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Um atleta representativo por tercil (mais perto da mediana do tercil).

    Serve para a UI contar a história com casos concretos.
    """
    if tabela.height == 0:
        return pl.DataFrame()

    meta_uso = meta if meta is not None else META_CLUSTER_PADRAO
    com, _, _ = _com_tercis(tabela, meta_uso)
    rng = np.random.default_rng(seed)
    escolhidos: list[dict] = []

    for rotulo in ROTULOS_TERCIL:
        sub = com.filter(pl.col("tercil_aderencia") == rotulo)
        if sub.height == 0:
            continue
        med_d = float(sub["distancia_meta"].median())
        med_t = float(sub[COL_TEMPO_PROVA].median())
        # Score: perto da mediana de distância e de tempo do tercil
        score = (
            (pl.col("distancia_meta") - med_d).abs()
            + (pl.col(COL_TEMPO_PROVA) - med_t).abs() / 50.0
        )
        ranqueado = sub.with_columns(score.alias("_score")).sort("_score")
        # Empate: amostra estável entre os 5 melhores
        top = ranqueado.head(min(5, ranqueado.height))
        idx = int(rng.integers(0, top.height))
        row = top.row(idx, named=True)

        mix = ", ".join(
            f"{tipo} {row[f'prop_{tipo}'] * 100:.0f}%" for tipo in TIPOS
        )
        meta_txt = ", ".join(f"{t} {p * 100:.0f}%" for t, p in meta_uso.items())
        escolhidos.append(
            {
                "tercil": rotulo,
                "rotulo": {
                    "alta": "Alta aderência (perto da meta)",
                    "media": "Média aderência",
                    "baixa": "Baixa aderência (longe da meta)",
                }[rotulo],
                "athlete_id": int(row[COL_ATHLETE]),
                "tempo_prova_min": round(float(row[COL_TEMPO_PROVA]), 1),
                "distancia_meta": round(float(row["distancia_meta"]), 4),
                "mix_treino": mix,
                "meta_referencia": meta_txt,
                "prop_leve": round(float(row["prop_leve"]), 4),
                "prop_longao": round(float(row["prop_longao"]), 4),
                "prop_ritmo": round(float(row["prop_ritmo"]), 4),
                "prop_intervalado": round(float(row["prop_intervalado"]), 4),
                "porque": (
                    f"Mix {mix}. Distância à meta = {row['distancia_meta']:.3f} "
                    f"(meta: {meta_txt}). Tempo ~21 km = "
                    f"{row[COL_TEMPO_PROVA]:.0f} min."
                ),
            }
        )

    return pl.DataFrame(escolhidos)


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
