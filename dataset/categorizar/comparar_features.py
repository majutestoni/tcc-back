from __future__ import annotations

import numpy as np
import polars as pl

from dataset.categorizar.perfil_atletas import montar_perfil_atletas
from dataset.constants import (
    COL_DISTANCE,
    COL_ELEVATION,
    COL_FREQUENCIA,
    COL_GENDER,
    COL_HR,
    COL_N_CORRIDAS,
    COL_PACE,
)

_LABEL_FEATURE = {
    COL_DISTANCE: "distância",
    COL_PACE: "pace",
    COL_HR: "FC",
    COL_FREQUENCIA: "frequência",
    COL_N_CORRIDAS: "nº corridas",
    COL_ELEVATION: "elevação",
}


def _titulo_combinacao(features: list[str]) -> str:
    return " + ".join(_LABEL_FEATURE.get(col, col) for col in features)


def comparar_combinacoes_features(
    df: pl.DataFrame,
    combinacoes: dict[str, list[str]],
    n_clusters: int = 2,
    por_genero: bool = True,
) -> pl.DataFrame:
    """
    Compara combinações de features para KMeans (k fixo).

    Retorna métricas agregadas e por gênero para escolher o melhor conjunto.
    """
    from dataset.categorizar.kmeans import treina_kmeans, treina_kmeans_por_genero

    perfil = montar_perfil_atletas(df)
    linhas: list[dict] = []

    for chave, features in combinacoes.items():
        faltando = [f for f in features if f not in perfil.columns]
        if faltando:
            continue

        if por_genero:
            resultado = treina_kmeans_por_genero(
                df,
                features=features,
                n_clusters=n_clusters,
                plotar=False,
                verbose=False,
            )
            contagens = [c["atletas"] for c in resultado.detalhes["contagem_clusters"]]
            dbs: list[float] = []

            for genero, m in resultado.detalhes["metricas_por_genero"].items():
                rotulo = "Masculino" if genero == "M" else "Feminino"
                if m.get("davies_bouldin") is not None:
                    dbs.append(m["davies_bouldin"])
                linhas.append({
                    "combinacao": chave,
                    "titulo": _titulo_combinacao(features),
                    "n_features": len(features),
                    "genero": rotulo,
                    "silhouette": round(m["silhouette"], 4),
                    "davies_bouldin": round(m["davies_bouldin"], 4)
                    if m.get("davies_bouldin") is not None
                    else None,
                    "min_cluster": min(
                        c["atletas"]
                        for c in resultado.detalhes["contagem_clusters"]
                        if c["genero"] == genero
                    ),
                    "max_cluster": max(
                        c["atletas"]
                        for c in resultado.detalhes["contagem_clusters"]
                        if c["genero"] == genero
                    ),
                })

            linhas.append({
                "combinacao": chave,
                "titulo": _titulo_combinacao(features),
                "n_features": len(features),
                "genero": "Ponderado",
                "silhouette": round(resultado.metricas["silhouette"], 4),
                "davies_bouldin": round(float(np.mean(dbs)), 4) if dbs else None,
                "min_cluster": min(contagens) if contagens else 0,
                "max_cluster": max(contagens) if contagens else 0,
            })
        else:
            resultado = treina_kmeans(
                df,
                features=features,
                n_clusters=n_clusters,
                plotar=False,
                verbose=False,
            )
            contagens = [c["atletas"] for c in resultado.detalhes["contagem_clusters"]]
            linhas.append({
                "combinacao": chave,
                "titulo": _titulo_combinacao(features),
                "n_features": len(features),
                "genero": "Global",
                "silhouette": round(resultado.metricas["silhouette"], 4),
                "davies_bouldin": round(resultado.metricas.get("davies_bouldin") or 0, 4)
                if resultado.metricas.get("davies_bouldin") is not None
                else None,
                "min_cluster": min(contagens) if contagens else 0,
                "max_cluster": max(contagens) if contagens else 0,
            })

    return pl.DataFrame(linhas)


def comparar_combinacoes_e_k(
    df: pl.DataFrame,
    combinacoes: dict[str, list[str]],
    ks: tuple[int, ...] = (2, 3, 4),
) -> pl.DataFrame:
    """
    Grid combinação × k × gênero com métricas internas e equilíbrio dos clusters.

    Usado para escolher features e k antes da clusterização final.
    """
    from dataset.categorizar.validacao_cluster import avaliar_k

    perfil_base = montar_perfil_atletas(df).with_columns(
        pl.col(COL_GENDER).cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    )
    linhas: list[dict] = []

    for chave, features in combinacoes.items():
        faltando = [f for f in features if f not in perfil_base.columns]
        if faltando:
            continue

        for genero in ("M", "F"):
            subset = perfil_base.filter(pl.col(COL_GENDER) == genero)
            if subset.height < 2:
                continue

            rotulo = "Masculino" if genero == "M" else "Feminino"
            grid = avaliar_k(subset, features, ks=ks)
            if grid.height == 0:
                continue

            for row in grid.iter_rows(named=True):
                linhas.append({
                    "combinacao": chave,
                    "titulo": _titulo_combinacao(features),
                    "n_features": len(features),
                    "genero": rotulo,
                    "k": row["k"],
                    "silhouette": round(row["silhouette"], 4),
                    "davies_bouldin": round(row["davies_bouldin"], 4)
                    if row.get("davies_bouldin") is not None
                    else None,
                    "calinski_harabasz": round(row["calinski_harabasz"], 4)
                    if row.get("calinski_harabasz") is not None
                    else None,
                    "min_cluster": row["min_cluster"],
                    "max_cluster": row["max_cluster"],
                })

    return pl.DataFrame(linhas)


def sugerir_melhor_config(
    grid: pl.DataFrame,
    min_cluster: int = 5,
) -> pl.DataFrame:
    """
    Sugere (combinação, k) por gênero priorizando silhouette e min_cluster >= limiar.
    """
    linhas: list[dict] = []
    for genero in grid["genero"].unique().to_list():
        sub = grid.filter(pl.col("genero") == genero)
        validos = sub.filter(pl.col("min_cluster") >= min_cluster)
        candidatos = validos if validos.height > 0 else sub
        melhor = candidatos.sort("silhouette", descending=True).row(0, named=True)
        linhas.append({
            "genero": genero,
            "combinacao": melhor["combinacao"],
            "titulo": melhor["titulo"],
            "k": melhor["k"],
            "silhouette": melhor["silhouette"],
            "min_cluster": melhor["min_cluster"],
            "max_cluster": melhor["max_cluster"],
            "criterio": "ok" if melhor["min_cluster"] >= min_cluster else "min_cluster baixo",
        })
    return pl.DataFrame(linhas)
