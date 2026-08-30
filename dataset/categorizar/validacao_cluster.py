from __future__ import annotations

from typing import Iterable

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler

from dataset.categorizar.perfil_atletas import definir_perfil_atleta
from dataset.constants import (
    COL_ATHLETE,
    COL_CLUSTER_CORREDOR,
    COL_GENDER,
    COL_NIVEL_DISTANCIA,
    COL_NIVEL_PACE,
    COL_NIVEL_VOLUME,
    COL_PERFIL_ATLETA,
)
from dataset.models.resultado import ResultadoModelo


def metricas_internas(X_scaled: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    """Silhouette, Davies-Bouldin e Calinski-Harabasz (k >= 2)."""
    k = len(set(labels))
    if k < 2 or X_scaled.shape[0] < 2:
        return {
            "silhouette": 0.0,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }
    return {
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
    }


def metricas_externas(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Concordância entre clusters e rótulos de referência."""
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "homogeneity": float(homogeneity_score(y_true, y_pred)),
        "completeness": float(completeness_score(y_true, y_pred)),
        "v_measure": float(v_measure_score(y_true, y_pred)),
    }


def estabilidade_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int,
    n_runs: int = 20,
    random_state: int = 42,
) -> dict[str, float]:
    """ARI médio entre execuções consecutivas do KMeans (mesmo k)."""
    if n_clusters < 2 or X_scaled.shape[0] < n_clusters:
        return {"ari_medio": 1.0, "ari_std": 0.0, "n_runs": 0.0}

    labels_base: np.ndarray | None = None
    aris: list[float] = []
    for seed in range(random_state, random_state + n_runs):
        labels = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=10,
        ).fit_predict(X_scaled)
        if labels_base is None:
            labels_base = labels
        else:
            aris.append(adjusted_rand_score(labels_base, labels))

    if not aris:
        return {"ari_medio": 1.0, "ari_std": 0.0, "n_runs": 0.0}

    return {
        "ari_medio": float(np.mean(aris)),
        "ari_std": float(np.std(aris)),
        "n_runs": float(len(aris)),
    }


def _contagem_clusters(labels: np.ndarray) -> dict[str, int]:
    unicos, contagens = np.unique(labels, return_counts=True)
    return {str(u): int(c) for u, c in zip(unicos, contagens)}


def _equilibrio_clusters(contagem: dict[str, int]) -> dict[str, int]:
    if not contagem:
        return {"min_cluster": 0, "max_cluster": 0}
    valores = list(contagem.values())
    return {"min_cluster": min(valores), "max_cluster": max(valores)}


def avaliar_k(
    perfil: pl.DataFrame,
    features: list[str],
    ks: Iterable[int] = range(2, 8),
    random_state: int = 42,
) -> pl.DataFrame:
    """Grid de métricas internas para escolha de k."""
    if perfil.height < 2:
        return pl.DataFrame()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(perfil.select(features).to_numpy())
    linhas: list[dict] = []

    for k in ks:
        if k >= perfil.height:
            continue
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(
            X_scaled
        )
        metricas = metricas_internas(X_scaled, labels)
        equilibrio = _equilibrio_clusters(_contagem_clusters(labels))
        linhas.append({"k": k, **metricas, **equilibrio})

    return pl.DataFrame(linhas)


def tabela_contingencia(
    y_true: np.ndarray | pl.Series,
    y_pred: np.ndarray | pl.Series,
    rotulo_true: str = "referencia",
    rotulo_pred: str = "cluster",
) -> pl.DataFrame:
    """Tabela cruzada entre rótulos externos e clusters."""
    df = pl.DataFrame({rotulo_true: y_true, rotulo_pred: y_pred})
    refs = sorted(df[rotulo_true].unique().to_list())
    preds = sorted(df[rotulo_pred].unique().to_list())

    linhas: list[dict] = []
    for ref in refs:
        linha = {rotulo_true: ref}
        sub = df.filter(pl.col(rotulo_true) == ref)
        for pred in preds:
            linha[str(pred)] = sub.filter(pl.col(rotulo_pred) == pred).height
        linha["total"] = sub.height
        linhas.append(linha)

    return pl.DataFrame(linhas)


def comparar_com_perfil_regras(
    resultado: ResultadoModelo,
    df: pl.DataFrame,
    min_corridas: int | None = None,
    col_cluster: str = COL_CLUSTER_CORREDOR,
    referencias: list[str] | None = None,
) -> dict:
    """
    Cruza clusters KMeans com perfis por regras interpretáveis.

    Retorna comparação por atleta, métricas externas e tabelas de contingência.
    """
    if resultado.df_exibir is None:
        raise ValueError("Resultado KMeans sem df_exibir.")

    if referencias is None:
        referencias = [
            COL_PERFIL_ATLETA,
            COL_NIVEL_PACE,
            COL_NIVEL_VOLUME,
            COL_NIVEL_DISTANCIA,
        ]

    perfil_regras = definir_perfil_atleta(df, min_corridas=min_corridas, verbose=False)
    cols_cluster = [COL_ATHLETE, col_cluster]
    if COL_GENDER in resultado.df_exibir.columns:
        cols_cluster.append(COL_GENDER)
    if "categoria" in resultado.df_exibir.columns:
        cols_cluster.append("categoria")

    comparacao = (
        resultado.df_exibir.select(cols_cluster)
        .join(
            perfil_regras.select([COL_ATHLETE, *referencias]),
            on=COL_ATHLETE,
            how="inner",
        )
    )

    col_pred = "categoria" if "categoria" in comparacao.columns else col_cluster
    metricas_por_referencia: dict[str, dict[str, float]] = {}
    contingencias: dict[str, pl.DataFrame] = {}

    for ref in referencias:
        y_true = comparacao[ref].to_numpy()
        y_pred = comparacao[col_pred].to_numpy()
        metricas_por_referencia[ref] = metricas_externas(y_true, y_pred)
        contingencias[ref] = tabela_contingencia(y_true, y_pred, ref, col_pred)

    metricas_por_genero: dict[str, dict[str, dict[str, float]]] = {}
    if COL_GENDER in comparacao.columns:
        for genero in comparacao[COL_GENDER].unique().to_list():
            sub = comparacao.filter(pl.col(COL_GENDER) == genero)
            metricas_por_genero[str(genero)] = {
                ref: metricas_externas(
                    sub[ref].to_numpy(),
                    sub[col_pred].to_numpy(),
                )
                for ref in referencias
            }

    return {
        "comparacao": comparacao,
        "metricas_externas": metricas_por_referencia,
        "metricas_externas_por_genero": metricas_por_genero,
        "contingencias": contingencias,
        "col_pred": col_pred,
        "n_atletas": comparacao.height,
    }


def validar_resultado_kmeans(
    resultado: ResultadoModelo,
    df: pl.DataFrame | None = None,
    features: list[str] | None = None,
    min_corridas: int | None = None,
    n_runs_estabilidade: int = 20,
    random_state: int = 42,
) -> ResultadoModelo:
    """
    Enriquece um ResultadoModelo com validação interna, estabilidade e externa.

    Para resultados separados por gênero, calcula métricas internas por subset.
    """
    if resultado.df_exibir is None:
        return resultado

    perfil = resultado.df_exibir
    features = features or resultado.detalhes.get("features")
    if not features:
        raise ValueError("Informe features ou use resultado com detalhes['features'].")

    validacao_interna: dict[str, dict] = {}
    validacao_estabilidade: dict[str, dict] = {}

    if resultado.detalhes.get("separado_por_genero"):
        for genero in ("M", "F"):
            subset = perfil.filter(pl.col(COL_GENDER) == genero)
            if subset.height < 2:
                continue
            k = min(
                resultado.detalhes.get("n_clusters_por_genero", 2),
                subset.height,
            )
            X = StandardScaler().fit_transform(subset.select(features).to_numpy())
            labels = subset[COL_CLUSTER_CORREDOR].to_numpy()
            rotulo = "Masculino" if genero == "M" else "Feminino"
            validacao_interna[rotulo] = {
                "n_atletas": subset.height,
                "k": k,
                **metricas_internas(X, labels),
                **_equilibrio_clusters(_contagem_clusters(labels)),
            }
            validacao_estabilidade[rotulo] = estabilidade_kmeans(
                X, k, n_runs=n_runs_estabilidade, random_state=random_state
            )
    else:
        X = StandardScaler().fit_transform(perfil.select(features).to_numpy())
        labels = perfil[COL_CLUSTER_CORREDOR].to_numpy()
        k = resultado.detalhes.get("n_clusters", len(set(labels)))
        validacao_interna["global"] = {
            "n_atletas": perfil.height,
            "k": k,
            **metricas_internas(X, labels),
            **_equilibrio_clusters(_contagem_clusters(labels)),
        }
        validacao_estabilidade["global"] = estabilidade_kmeans(
            X, k, n_runs=n_runs_estabilidade, random_state=random_state
        )

    resultado.detalhes["validacao_interna"] = validacao_interna
    resultado.detalhes["validacao_estabilidade"] = validacao_estabilidade

    for chave, metricas in validacao_interna.items():
        prefixo = chave.lower().replace(" ", "_")
        for nome, valor in metricas.items():
            if valor is not None and isinstance(valor, (int, float)):
                resultado.metricas[f"{prefixo}_{nome}"] = float(valor)

    for chave, metricas in validacao_estabilidade.items():
        prefixo = chave.lower().replace(" ", "_")
        for nome, valor in metricas.items():
            if isinstance(valor, (int, float)):
                resultado.metricas[f"{prefixo}_estab_{nome}"] = float(valor)

    if df is not None:
        externa = comparar_com_perfil_regras(resultado, df, min_corridas=min_corridas)
        resultado.detalhes["validacao_externa"] = {
            "metricas": externa["metricas_externas"],
            "metricas_por_genero": externa["metricas_externas_por_genero"],
            "contingencias": {k: v.to_dicts() for k, v in externa["contingencias"].items()},
            "n_atletas": externa["n_atletas"],
            "col_pred": externa["col_pred"],
        }
        for ref, vals in externa["metricas_externas"].items():
            for nome, valor in vals.items():
                resultado.metricas[f"ext_{ref}_{nome}"] = float(valor)

    return resultado


def silhouette_por_amostra(
    perfil: pl.DataFrame,
    features: list[str],
    col_cluster: str = COL_CLUSTER_CORREDOR,
) -> pl.DataFrame:
    """Silhouette individual — útil para identificar atletas mal clusterizados."""
    if perfil.height < 2 or col_cluster not in perfil.columns:
        return perfil

    X = StandardScaler().fit_transform(perfil.select(features).to_numpy())
    labels = perfil[col_cluster].to_numpy()
    if len(set(labels)) < 2:
        return perfil.with_columns(pl.lit(0.0).alias("silhouette_amostra"))

    sil = silhouette_samples(X, labels)
    cols = [c for c in [COL_ATHLETE, col_cluster, COL_GENDER, "categoria"] if c in perfil.columns]
    return (
        perfil.select(cols)
        .with_columns(pl.Series("silhouette_amostra", sil))
        .sort("silhouette_amostra")
    )
