from __future__ import annotations

import polars as pl

from dataset.constants import COL_DISTANCE, COL_FREQUENCIA, COL_GENDER, COL_PACE
from dataset.preprocessamento import formatar_pace

PERFIS_CLUSTER = {
    "volume_alto": {
        "nome": "Volume alto",
        "descricao": (
            "Distância mediana maior, pace mais rápido e frequência semanal "
            "mais alta — perfil de treino mais consistente."
        ),
    },
    "volume_moderado": {
        "nome": "Volume moderado",
        "descricao": (
            "Distância mediana menor, pace mais lento e frequência semanal "
            "mais baixa — perfil mais recreativo ou ocasional."
        ),
    },
}


def _chave_perfil(rank: int, n_clusters: int) -> str:
    if n_clusters == 2:
        return "volume_alto" if rank == 0 else "volume_moderado"
    if rank == 0:
        return "volume_alto"
    if rank == n_clusters - 1:
        return "volume_moderado"
    return "volume_moderado"


def nomear_categorias(perfil: pl.DataFrame) -> pl.DataFrame:
    """
    Atribui nome e descrição de negócio a cada categoria (cluster × gênero).

    Dentro de cada gênero, ordena clusters pela distância mediana (maior = volume alto).
    Assim o rótulo não depende do índice bruto do KMeans (C0/C1).
    """
    if "categoria" not in perfil.columns:
        raise ValueError("perfil precisa da coluna 'categoria'")

    resumo = (
        perfil.group_by("categoria", COL_GENDER)
        .agg(
            pl.len().alias("atletas"),
            pl.col(COL_DISTANCE).median().alias("_dist_med"),
            pl.col(COL_PACE).median().alias("_pace_med"),
            pl.col(COL_FREQUENCIA).median().alias("_freq_med")
            if COL_FREQUENCIA in perfil.columns
            else pl.lit(None).alias("_freq_med"),
        )
        .with_columns(
            pl.col("_dist_med")
            .rank("ordinal", descending=True)
            .over(COL_GENDER)
            .alias("_rank_dist"),
            pl.col("categoria").n_unique().over(COL_GENDER).alias("_n_clusters"),
        )
    )

    mapas: list[dict] = []
    for row in resumo.iter_rows(named=True):
        chave = _chave_perfil(int(row["_rank_dist"]) - 1, int(row["_n_clusters"]))
        meta = PERFIS_CLUSTER[chave]
        genero = "Masculino" if row[COL_GENDER] == "M" else "Feminino"
        mapas.append(
            {
                "categoria": row["categoria"],
                "perfil_cluster": chave,
                "nome_cluster": meta["nome"],
                "rotulo_cluster": f"{meta['nome']} · {genero}",
                "descricao_cluster": meta["descricao"],
                "atletas": row["atletas"],
                "dist_km_med": round(row["_dist_med"] / 1000, 2),
                "pace_med": formatar_pace(row["_pace_med"]),
                "freq_semana_med": (
                    round(row["_freq_med"], 2) if row["_freq_med"] is not None else None
                ),
            }
        )

    mapa_df = pl.DataFrame(mapas)
    cols_join = [
        "categoria",
        "perfil_cluster",
        "nome_cluster",
        "rotulo_cluster",
        "descricao_cluster",
    ]
    return perfil.join(mapa_df.select(cols_join), on="categoria", how="left")


def tabela_descritiva_clusters(perfil: pl.DataFrame) -> pl.DataFrame:
    """Tabela pronta para o TCC: id técnico, nome, métricas e descrição."""
    if "nome_cluster" not in perfil.columns:
        perfil = nomear_categorias(perfil)

    return (
        perfil.group_by(
            "categoria",
            COL_GENDER,
            "perfil_cluster",
            "nome_cluster",
            "rotulo_cluster",
            "descricao_cluster",
        )
        .agg(
            pl.len().alias("atletas"),
            (pl.col(COL_DISTANCE).median() / 1000).round(2).alias("dist_km_med"),
            pl.col(COL_PACE).median().alias("_pace_med"),
            pl.col(COL_FREQUENCIA).median().round(2).alias("freq_semana_med")
            if COL_FREQUENCIA in perfil.columns
            else pl.lit(None).alias("freq_semana_med"),
        )
        .with_columns(
            pl.col("_pace_med")
            .map_elements(formatar_pace, return_dtype=pl.Utf8)
            .alias("pace_med"),
        )
        .drop("_pace_med")
        .sort("categoria")
    )
