from __future__ import annotations

import polars as pl

from dataset.constants import COL_ATHLETE, COL_TIPO_TREINO, SEMANAS_JANELA
from dataset.models.previsao_treino.model import (
    META,
    PARAMETROS,
    _gerar_recomendacoes,
    calibrar_parametros,
    distribuicao_por_atleta,
    filtrar_janela_temporal,
)
from dataset.preprocessamento import formatar_pace

META_POR_PERFIL: dict[str, dict[str, float]] = {
    "volume_alto": {
        "leve": 0.50,
        "longao": 0.25,
        "ritmo": 0.15,
        "intervalado": 0.10,
    },
    "volume_moderado": {
        "leve": 0.70,
        "longao": 0.15,
        "ritmo": 0.10,
        "intervalado": 0.05,
    },
}


def calibrar_parametros_por_perfil(
    df: pl.DataFrame,
    perfil: pl.DataFrame,
    col_tipo: str = COL_TIPO_TREINO,
) -> dict[str, dict]:
    """Calibra distância/pace offset com o histórico de cada perfil_cluster."""
    if "perfil_cluster" not in perfil.columns:
        raise ValueError("perfil precisa da coluna perfil_cluster (rode nomear_categorias)")

    params: dict[str, dict] = {}
    for chave in perfil["perfil_cluster"].unique().to_list():
        ids = perfil.filter(pl.col("perfil_cluster") == chave)[COL_ATHLETE]
        df_sub = df.filter(pl.col(COL_ATHLETE).is_in(ids.to_list()))
        if df_sub.height < 10:
            params[chave] = {k: v.copy() for k, v in PARAMETROS.items()}
        else:
            params[chave] = calibrar_parametros(df_sub, col_tipo=col_tipo)
    return params


def gerar_recomendacoes_por_cluster(
    df: pl.DataFrame,
    perfil: pl.DataFrame,
    col_tipo: str = COL_TIPO_TREINO,
    athlete_id: int | None = None,
    metodo: str = "regras_cluster",
) -> pl.DataFrame:
    """
    Recomenda próximo treino com META e parâmetros calibrados pelo perfil_cluster.
    """
    cols_perfil = [
        COL_ATHLETE,
        "categoria",
        "perfil_cluster",
        "nome_cluster",
        "rotulo_cluster",
    ]
    faltando = [c for c in cols_perfil if c not in perfil.columns]
    if faltando:
        raise ValueError(f"perfil incompleto; faltam colunas: {faltando}")

    params_por_perfil = calibrar_parametros_por_perfil(df, perfil, col_tipo=col_tipo)
    mapa = perfil.select(cols_perfil).unique(subset=[COL_ATHLETE])

    atletas = (
        [athlete_id]
        if athlete_id is not None
        else mapa[COL_ATHLETE].to_list()
    )

    blocos: list[pl.DataFrame] = []
    for aid in atletas:
        info = mapa.filter(pl.col(COL_ATHLETE) == aid)
        if info.height == 0:
            continue
        chave = info["perfil_cluster"][0]
        meta = META_POR_PERFIL.get(chave, META)
        parametros = params_por_perfil.get(chave) or {
            k: v.copy() for k, v in PARAMETROS.items()
        }
        rec = _gerar_recomendacoes(
            df,
            col_tipo=col_tipo,
            parametros=parametros,
            athlete_id=aid,
            metodo=metodo,
            meta=meta,
        )
        if rec.height == 0:
            continue
        rec = rec.with_columns(
            pl.lit(info["categoria"][0]).alias("categoria"),
            pl.lit(chave).alias("perfil_cluster"),
            pl.lit(info["nome_cluster"][0]).alias("nome_cluster"),
            pl.lit(info["rotulo_cluster"][0]).alias("rotulo_cluster"),
        )
        blocos.append(rec)

    if not blocos:
        return pl.DataFrame()
    return pl.concat(blocos).sort(COL_ATHLETE)


def comparar_recomendacoes(
    rec_global: pl.DataFrame,
    rec_cluster: pl.DataFrame,
) -> pl.DataFrame:
    """Junta baseline global × personalizada por cluster."""
    g = rec_global.select(
        COL_ATHLETE,
        pl.col("tipo").alias("tipo_global"),
        pl.col("distancia_km").alias("dist_global"),
        pl.col("pace_alvo_min_km").alias("pace_global"),
        pl.col("erro_distribuicao").alias("erro_global"),
    )
    c = rec_cluster.select(
        COL_ATHLETE,
        "categoria",
        "perfil_cluster",
        "rotulo_cluster",
        pl.col("tipo").alias("tipo_cluster"),
        pl.col("distancia_km").alias("dist_cluster"),
        pl.col("pace_alvo_min_km").alias("pace_cluster"),
        pl.col("erro_distribuicao").alias("erro_cluster"),
    )
    return (
        g.join(c, on=COL_ATHLETE, how="inner")
        .with_columns(
            (pl.col("tipo_global") != pl.col("tipo_cluster")).alias("tipo_mudou"),
            (pl.col("dist_cluster") - pl.col("dist_global"))
            .round(2)
            .alias("delta_dist_km"),
            (pl.col("pace_cluster") - pl.col("pace_global"))
            .round(3)
            .alias("delta_pace"),
        )
        .sort("rotulo_cluster", COL_ATHLETE)
    )


def resumo_comparacao(comparacao: pl.DataFrame) -> dict[str, float | int]:
    if comparacao.height == 0:
        return {"n_atletas": 0}
    return {
        "n_atletas": comparacao.height,
        "pct_tipo_mudou": round(100 * comparacao["tipo_mudou"].mean(), 1),
        "delta_dist_medio_km": round(comparacao["delta_dist_km"].mean(), 2),
        "delta_pace_medio": round(comparacao["delta_pace"].mean(), 3),
        "erro_global_medio": round(comparacao["erro_global"].mean(), 3),
        "erro_cluster_medio": round(comparacao["erro_cluster"].mean(), 3),
    }


def exibir_card_recomendacao(
    df: pl.DataFrame,
    recomendacao: pl.DataFrame,
    athlete_id: int,
    col_tipo: str = COL_TIPO_TREINO,
    meta: dict[str, float] | None = None,
) -> None:
    row_df = recomendacao.filter(pl.col(COL_ATHLETE) == athlete_id)
    if row_df.height == 0:
        print(f"Sem recomendação para atleta {athlete_id}")
        return

    row = row_df.row(0, named=True)
    meta_uso = meta
    if meta_uso is None:
        chave = row.get("perfil_cluster")
        meta_uso = META_POR_PERFIL.get(chave, META) if chave else META

    df_janela = filtrar_janela_temporal(df)
    dist = distribuicao_por_atleta(df_janela, col_tipo).filter(
        pl.col(COL_ATHLETE) == athlete_id
    )

    rotulo = row.get("rotulo_cluster") or row.get("categoria") or "—"
    print("=" * 60)
    print(f"Atleta {athlete_id}  ·  {rotulo}")
    print(f"Janela: últimas {SEMANAS_JANELA} semanas")
    print("-" * 60)
    print("Distribuição atual vs meta do perfil:")
    for d in dist.iter_rows(named=True):
        tipo = d[col_tipo]
        pct = d["percentual"] * 100
        meta_pct = meta_uso.get(tipo, 0.0) * 100
        print(f"  • {tipo:12s}: {pct:5.1f}%  (meta: {meta_pct:.0f}%)")
    print("-" * 60)
    print(f"Próximo treino: {row['tipo'].upper()}")
    print(f"  {row['descricao']}")
    print(f"  Distância:      {row['distancia_km']:.1f} km")
    print(f"  Pace alvo:      {formatar_pace(row['pace_alvo_min_km'])}")
    print(f"  Tempo estimado: {row['tempo_estimado_min']:.0f} min")
    print(f"  Erro vs meta:   {row.get('erro_distribuicao', '—')}")
    print("=" * 60)


def atletas_exemplo_por_perfil(perfil: pl.DataFrame, seed: int = 42) -> list[int]:
    """Um atleta por categoria (estratificado) para vitrine do TCC."""
    ids: list[int] = []
    for cat in sorted(perfil["categoria"].unique().to_list()):
        subset = perfil.filter(pl.col("categoria") == cat)
        amostra = subset.sample(n=min(1, subset.height), seed=seed)
        ids.extend(amostra[COL_ATHLETE].to_list())
    return ids
