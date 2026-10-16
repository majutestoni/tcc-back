from __future__ import annotations

import polars as pl

from dataset.constants import (
    COL_ATHLETE,
    COL_DISTANCE,
    COL_PACE,
    COL_TIPO_TREINO,
    SEMANAS_JANELA,
)
from dataset.models.previsao_treino.model import (
    PARAMETROS,
    PESOS_DEFICIT,
    _gerar_recomendacoes,
    calibrar_parametros,
    distribuicao_por_atleta,
    filtrar_janela_temporal,
)
from dataset.preprocessamento import formatar_delta_pace, formatar_pace

NOMES_TIPO = {
    "leve": "leve (regenerativo)",
    "longao": "longão",
    "ritmo": "ritmo",
    "intervalado": "intervalado",
}

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

# Fallback quando o atleta ainda não tem perfil_cluster (ADR-024: sem meta global).
META_CLUSTER_PADRAO = META_POR_PERFIL["volume_moderado"]


def meta_para_perfil(chave: str | None) -> dict[str, float]:
    """Meta de distribuição do perfil_cluster; sem chave → volume_moderado."""
    if chave and chave in META_POR_PERFIL:
        return META_POR_PERFIL[chave]
    return META_CLUSTER_PADRAO


def chave_perfil_cluster(perfil: pl.DataFrame, athlete_id: int) -> str | None:
    """Lê `perfil_cluster` do atleta num DataFrame de perfil já nomeado."""
    if perfil.height == 0 or "perfil_cluster" not in perfil.columns:
        return None
    info = perfil.filter(pl.col(COL_ATHLETE) == athlete_id)
    if info.height == 0:
        return None
    return info["perfil_cluster"][0]


def montar_perfil_cluster(df: pl.DataFrame) -> pl.DataFrame:
    """KMeans por gênero + nomeação (mesmo pipeline da demo Streamlit)."""
    from dataset.categorizar.kmeans import treina_kmeans_por_genero
    from dataset.categorizar.nomear_clusters import nomear_categorias

    resultado = treina_kmeans_por_genero(
        df,
        features=[COL_DISTANCE, COL_PACE],
        n_clusters=2,
        plotar=False,
        verbose=False,
    )
    return nomear_categorias(resultado.df_exibir)


def meta_do_atleta(df: pl.DataFrame, athlete_id: int) -> dict[str, float]:
    """Meta do cluster do atleta (volume_moderado se sem perfil)."""
    perfil = montar_perfil_cluster(df)
    return meta_para_perfil(chave_perfil_cluster(perfil, athlete_id))


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
        meta = meta_para_perfil(chave)
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


def _pct(fracao: float) -> str:
    return f"{fracao * 100:.0f}%"


def _tipo_maior_deficit(
    deficits: dict[str, float],
    pesos: dict[str, float] | None = None,
) -> str:
    pesos_uso = pesos if pesos is not None else PESOS_DEFICIT
    return max(deficits, key=lambda t: deficits[t] * pesos_uso.get(t, 1.0))


def _mediana_dist_km(
    corridas: pl.DataFrame | None,
    tipo: str,
    col_tipo: str = "tipo",
    col_dist: str = "distancia_km",
) -> float | None:
    if corridas is None or corridas.height == 0:
        return None
    if col_tipo not in corridas.columns or col_dist not in corridas.columns:
        return None
    sub = corridas.filter(pl.col(col_tipo) == tipo)
    if sub.height == 0:
        return None
    return round(float(sub[col_dist].median()), 1)


def explicar_recomendacao(
    *,
    rec: dict,
    atual: dict[str, float],
    meta: dict[str, float],
    deficits: dict[str, float],
    ultimos: list[str],
    corridas_janela: pl.DataFrame | None = None,
    info_cluster: dict | None = None,
    parametros_tipo: dict | None = None,
) -> str:
    """
    Texto descritivo do porquê da recomendação (tipo, distância, pace, tempo)
    para aquele atleta — sem jargão de ADR.
    """
    if not rec or "tipo" not in rec:
        return ""

    tipo = rec["tipo"]
    nome = NOMES_TIPO.get(tipo, tipo)
    info = info_cluster or {}
    rotulo = (
        info.get("rotulo_cluster")
        or rec.get("rotulo_cluster")
        or info.get("perfil_cluster")
        or "seu cluster"
    )
    perfil_chave = info.get("perfil_cluster") or rec.get("perfil_cluster") or ""

    tipo_bruto = _tipo_maior_deficit(deficits) if deficits else tipo
    atual_pct = atual.get(tipo, 0.0)
    meta_pct = meta.get(tipo, 0.0)
    deficit_tipo = deficits.get(tipo, meta_pct - atual_pct)

    partes: list[str] = []

    # --- Tipo ---
    if deficit_tipo > 0.01:
        partes.append(
            f"**Tipo — {nome}.** Nas últimas {SEMANAS_JANELA} semanas, cerca de "
            f"{_pct(atual_pct)} das corridas foram `{tipo}`, enquanto a meta do "
            f"cluster **{rotulo}** pede {_pct(meta_pct)}. O déficit "
            f"({_pct(deficit_tipo)}) é o maior entre os tipos, então o próximo "
            f"treino prioriza **{nome}**."
        )
    elif deficit_tipo < -0.01:
        partes.append(
            f"**Tipo — {nome}.** Você já está acima da meta de `{tipo}` "
            f"({_pct(atual_pct)} vs {_pct(meta_pct)} no cluster **{rotulo}**). "
            f"Ainda assim o algoritmo apontou **{nome}** "
            f"(em geral por empate ou pela regra anti-fadiga)."
        )
    else:
        partes.append(
            f"**Tipo — {nome}.** A distribuição na janela está próxima da meta "
            f"do cluster **{rotulo}** ({_pct(atual_pct)} vs {_pct(meta_pct)} "
            f"de `{tipo}`). O próximo treino segue como **{nome}**."
        )

    if tipo_bruto != tipo and ultimos:
        ultimo = ultimos[-1]
        nome_bruto = NOMES_TIPO.get(tipo_bruto, tipo_bruto)
        partes.append(
            f"O déficit puro apontava **{nome_bruto}**, mas o último treino da "
            f"janela foi `{ultimo}`: a guarda anti-fadiga evita repetir longão/"
            f"intervalado pesado e troca para **{nome}**."
        )
    elif ultimos:
        partes.append(
            f"O último treino na janela foi `{ultimos[-1]}` — a regra "
            f"anti-fadiga não bloqueou este `{tipo}`."
        )

    if perfil_chave:
        partes.append(
            f"A meta usada é a do perfil `{perfil_chave}` "
            f"(leve {_pct(meta.get('leve', 0))}, longão {_pct(meta.get('longao', 0))}, "
            f"ritmo {_pct(meta.get('ritmo', 0))}, intervalado "
            f"{_pct(meta.get('intervalado', 0))})."
        )

    # --- Distância ---
    dist = float(rec.get("distancia_km") or 0)
    params = parametros_tipo or PARAMETROS.get(tipo, {})
    faixa = params.get("distancia_km")
    med_janela = _mediana_dist_km(corridas_janela, tipo)
    med_geral_janela = None
    if corridas_janela is not None and corridas_janela.height > 0:
        if "distancia_km" in corridas_janela.columns:
            med_geral_janela = round(float(corridas_janela["distancia_km"].median()), 1)

    if faixa and len(faixa) == 2:
        d_min, d_max = faixa
        trecho_faixa = (
            f"ponto médio da faixa calibrada nos longões/corridas `{tipo}` "
            f"do cluster (**{d_min:.1f}–{d_max:.1f} km**)"
            if tipo == "longao"
            else (
                f"ponto médio da faixa calibrada nas corridas `{tipo}` "
                f"do cluster (**{d_min:.1f}–{d_max:.1f} km**)"
            )
        )
    else:
        trecho_faixa = "valor calibrado no histórico do cluster"

    trecho_atleta = ""
    if med_janela is not None:
        trecho_atleta = (
            f" Nas suas corridas `{tipo}` desta janela, a distância mediana "
            f"fica em torno de **{med_janela:.1f} km**"
        )
        if med_geral_janela is not None and tipo == "longao":
            trecho_atleta += (
                f"; nas demais corridas da janela a mediana é "
                f"**{med_geral_janela:.1f} km** — o longão recomenda um volume "
                f"maior que o dia a dia"
            )
        trecho_atleta += "."
    elif med_geral_janela is not None:
        trecho_atleta = (
            f" Nesta janela você não tem corrida classificada como `{tipo}`; "
            f"suas corridas recentes ficam em torno de **{med_geral_janela:.1f} km** "
            f"de mediana."
        )

    partes.append(
        f"**Distância — {dist:.1f} km.** Esse alvo é o {trecho_faixa}."
        f"{trecho_atleta}"
    )

    # --- Pace ---
    pace_alvo = float(rec.get("pace_alvo_min_km") or 0)
    pace_medio = None
    if corridas_janela is not None and corridas_janela.height > 0:
        if "pace_min_km" in corridas_janela.columns:
            pace_medio = float(corridas_janela["pace_min_km"].mean())

    offset = params.get("pace_offset_min")
    if offset is None and pace_medio is not None:
        offset = round(pace_alvo - pace_medio, 2)

    if pace_medio is not None and offset is not None:
        if offset > 0.05:
            sentido = (
                f"mais lento que a sua média ({formatar_delta_pace(offset)}) — "
                f"ritmo confortável / regenerativo"
            )
        elif offset < -0.05:
            sentido = (
                f"mais rápido que a sua média ({formatar_delta_pace(offset)}) — "
                f"estímulo de velocidade"
            )
        else:
            sentido = "próximo do seu pace médio (treino de ritmo)"

        partes.append(
            f"**Pace alvo — {formatar_pace(pace_alvo)}.** Seu pace médio na "
            f"janela é **{formatar_pace(pace_medio)}**. Para `{tipo}`, o "
            f"offset do cluster deixa o alvo **{sentido}**, resultando em "
            f"**{formatar_pace(pace_alvo)}**."
        )
    else:
        partes.append(
            f"**Pace alvo — {formatar_pace(pace_alvo)}.** Calculado a partir "
            f"do pace médio do atleta na janela mais o offset calibrado para "
            f"`{tipo}` no cluster."
        )

    # --- Tempo ---
    tempo = float(rec.get("tempo_estimado_min") or dist * pace_alvo)
    partes.append(
        f"**Tempo estimado — ~{tempo:.0f} min.** É simplesmente "
        f"**{dist:.1f} km × {formatar_pace(pace_alvo)}** "
        f"(distância × pace alvo), para dimensionar a sessão."
    )

    descricao = rec.get("descricao") or params.get("descricao")
    if descricao:
        partes.append(f"_{descricao}._")

    return "\n\n".join(partes)


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
        meta_uso = meta_para_perfil(row.get("perfil_cluster"))

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
