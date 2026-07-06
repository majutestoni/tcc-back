from dataset.constants import *
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

META = {
    "leve": 0.60,
    "longao": 0.20,
    "ritmo": 0.12,
    "intervalado": 0.08,
}

PESOS_DEFICIT = {
    "leve": 1.0,
    "longao": 1.1,
    "ritmo": 1.0,
    "intervalado": 1.2,
}

PARAMETROS = {
    "leve": {
        "distancia_km": (5.0, 8.0),
        "pace_offset_min": 0.75,
        "descricao": "Corrida leve / regenerativa",
    },
    "longao": {
        "distancia_km": (12.0, 18.0),
        "pace_offset_min": 0.45,
        "descricao": "Longão em ritmo confortável",
    },
    "ritmo": {
        "distancia_km": (6.0, 10.0),
        "pace_offset_min": 0.0,
        "descricao": "Treino de ritmo de prova",
    },
    "intervalado": {
        "distancia_km": (5.0, 8.0),
        "pace_offset_min": -0.35,
        "descricao": "Treino intervalado (tiros)",
    },
}


def filtrar_janela_temporal(df: pl.DataFrame, semanas: int = SEMANAS_JANELA) -> pl.DataFrame:
    """Mantém apenas corridas das últimas N semanas (por data máxima do dataset)."""
    df = df.with_columns(
        pl.col(COL_TIMESTAMP)
        .str.strptime(pl.Datetime, format="%d/%m/%Y %H:%M", strict=False)
        .alias("_ts")
    )
    data_max = df["_ts"].max()
    if data_max is None:
        return df.drop("_ts")

    limite = data_max - pl.duration(weeks=semanas)
    return df.filter(pl.col("_ts") >= limite).drop("_ts")


def calibrar_parametros(
    df: pl.DataFrame,
    col_tipo: str = COL_TIPO_TREINO,
) -> dict:
    """Calibra distância e offset de pace a partir das medianas por tipo no histórico."""
    pace_global = df[COL_PACE].median()
    calibrados = {}

    for tipo, base in PARAMETROS.items():
        subset = df.filter(pl.col(col_tipo) == tipo)
        if subset.height < 5:
            calibrados[tipo] = base.copy()
            continue

        dist_km = (subset[COL_DISTANCE] / 1000).median()
        pace_tipo = subset[COL_PACE].median()
        margem = max(dist_km * 0.15, 0.5)

        calibrados[tipo] = {
            "distancia_km": (
                round(max(3.0, dist_km - margem), 1),
                round(dist_km + margem, 1),
            ),
            "pace_offset_min": round(pace_tipo - pace_global, 2),
            "descricao": base["descricao"],
        }

    return calibrados


def calcular_deficit(meta: dict[str, float], atual: dict[str, float]) -> dict[str, float]:
    return {tipo: meta[tipo] - atual.get(tipo, 0.0) for tipo in meta}


def calcular_erro_distribuicao(atual: dict[str, float], meta: dict[str, float] = META) -> float:
    """Soma das diferenças absolutas entre distribuição atual e meta (menor = melhor)."""
    return sum(abs(atual.get(tipo, 0.0) - meta[tipo]) for tipo in meta)


def recomendar_proximo_treino(
    deficits: dict[str, float],
    ultimos_treinos: list[str],
    pace_medio_atleta: float,
    parametros: dict | None = None,
) -> dict:
    params_map = parametros or PARAMETROS

    deficits_ponderados = {
        tipo: deficits[tipo] * PESOS_DEFICIT[tipo] for tipo in deficits
    }
    tipo = max(deficits_ponderados, key=deficits_ponderados.get)

    if ultimos_treinos:
        ultimo = ultimos_treinos[-1]
        if ultimo == "intervalado" and tipo in ("intervalado", "longao"):
            tipo = "leve"
        elif ultimo == "longao" and tipo == "longao":
            tipo = "leve"

    params = params_map[tipo]
    dist_min, dist_max = params["distancia_km"]
    distancia_km = (dist_min + dist_max) / 2
    pace_alvo = pace_medio_atleta + params["pace_offset_min"]

    return {
        "tipo": tipo,
        "descricao": params["descricao"],
        "distancia_km": round(distancia_km, 1),
        "pace_alvo_min_km": round(pace_alvo, 2),
        "tempo_estimado_min": round(distancia_km * pace_alvo, 1),
        "deficits": deficits,
    }


def distribuicao_por_atleta(
    df: pl.DataFrame,
    col_tipo: str = COL_TIPO_TREINO,
) -> pl.DataFrame:
    return (
        df.group_by([COL_ATHLETE, col_tipo])
        .agg(pl.len().alias("qtd"))
        .with_columns(
            (pl.col("qtd") / pl.col("qtd").sum().over(COL_ATHLETE)).alias("percentual")
        )
    )


def ultimos_treinos_por_atleta(
    df: pl.DataFrame,
    col_tipo: str = COL_TIPO_TREINO,
    n: int = 3,
) -> dict:
    df_ord = df.sort(COL_TIMESTAMP)
    ultimos = {}
    for athlete_id in df_ord[COL_ATHLETE].unique().to_list():
        tipos = (
            df_ord.filter(pl.col(COL_ATHLETE) == athlete_id)
            .select(col_tipo)
            .to_series()
            .to_list()
        )
        ultimos[athlete_id] = tipos[-n:]
    return ultimos


def _gerar_recomendacoes(
    df: pl.DataFrame,
    col_tipo: str,
    parametros: dict,
    athlete_id: int | None = None,
    metodo: str = "regras",
) -> pl.DataFrame:
    df_janela = filtrar_janela_temporal(df)

    pace_por_atleta = (
        df_janela.group_by(COL_ATHLETE)
        .agg(pl.col(COL_PACE).mean().alias("pace_medio_atleta"))
    )
    distribuicao = distribuicao_por_atleta(df_janela, col_tipo)
    ultimos = ultimos_treinos_por_atleta(df_janela, col_tipo)

    atletas = [athlete_id] if athlete_id is not None else df_janela[COL_ATHLETE].unique().to_list()
    recomendacoes = []

    for aid in atletas:
        dist_atleta = distribuicao.filter(pl.col(COL_ATHLETE) == aid)
        atual = {
            row[col_tipo]: row["percentual"]
            for row in dist_atleta.iter_rows(named=True)
        }
        deficits = calcular_deficit(META, atual)
        pace_row = pace_por_atleta.filter(pl.col(COL_ATHLETE) == aid)
        if pace_row.height == 0:
            continue

        pace_medio = pace_row["pace_medio_atleta"][0]
        rec = recomendar_proximo_treino(
            deficits=deficits,
            ultimos_treinos=ultimos.get(aid, []),
            pace_medio_atleta=pace_medio,
            parametros=parametros,
        )
        rec[COL_ATHLETE] = aid
        rec["metodo"] = metodo
        rec["erro_distribuicao"] = round(calcular_erro_distribuicao(atual), 3)
        recomendacoes.append(rec)

    return pl.DataFrame(recomendacoes)


def exibir_exemplos_recomendacao(
    df: pl.DataFrame,
    resultado: pl.DataFrame,
    n: int = 5,
    col_tipo: str = COL_TIPO_TREINO,
) -> None:
    print("\n" + "=" * 60)
    print(f"EXEMPLOS DE RECOMENDAÇÃO ({n} atletas) — janela {SEMANAS_JANELA} semanas")
    print("=" * 60)

    df_janela = filtrar_janela_temporal(df)
    distribuicao = distribuicao_por_atleta(df_janela, col_tipo)

    for row in resultado.head(n).iter_rows(named=True):
        aid = row[COL_ATHLETE]
        dist = distribuicao.filter(pl.col(COL_ATHLETE) == aid)

        print(f"\nAtleta {aid}")
        print("   Distribuição atual:")
        for d in dist.iter_rows(named=True):
            pct = d["percentual"] * 100
            meta_pct = META[d[col_tipo]] * 100
            print(f"      • {d[col_tipo]:12s}: {pct:5.1f}%  (meta: {meta_pct:.0f}%)")

        print(f"\n   Próximo treino recomendado: {row['tipo'].upper()}")
        print(f"      {row['descricao']}")
        print(f"      Distância: {row['distancia_km']:.1f} km")
        print(f"      Pace alvo: {row['pace_alvo_min_km']:.2f} min/km")
        print(f"      Tempo estimado: {row['tempo_estimado_min']:.0f} min")
        print(f"      Erro distribuição vs meta: {row.get('erro_distribuicao', '—')}")


def plotar_recomendacao_atleta(
    df: pl.DataFrame,
    resultado: pl.DataFrame,
    athlete_id: int,
    col_tipo: str = COL_TIPO_TREINO,
) -> None:
    df_janela = filtrar_janela_temporal(df)
    distribuicao = distribuicao_por_atleta(df_janela, col_tipo)
    dist = distribuicao.filter(pl.col(COL_ATHLETE) == athlete_id)

    tipos = list(META.keys())
    atual = [0.0] * len(tipos)
    for row in dist.iter_rows(named=True):
        idx = tipos.index(row[col_tipo])
        atual[idx] = row["percentual"] * 100

    meta = [META[t] * 100 for t in tipos]
    rec = resultado.filter(pl.col(COL_ATHLETE) == athlete_id)
    tipo_rec = rec["tipo"][0] if rec.height > 0 else "?"

    x = np.arange(len(tipos))
    width = 0.35
    _, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, atual, width, label="Atual (%)", color="steelblue")
    ax.bar(x + width / 2, meta, width, label="Meta (%)", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tipos)
    ax.set_ylabel("Percentual (%)")
    ax.set_title(f"Atleta {athlete_id} — últimas {SEMANAS_JANELA} semanas\nRecomendado: {tipo_rec}")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plotar_resumo_recomendacoes(resultado: pl.DataFrame) -> None:
    contagem = (
        resultado.group_by("tipo")
        .agg(pl.len().alias("qtd"))
        .sort("qtd", descending=True)
    )
    tipos = contagem["tipo"].to_list()
    qtds = contagem["qtd"].to_list()

    _, ax = plt.subplots(figsize=(7, 4))
    cores = {"leve": "#4CAF50", "longao": "#2196F3", "ritmo": "#FF9800", "intervalado": "#F44336"}
    ax.bar(tipos, qtds, color=[cores.get(t, "gray") for t in tipos])
    ax.set_xlabel("Tipo recomendado")
    ax.set_ylabel("Nº de atletas")
    ax.set_title("Recomendações de próximo treino (todos os atletas)")
    plt.tight_layout()
    plt.show()


def treina_modelo(
    df: pl.DataFrame,
    athlete_id: int | None = None,
    plotar: bool = True,
    verbose: bool = True,
):
    from dataset.models.resultado import ResultadoModelo
    from dataset.models.visualizacao import exibir_teste_recomendacao

    parametros = calibrar_parametros(df)

    if verbose:
        print("\nParâmetros calibrados (mediana por tipo):")
        for tipo, p in parametros.items():
            print(f"  {tipo}: dist={p['distancia_km']} km, pace_offset={p['pace_offset_min']:+.2f}")

    recomendacoes = _gerar_recomendacoes(
        df, COL_TIPO_TREINO, parametros, athlete_id, metodo="regras"
    )

    erro_medio = recomendacoes["erro_distribuicao"].mean() if recomendacoes.height > 0 else None

    resultado_modelo = ResultadoModelo(
        nome="Regras (recomendação)",
        categoria="treino",
        metricas={"erro_distribuicao_medio": float(erro_medio) if erro_medio is not None else 0.0},
        recomendacoes=recomendacoes,
        df_exibir=df,
        col_tipo=COL_TIPO_TREINO,
        parametros_calibrados=parametros,
    )

    if verbose or plotar:
        exibir_teste_recomendacao(resultado_modelo, n=5, plotar=plotar)

    return resultado_modelo
