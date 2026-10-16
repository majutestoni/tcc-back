from __future__ import annotations

import polars as pl

from dataset.categorizar.definir_perfil_atleta import definir_perfil_atleta
from dataset.categorizar.perfil_atletas import montar_perfil_atletas
from dataset.constants import (
    COL_ATHLETE,
    COL_DISTANCE,
    COL_FREQUENCIA,
    COL_GENDER,
    COL_HR,
    COL_N_CORRIDAS,
    COL_PACE,
    COL_PERFIL_ATLETA,
    COL_TIPO_TREINO,
)
from dataset.models.previsao_treino.model import (
    SEMANAS_JANELA,
    calcular_deficit,
    distribuicao_por_atleta,
    filtrar_janela_temporal,
    ultimos_treinos_por_atleta,
)
from dataset.models.previsao_treino.recomendacao_cluster import meta_do_atleta
from gemini.schemas import FeaturesAtleta


def montar_features_atleta(df: pl.DataFrame, athlete_id: int) -> FeaturesAtleta:
    """Constrói o resumo determinístico de um atleta para enviar ao Gemini.

    Reaproveita janela temporal, distribuição e déficits vs meta do cluster
    (ADR-024) e o perfil interpretável por regras.
    """
    perfil = montar_perfil_atletas(df).filter(pl.col(COL_ATHLETE) == athlete_id)
    if perfil.height == 0:
        raise ValueError(
            f"Atleta {athlete_id} não encontrado após preprocessamento."
        )
    linha = perfil.row(0, named=True)

    perfil_regras = definir_perfil_atleta(df, verbose=False)
    linha_regras = perfil_regras.filter(pl.col(COL_ATHLETE) == athlete_id)
    perfil_label = (
        linha_regras[COL_PERFIL_ATLETA][0] if linha_regras.height > 0 else None
    )

    meta = meta_do_atleta(df, athlete_id)

    df_janela = filtrar_janela_temporal(df)
    distribuicao = distribuicao_por_atleta(df_janela).filter(
        pl.col(COL_ATHLETE) == athlete_id
    )
    atual = {
        item[COL_TIPO_TREINO]: round(item["percentual"], 4)
        for item in distribuicao.iter_rows(named=True)
    }
    deficits = {
        tipo: round(valor, 4) for tipo, valor in calcular_deficit(meta, atual).items()
    }
    ultimos = ultimos_treinos_por_atleta(df_janela).get(athlete_id, [])

    pace_janela = df_janela.filter(pl.col(COL_ATHLETE) == athlete_id)[COL_PACE].mean()
    fc_mediana = linha.get(COL_HR)

    return FeaturesAtleta(
        athlete_id=athlete_id,
        genero=str(linha[COL_GENDER]).strip().upper() if linha.get(COL_GENDER) else None,
        perfil_atleta=perfil_label,
        n_corridas=int(linha[COL_N_CORRIDAS]),
        frequencia_semana=round(float(linha[COL_FREQUENCIA]), 2),
        pace_mediano_min_km=round(float(linha[COL_PACE]), 2),
        distancia_mediana_km=round(float(linha[COL_DISTANCE]) / 1000, 2),
        fc_mediana=round(float(fc_mediana), 1) if fc_mediana is not None else None,
        pace_medio_janela_min_km=(
            round(float(pace_janela), 2) if pace_janela is not None else None
        ),
        distribuicao_atual=atual,
        meta_distribuicao=meta,
        deficits=deficits,
        ultimos_treinos=ultimos,
        janela_semanas=SEMANAS_JANELA,
    )
