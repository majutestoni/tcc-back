"""Testes da eficácia por aderência × desfecho (ADR-026)."""

import numpy as np
import polars as pl

from dataset.constants import COL_ATHLETE, COL_TEMPO_PROVA
from dataset.experimentos.desfecho import TIPOS
from dataset.experimentos.eficacia_desfecho import (
    comparar_metas_no_desfecho,
    resumo_aderencia_desfecho,
    tabela_metas_vs_regiao,
)
from dataset.models.previsao_treino.recomendacao_cluster import META_POR_PERFIL


def _tabela_sintetica(n: int = 90) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    linhas = []
    for i in range(n):
        # Mais longe da meta → tempo pior (relação positiva distancia×tempo)
        ruido = rng.normal(0, 0.05)
        prop_leve = float(np.clip(0.35 + ruido, 0.1, 0.5))
        prop_longao = float(np.clip(0.25 - ruido / 2, 0.1, 0.4))
        prop_ritmo = 0.22
        prop_int = max(0.05, 1.0 - prop_leve - prop_longao - prop_ritmo)
        dist_aprox = abs(prop_leve - 0.42) + abs(prop_longao - 0.22)
        tempo = 95 + 40 * dist_aprox + rng.normal(0, 2)
        linhas.append(
            {
                COL_ATHLETE: i,
                COL_TEMPO_PROVA: tempo,
                "prop_leve": prop_leve,
                "prop_longao": prop_longao,
                "prop_ritmo": prop_ritmo,
                "prop_intervalado": prop_int,
                "volume_semanal_km": 40.0,
                "n_treinos": 20,
                "longao_max_km": 18.0,
                "pace_treino_mediano": 5.5,
            }
        )
    return pl.DataFrame(linhas)


def test_metas_somam_um_e_leve_na_faixa_factivel():
    for chave, meta in META_POR_PERFIL.items():
        assert abs(sum(meta.values()) - 1.0) < 1e-9, chave
        assert 0.12 <= meta["leve"] <= 0.48, chave


def test_resumo_aderencia_detecta_correlacao_positiva():
    tabela = _tabela_sintetica()
    resumo = resumo_aderencia_desfecho(tabela, nome_meta="teste")
    assert resumo["n"] == tabela.height
    assert resumo["corr_distancia_tempo"] is not None
    assert resumo["corr_distancia_tempo"] > 0.05
    assert resumo["tempo_mediano_alta_aderencia_min"] is not None
    assert (
        resumo["tempo_mediano_alta_aderencia_min"]
        <= resumo["tempo_mediano_baixa_aderencia_min"]
    )


def test_comparar_metas_e_regiao():
    tabela = _tabela_sintetica()
    comp = comparar_metas_no_desfecho(tabela)
    assert comp.height >= 2
    assert "corr_distancia_tempo" in comp.columns

    regiao = tabela_metas_vs_regiao(tabela)
    assert set(regiao["tipo"].unique().to_list()) == set(TIPOS)
    assert regiao.height == len(META_POR_PERFIL) * len(TIPOS)
    assert "dentro_da_regiao" in regiao.columns


def test_recomendacao_mistura_distancia_atleta():
    from dataset.models.previsao_treino.model import recomendar_proximo_treino

    deficits = {
        "leve": 0.0,
        "longao": 0.3,
        "ritmo": 0.0,
        "intervalado": 0.0,
    }
    sem_ref = recomendar_proximo_treino(deficits, ["leve"], 5.5)
    com_ref = recomendar_proximo_treino(
        deficits,
        ["leve"],
        5.5,
        distancias_atleta_por_tipo={"longao": 10.0},
    )
    assert sem_ref["tipo"] == "longao"
    assert com_ref["tipo"] == "longao"
    assert com_ref["distancia_km"] != sem_ref["distancia_km"]


def test_antifadiga_carga_recente():
    from dataset.models.previsao_treino.model import recomendar_proximo_treino

    deficits = {
        "leve": 0.0,
        "longao": 0.3,
        "ritmo": 0.0,
        "intervalado": 0.0,
    }
    rec = recomendar_proximo_treino(
        deficits,
        ["intervalado", "longao", "leve"],
        5.5,
    )
    assert rec["tipo"] == "leve"
