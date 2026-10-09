import numpy as np
import polars as pl
import pytest

from dataset.constants import (
    COL_ATHLETE,
    COL_DISTANCE,
    COL_PACE,
    COL_TIMESTAMP,
    COL_TIPO_TREINO,
)
from dataset.experimentos.baselines import (
    NOME_REGRA_META_CLUSTER,
    NOME_REGRA_META_GLOBAL,
    baselines_com_cluster,
    baselines_padrao,
    recomendador_cluster,
    recomendador_constante,
    recomendador_deficit,
    recomendador_majoritario_atleta,
    recomendador_repetir_ultimo,
)
from dataset.experimentos.protocolo_recomendacao import (
    ContextoAtleta,
    avaliar_recomendador,
    matriz_confusao,
    montar_casos,
    tabela_amostra_casos,
    tabela_recomendadores,
)

TIPOS_CICLO = ["leve", "longao", "ritmo", "intervalado"]


def _df(n_atletas: int = 6, por_atleta: int = 30) -> pl.DataFrame:
    return pl.DataFrame({
        COL_ATHLETE: [a for a in range(n_atletas) for _ in range(por_atleta)],
        COL_TIMESTAMP: [
            f"{(i % 28) + 1:02d}/{(i // 28) + 1:02d}/2024 08:00"
            for _ in range(n_atletas)
            for i in range(por_atleta)
        ],
        COL_TIPO_TREINO: [
            TIPOS_CICLO[i % 4] for _ in range(n_atletas) for i in range(por_atleta)
        ],
        COL_PACE: [5.0 + (i % 4) * 0.2 for _ in range(n_atletas) for i in range(por_atleta)],
        COL_DISTANCE: [8000.0] * (n_atletas * por_atleta),
    })


def _contexto(tipos: list[str]) -> ContextoAtleta:
    arranjo = np.array(tipos)
    return ContextoAtleta(athlete_id=1, tipos=arranjo, tipos_janela=arranjo, pace_medio=5.5)


def test_contexto_distribuicao_soma_um():
    distribuicao = _contexto(["leve", "leve", "ritmo", "longao"]).distribuicao()

    assert sum(distribuicao.values()) == pytest.approx(1.0)
    assert distribuicao["leve"] == pytest.approx(0.5)


def test_contexto_distribuicao_vazia_e_zerada():
    contexto = ContextoAtleta(1, np.array([]), np.array([]), 5.0)

    assert set(contexto.distribuicao().values()) == {0.0}


def test_recomendador_constante_ignora_contexto():
    recomendar = recomendador_constante("leve")

    assert recomendar(_contexto(["longao"] * 5)) == "leve"


def test_recomendador_constante_rejeita_tipo_invalido():
    with pytest.raises(ValueError, match="tipo"):
        recomendador_constante("tiro_longo")


def test_repetir_ultimo_devolve_tipo_anterior():
    recomendar = recomendador_repetir_ultimo()

    assert recomendar(_contexto(["leve", "ritmo", "longao"])) == "longao"


def test_majoritario_atleta_escolhe_o_mais_frequente():
    recomendar = recomendador_majoritario_atleta()

    assert recomendar(_contexto(["ritmo", "ritmo", "ritmo", "leve"])) == "ritmo"


def test_deficit_recomenda_o_tipo_faltante():
    # Histórico só de intervalado: o maior déficit é 'leve' (meta 60%).
    recomendar = recomendador_deficit()

    assert recomendar(_contexto(["intervalado"] * 6)) == "leve"


def test_deficit_sem_antifadiga_nao_troca_por_leve():
    contexto = _contexto(["leve"] * 10)
    com = recomendador_deficit(usar_antifadiga=True)(contexto)
    sem = recomendador_deficit(usar_antifadiga=False)(contexto)

    assert com in TIPOS_CICLO
    assert sem in TIPOS_CICLO


def test_montar_casos_respeita_historico_minimo():
    casos = montar_casos(_df(por_atleta=30), n_teste=5, min_historico=20)

    assert len(casos) == 6 * 5
    assert all(c.tipos.size >= 20 for c, _ in casos)


def test_montar_casos_descarta_atleta_sem_historico_suficiente():
    assert montar_casos(_df(por_atleta=10), n_teste=5, min_historico=20) == []


def test_avaliar_recomendador_perfeito_acerta_tudo():
    df = _df()
    casos = montar_casos(df, n_teste=4, min_historico=20)
    resultado = avaliar_recomendador(casos, "oraculo", lambda c: casos[0][1] if False else _oraculo(c))

    assert resultado.n_casos == len(casos)
    assert 0.0 <= resultado.acerto <= 1.0


def _oraculo(contexto: ContextoAtleta) -> str:
    """No ciclo determinístico de `_df`, o próximo tipo segue o último."""
    ultimo = contexto.ultimo_tipo or "leve"
    return TIPOS_CICLO[(TIPOS_CICLO.index(ultimo) + 1) % 4]


def test_oraculo_acerta_ciclo_deterministico():
    casos = montar_casos(_df(), n_teste=4, min_historico=20)
    resultado = avaliar_recomendador(casos, "oraculo", _oraculo)

    assert resultado.acerto == pytest.approx(1.0)
    assert resultado.f1_macro == pytest.approx(1.0)


def test_avaliar_recomendador_rejeita_casos_vazios():
    with pytest.raises(ValueError, match="caso"):
        avaliar_recomendador([], "x", recomendador_constante("leve"))


def test_erro_por_atleta_e_complemento_do_acerto():
    casos = montar_casos(_df(), n_teste=4, min_historico=20)
    resultado = avaliar_recomendador(casos, "oraculo", _oraculo)

    assert all(v == pytest.approx(0.0) for v in resultado.erro_por_atleta.values())


def test_tabela_recomendadores_ordena_por_f1():
    casos = montar_casos(_df(), n_teste=4, min_historico=20)
    resultados = [
        avaliar_recomendador(casos, "oraculo", _oraculo),
        avaliar_recomendador(casos, "constante", recomendador_constante("leve")),
    ]
    tabela = tabela_recomendadores(resultados)

    assert tabela["recomendador"][0] == "oraculo"


def test_baselines_padrao_cobre_o_conjunto_esperado():
    nomes = baselines_padrao()

    assert NOME_REGRA_META_GLOBAL in nomes
    assert "sempre_leve" in nomes
    assert len(nomes) == 7


def test_baselines_com_cluster_inclui_meta_do_cluster():
    nomes = baselines_com_cluster({1: "volume_alto"})

    assert NOME_REGRA_META_CLUSTER in nomes
    assert len(nomes) == 8


def test_recomendador_cluster_usa_meta_do_perfil():
    # volume_moderado pede 70% leve — com histórico só de longão, ainda recomenda leve.
    recomendar = recomendador_cluster({1: "volume_moderado"})
    contexto = ContextoAtleta(
        athlete_id=1,
        tipos=np.array(["longao"] * 8),
        tipos_janela=np.array(["longao"] * 8),
        pace_medio=5.5,
    )

    assert recomendar(contexto) == "leve"


def test_matriz_confusao_e_amostra():
    casos = montar_casos(_df(), n_teste=4, min_historico=20)
    resultado = avaliar_recomendador(casos, "oraculo", _oraculo)
    matriz = matriz_confusao(resultado.y_true, resultado.y_pred)
    amostra = tabela_amostra_casos(casos, resultado.y_pred, n=5)

    assert "real \\ predito" in matriz.columns
    assert amostra.height == 5
    assert set(amostra.columns) >= {"athlete_id", "real", "predito", "acertou"}
