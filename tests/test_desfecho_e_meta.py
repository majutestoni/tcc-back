import numpy as np
import polars as pl
import pytest

from dataset.constants import (
    COL_ATHLETE,
    COL_DISTANCE,
    COL_ELAPSED,
    COL_ELEVATION,
    COL_GENDER,
    COL_HR,
    COL_TEMPO_PROVA,
    COL_TIMESTAMP,
    DISTANCIA_MEIA_KM,
)
from dataset.experimentos.aprender_meta import (
    TIPOS,
    _termos_scheffe,
    aprender_meta,
    tabela_regiao_factivel,
)
from dataset.experimentos.desfecho import (
    distancia_a_meta,
    extrair_melhor_esforco,
    montar_tabela_desfecho,
)
from dataset.preprocessamento import preprocessar


def _historico(n_atletas: int = 40, treinos: int = 40) -> pl.DataFrame:
    """Histórico sintético: cada atleta treina e depois faz uma meia maratona."""
    rng = np.random.default_rng(7)
    linhas = []

    for atleta in range(n_atletas):
        habilidade = 4.5 + (atleta % 10) * 0.2
        for i in range(treinos):
            dia = i + 1
            distancia = float(rng.uniform(5000, 16000))
            linhas.append({
                COL_ATHLETE: atleta,
                COL_GENDER: "M" if atleta % 2 else "F",
                COL_TIMESTAMP: f"{(dia % 28) + 1:02d}/{(dia // 28) + 1:02d}/2024 07:00",
                COL_DISTANCE: distancia,
                COL_ELAPSED: distancia / 1000 * habilidade * 60,
                COL_ELEVATION: 20.0,
                COL_HR: None,
            })
        # a prova, no fim do período
        linhas.append({
            COL_ATHLETE: atleta,
            COL_GENDER: "M" if atleta % 2 else "F",
            COL_TIMESTAMP: "15/03/2024 08:00",
            COL_DISTANCE: DISTANCIA_MEIA_KM * 1000,
            COL_ELAPSED: DISTANCIA_MEIA_KM * habilidade * 60,
            COL_ELEVATION: 20.0,
            COL_HR: None,
        })

    return pl.DataFrame(linhas)


def test_extrair_melhor_esforco_encontra_a_meia():
    df = preprocessar(_historico(n_atletas=5))
    desfecho = extrair_melhor_esforco(df)

    assert desfecho.height == 5
    assert COL_TEMPO_PROVA in desfecho.columns
    assert desfecho[COL_TEMPO_PROVA].min() > 60


def test_extrair_melhor_esforco_escolhe_o_mais_rapido():
    df = preprocessar(_historico(n_atletas=1))
    lento = df.filter(pl.col(COL_ATHLETE) == 0).head(1).with_columns([
        pl.lit(DISTANCIA_MEIA_KM * 1000).alias(COL_DISTANCE),
        pl.lit(DISTANCIA_MEIA_KM * 9.0 * 60).alias(COL_ELAPSED),
    ])
    desfecho = extrair_melhor_esforco(pl.concat([df, lento]))

    assert desfecho[COL_TEMPO_PROVA][0] < DISTANCIA_MEIA_KM * 9.0 * 60 / 60


def test_extrair_melhor_esforco_sem_candidatos_devolve_vazio():
    df = preprocessar(_historico(n_atletas=3)).filter(
        pl.col(COL_DISTANCE) < 17000
    )

    assert extrair_melhor_esforco(df).height == 0


def test_montar_tabela_desfecho_traz_features_de_preparo():
    tabela = montar_tabela_desfecho(preprocessar(_historico()), min_treinos=10)

    assert tabela.height > 0
    for coluna in ("volume_semanal_km", "n_treinos", "distancia_meta"):
        assert coluna in tabela.columns


def test_prova_nao_entra_na_janela_de_preparo():
    tabela = montar_tabela_desfecho(preprocessar(_historico()), min_treinos=10)

    # O longão máximo do preparo não pode ser a própria meia maratona.
    assert tabela["longao_max_km"].max() < DISTANCIA_MEIA_KM


def test_distancia_a_meta_e_zero_para_distribuicao_identica():
    meta = dict.fromkeys(TIPOS, 0.25)
    tabela = pl.DataFrame({f"prop_{t}": [0.25] for t in TIPOS})

    assert distancia_a_meta(tabela, meta=meta)["distancia_meta"][0] == pytest.approx(0.0)


def test_distancia_a_meta_e_um_para_distribuicao_disjunta():
    meta = {"leve": 1.0, "longao": 0.0, "ritmo": 0.0, "intervalado": 0.0}
    tabela = pl.DataFrame({
        "prop_leve": [0.0], "prop_longao": [1.0],
        "prop_ritmo": [0.0], "prop_intervalado": [0.0],
    })

    assert distancia_a_meta(tabela, meta=meta)["distancia_meta"][0] == pytest.approx(1.0)


def test_termos_scheffe_tem_lineares_mais_cruzados():
    termos = _termos_scheffe(np.array([0.4, 0.3, 0.2, 0.1]))

    assert termos.shape == (1, 4 + 6)
    assert termos[0, 4] == pytest.approx(0.4 * 0.3)


def test_aprender_meta_devolve_distribuicao_valida():
    tabela = montar_tabela_desfecho(preprocessar(_historico(n_atletas=60)), min_treinos=10)
    meta = aprender_meta(tabela, restrita=True, n_bootstrap=0)

    assert sum(meta.meta.values()) == pytest.approx(1.0)
    assert all(0.0 <= v <= 1.0 for v in meta.meta.values())


def test_aprender_meta_restrita_respeita_os_limites():
    tabela = montar_tabela_desfecho(preprocessar(_historico(n_atletas=60)), min_treinos=10)
    meta = aprender_meta(tabela, restrita=True, n_bootstrap=0)

    for tipo, valor in meta.meta.items():
        inferior, superior = meta.limites[tipo]
        assert inferior - 1e-6 <= valor <= superior + 1e-6


def test_tabela_regiao_factivel_marca_meta_inalcancavel():
    tabela = montar_tabela_desfecho(preprocessar(_historico(n_atletas=60)), min_treinos=10)
    regiao = tabela_regiao_factivel(tabela)

    assert set(regiao["tipo"].to_list()) == set(TIPOS)
    assert regiao["dentro_da_regiao"].dtype == pl.Boolean
