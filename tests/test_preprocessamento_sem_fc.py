import polars as pl

from dataset.constants import (
    COL_ATHLETE,
    COL_DISTANCE,
    COL_ELAPSED,
    COL_ELEVATION,
    COL_GENDER,
    COL_HR,
    COL_TIMESTAMP,
    COL_TIPO_TREINO,
)
from dataset.preprocessamento import (
    classificar_tipo_treino,
    filtrar_corridas_validas,
    preprocessar,
    tem_frequencia_cardiaca,
)


def _corridas(fc: list[float | None], n: int = 12) -> pl.DataFrame:
    return pl.DataFrame({
        COL_ATHLETE: [1] * n,
        COL_GENDER: ["M"] * n,
        COL_TIMESTAMP: [f"{i + 1:02d}/01/2024 08:00" for i in range(n)],
        COL_DISTANCE: [5000.0 + i * 500 for i in range(n)],
        COL_ELAPSED: [1800.0 + i * 120 for i in range(n)],
        COL_ELEVATION: [10.0] * n,
        COL_HR: fc,
    })


def test_detecta_fc_ausente_quando_toda_nula():
    assert tem_frequencia_cardiaca(_corridas([None] * 12)) is False


def test_detecta_fc_ausente_quando_constante():
    assert tem_frequencia_cardiaca(_corridas([150.0] * 12)) is False


def test_detecta_fc_presente_quando_varia():
    assert tem_frequencia_cardiaca(_corridas([140.0 + i for i in range(12)])) is True


def test_detecta_fc_ausente_quando_coluna_nao_existe():
    df = _corridas([None] * 12).drop(COL_HR)

    assert tem_frequencia_cardiaca(df) is False


def test_pipeline_sem_fc_nao_zera_o_dataset():
    saida = preprocessar(_corridas([None] * 12))

    assert saida.height > 0
    assert COL_TIPO_TREINO in saida.columns


def test_pipeline_com_fc_ainda_filtra_valores_irreais():
    fc = [150.0 + i for i in range(11)] + [10.0]
    saida = filtrar_corridas_validas(_corridas(fc), usar_fc=True)

    assert 10.0 not in saida[COL_HR].to_list()


def test_classificacao_sem_fc_produz_tipos_variados():
    # `classificar_tipo_treino` consome COL_PACE, criado por `filtrar_corridas_validas`.
    variadas = _corridas([None] * 12).with_columns(
        pl.Series(COL_ELAPSED, [1500.0, 2400.0, 1800.0, 2100.0] * 3)
    )
    com_pace = filtrar_corridas_validas(variadas, usar_fc=False)
    saida = classificar_tipo_treino(com_pace, usar_fc=False)

    assert saida[COL_TIPO_TREINO].n_unique() >= 2
    assert saida.height == com_pace.height


def test_classificacao_sem_fc_nao_deixa_colunas_auxiliares():
    com_pace = filtrar_corridas_validas(_corridas([None] * 12), usar_fc=False)
    saida = classificar_tipo_treino(com_pace, usar_fc=False)

    assert not [c for c in saida.columns if c.startswith("_")]
