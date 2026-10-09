import numpy as np
import polars as pl
import pytest

from dataset.avaliacao.particionamento import (
    folds_por_atleta,
    split_temporal,
    split_temporal_por_atleta,
)
from dataset.constants import COL_ATHLETE, COL_TIMESTAMP


def _df(n_atletas: int = 10, por_atleta: int = 8) -> pl.DataFrame:
    return pl.DataFrame({
        COL_ATHLETE: [a for a in range(n_atletas) for _ in range(por_atleta)],
        COL_TIMESTAMP: [
            f"{(i % 28) + 1:02d}/01/2024 10:00"
            for a in range(n_atletas)
            for i in range(por_atleta)
        ],
    })


def test_folds_nao_compartilham_atleta_entre_treino_e_teste():
    df = _df()
    atletas = df[COL_ATHLETE].to_numpy()

    for treino, teste in folds_por_atleta(df, n_folds=5):
        assert not set(atletas[treino]) & set(atletas[teste])


def test_folds_cobrem_todas_as_linhas_exatamente_uma_vez():
    df = _df()
    cobertura = np.concatenate([teste for _, teste in folds_por_atleta(df, n_folds=5)])

    assert sorted(cobertura) == list(range(df.height))


def test_folds_rejeita_menos_atletas_que_folds():
    with pytest.raises(ValueError, match="folds"):
        folds_por_atleta(_df(n_atletas=3), n_folds=5)


def test_split_temporal_mantem_teste_depois_do_treino():
    df = _df(n_atletas=4, por_atleta=10)
    treino, teste = split_temporal(df, fracao_teste=0.3)

    maior_treino = treino[COL_TIMESTAMP].str.strptime(
        pl.Datetime, format="%d/%m/%Y %H:%M"
    ).max()
    menor_teste = teste[COL_TIMESTAMP].str.strptime(
        pl.Datetime, format="%d/%m/%Y %H:%M"
    ).min()

    assert maior_treino < menor_teste
    assert treino.height + teste.height == df.height


def test_split_temporal_rejeita_fracao_invalida():
    with pytest.raises(ValueError, match="fracao_teste"):
        split_temporal(_df(), fracao_teste=1.5)


def test_split_temporal_por_atleta_separa_ultimas_corridas_de_cada_um():
    df = _df(n_atletas=6, por_atleta=8)
    treino, teste = split_temporal_por_atleta(df, n_teste=2)

    assert teste.height == 6 * 2
    assert treino.height == 6 * 6
    contagem = teste.group_by(COL_ATHLETE).len()
    assert set(contagem["len"].to_list()) == {2}


def test_split_temporal_por_atleta_rejeita_n_teste_zero():
    with pytest.raises(ValueError, match="n_teste"):
        split_temporal_por_atleta(_df(), n_teste=0)
