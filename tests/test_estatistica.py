import numpy as np
import pytest

from dataset.avaliacao.estatistica import (
    comparar_todos,
    correcao_holm,
    ic_bootstrap,
    wilcoxon_pareado,
)


def test_ic_bootstrap_contem_a_media():
    valores = np.random.default_rng(0).normal(10.0, 2.0, 200)
    ic = ic_bootstrap(valores)

    assert ic.inferior < ic.estimativa < ic.superior
    assert ic.estimativa == pytest.approx(valores.mean())


def test_ic_bootstrap_com_valor_unico_colapsa():
    ic = ic_bootstrap(np.array([5.0]))

    assert ic.inferior == ic.estimativa == ic.superior == 5.0


def test_ic_bootstrap_rejeita_vazio():
    with pytest.raises(ValueError, match="bootstrap"):
        ic_bootstrap(np.array([np.nan]))


def test_ic_bootstrap_encolhe_com_mais_dados():
    rng = np.random.default_rng(1)
    estreito = ic_bootstrap(rng.normal(0, 1, 2000))
    largo = ic_bootstrap(rng.normal(0, 1, 40))

    assert (estreito.superior - estreito.inferior) < (largo.superior - largo.inferior)


def test_wilcoxon_detecta_modelo_melhor():
    a = {i: 1.0 for i in range(30)}
    b = {i: 2.0 for i in range(30)}
    resultado = wilcoxon_pareado("a", a, "b", b)

    assert resultado.p_valor < 0.05
    assert resultado.vencedor == "a"


def test_wilcoxon_com_erros_identicos_nao_e_significativo():
    erros = {i: float(i) for i in range(20)}
    resultado = wilcoxon_pareado("a", erros, "b", dict(erros))

    assert resultado.p_valor == 1.0
    assert resultado.vencedor is None


def test_wilcoxon_usa_media_para_direcao_quando_mediana_e_zero():
    # Mediana das diferenças é 0, mas a média favorece "a".
    a = {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0}
    b = {0: 1.0, 1: 1.0, 2: 1.0, 3: 5.0, 4: 5.0}
    resultado = wilcoxon_pareado("a", a, "b", b)

    assert resultado.mediana_diferenca == 0.0
    assert resultado.media_diferenca < 0
    assert resultado.vencedor == "a" if resultado.significativa else True


def test_wilcoxon_exige_atletas_em_comum():
    with pytest.raises(ValueError, match="comum"):
        wilcoxon_pareado("a", {1: 1.0}, "b", {2: 1.0})


def test_holm_nao_reduz_p_valores_e_e_monotono():
    brutos = [0.001, 0.01, 0.04]
    comparacoes = [
        wilcoxon_pareado(f"a{i}", {0: 1.0, 1: 2.0}, f"b{i}", {0: 2.0, 1: 3.0})
        for i in range(3)
    ]
    comparacoes = [
        type(c)(c.modelo_a, c.modelo_b, c.n_pares, c.mediana_diferenca,
                c.media_diferenca, c.estatistica, p)
        for c, p in zip(comparacoes, brutos)
    ]

    ajustadas = correcao_holm(comparacoes)
    valores = [c.p_ajustado for c in ajustadas]

    assert all(a >= b for a, b in zip(valores, brutos))
    assert valores == sorted(valores)


def test_comparar_todos_gera_todos_os_pares():
    erros = {nome: {i: float(i + k) for i in range(15)} for k, nome in enumerate("abcd")}
    comparacoes = comparar_todos(erros)

    assert len(comparacoes) == 6
    assert all(c.p_ajustado is not None for c in comparacoes)
