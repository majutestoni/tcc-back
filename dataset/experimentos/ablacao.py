"""
Ablação da regra de recomendação de ADR-007 (ADR-017, item 2).

ADR-007 tem quatro componentes que nunca foram testados isoladamente:

1. a meta de distribuição `META`;
2. os pesos `PESOS_DEFICIT` que priorizam longão e intervalado;
3. a regra anti-fadiga que troca por "leve" depois de longão/intervalado;
4. a janela `SEMANAS_JANELA = 6`.

Ablação responde "qual destes de fato muda o resultado". Componente cuja
remoção não altera a métrica é complexidade sem retorno — e, no caso da
`META`, a ablação mostra se o valor específico importa ou se qualquer meta
com a mesma ordenação produziria a mesma recomendação.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from dataset.constants import SEMANAS_JANELA
from dataset.experimentos.baselines import recomendador_deficit
from dataset.experimentos.protocolo_recomendacao import (
    TIPOS,
    avaliar_recomendador,
    montar_casos,
)
from dataset.models.previsao_treino.model import META, PESOS_DEFICIT

META_UNIFORME = dict.fromkeys(TIPOS, 1.0 / len(TIPOS))
PESOS_NEUTROS = dict.fromkeys(TIPOS, 1.0)


@dataclass(frozen=True)
class Variante:
    nome: str
    componente: str
    meta: dict[str, float]
    pesos: dict[str, float]
    usar_antifadiga: bool
    semanas_janela: int


def variantes(semanas_grade: tuple[int, ...] = (2, 4, 6, 8, 12)) -> list[Variante]:
    """Configuração completa mais uma variante por componente removido."""
    base = Variante(
        nome="completa (meta global)",
        componente="—",
        meta=META,
        pesos=PESOS_DEFICIT,
        usar_antifadiga=True,
        semanas_janela=SEMANAS_JANELA,
    )

    lista = [
        base,
        Variante(
            nome="sem META (uniforme 25%)",
            componente="META",
            meta=META_UNIFORME,
            pesos=PESOS_DEFICIT,
            usar_antifadiga=True,
            semanas_janela=SEMANAS_JANELA,
        ),
        Variante(
            nome="sem PESOS_DEFICIT",
            componente="pesos",
            meta=META,
            pesos=PESOS_NEUTROS,
            usar_antifadiga=True,
            semanas_janela=SEMANAS_JANELA,
        ),
        Variante(
            nome="sem anti-fadiga",
            componente="anti-fadiga",
            meta=META,
            pesos=PESOS_DEFICIT,
            usar_antifadiga=False,
            semanas_janela=SEMANAS_JANELA,
        ),
        Variante(
            nome="só anti-fadiga (meta uniforme, pesos neutros)",
            componente="META+pesos",
            meta=META_UNIFORME,
            pesos=PESOS_NEUTROS,
            usar_antifadiga=True,
            semanas_janela=SEMANAS_JANELA,
        ),
    ]

    lista += [
        Variante(
            nome=f"janela = {semanas} semanas",
            componente="janela",
            meta=META,
            pesos=PESOS_DEFICIT,
            usar_antifadiga=True,
            semanas_janela=semanas,
        )
        for semanas in semanas_grade
        if semanas != SEMANAS_JANELA
    ]

    return lista


def rodar_ablacao(
    df: pl.DataFrame,
    n_teste: int = 5,
    min_historico: int = 20,
    semanas_grade: tuple[int, ...] = (2, 4, 6, 8, 12),
) -> pl.DataFrame:
    """
    Avalia cada variante no mesmo protocolo de origem rolante.

    Os casos são remontados quando a janela muda, porque a janela define o
    contexto — mas o conjunto de corridas-alvo é o mesmo, então as variantes
    seguem comparáveis entre si.
    """
    cache_casos: dict[int, list] = {}
    linhas = []

    for variante in variantes(semanas_grade):
        if variante.semanas_janela not in cache_casos:
            cache_casos[variante.semanas_janela] = montar_casos(
                df,
                n_teste=n_teste,
                min_historico=min_historico,
                semanas_janela=variante.semanas_janela,
            )

        resultado = avaliar_recomendador(
            cache_casos[variante.semanas_janela],
            variante.nome,
            recomendador_deficit(
                meta=variante.meta,
                pesos=variante.pesos,
                usar_antifadiga=variante.usar_antifadiga,
            ),
        )

        linhas.append({
            "variante": variante.nome,
            "componente_removido": variante.componente,
            "acerto": round(resultado.acerto, 4),
            "f1_macro": round(resultado.f1_macro, 4),
            "casos": resultado.n_casos,
            **{f"rec_{t}": round(resultado.distribuicao_recomendada[t], 3) for t in TIPOS},
        })

    tabela = pl.DataFrame(linhas)
    referencia = tabela.filter(pl.col("componente_removido") == "—")["f1_macro"][0]

    return tabela.with_columns(
        (pl.col("f1_macro") - referencia).round(4).alias("delta_f1_vs_completa")
    )
