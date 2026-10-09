"""
Comparação de modelos sob o protocolo corrigido (ADR-017).

Junta avaliação, intervalos de confiança e testes pareados numa tabela só, e
mede o otimismo do protocolo antigo em cada modelo.
"""

from __future__ import annotations

import polars as pl

from dataset.avaliacao.avaliador import (
    ResultadoAvaliacao,
    avaliar,
    avaliar_protocolo_antigo,
)
from dataset.avaliacao.estatistica import ComparacaoPareada, comparar_todos
from dataset.avaliacao.registro_modelos import especificacoes_aplicaveis
from dataset.constants import N_BOOTSTRAP, N_FOLDS


def avaliar_todos(
    df: pl.DataFrame,
    tem_fc: bool = True,
    tem_elevacao: bool = True,
    n_folds: int = N_FOLDS,
    incluir_protocolo_antigo: bool = True,
    verbose: bool = True,
    n_reamostras: int = N_BOOTSTRAP,
) -> list[ResultadoAvaliacao]:
    """Avalia todo modelo aplicável à fonte com CV agrupada por atleta."""
    resultados = []

    for espec in especificacoes_aplicaveis(tem_fc, tem_elevacao):
        if verbose:
            print(f"  avaliando {espec.nome}...")

        resultado = avaliar(df, espec, n_folds=n_folds, n_reamostras=n_reamostras)

        if incluir_protocolo_antigo:
            resultado.detalhes["protocolo_antigo"] = avaliar_protocolo_antigo(df, espec)

        resultados.append(resultado)

    return resultados


def tabela_comparativa(resultados: list[ResultadoAvaliacao]) -> pl.DataFrame:
    """
    Uma linha por modelo, com a métrica principal e seu IC.

    `otimismo` é a diferença entre o protocolo antigo e o corrigido: positivo
    significa que o protocolo antigo reportava desempenho melhor do que o real.
    """
    linhas = []

    for r in resultados:
        chave, ic = r.metrica_principal()
        antigo = r.detalhes.get("protocolo_antigo", {})
        valor_antigo = antigo.get(chave)

        # Em MAE/erro menor é melhor, então o otimismo tem sinal invertido.
        otimismo = None
        if valor_antigo is not None:
            bruto = ic.estimativa - valor_antigo
            otimismo = bruto if r.tarefa == "regressao" else -bruto

        linhas.append({
            "modelo": r.nome,
            "categoria": r.categoria,
            "metrica": chave,
            "valor": round(ic.estimativa, 4),
            "ic_inferior": round(ic.inferior, 4),
            "ic_superior": round(ic.superior, 4),
            "protocolo_antigo": round(valor_antigo, 4) if valor_antigo is not None else None,
            "otimismo": round(otimismo, 4) if otimismo is not None else None,
            "atletas": r.n_atletas,
            "corridas": r.n_corridas,
        })

    return pl.DataFrame(linhas).sort(["categoria", "valor"])


def comparar_por_categoria(
    resultados: list[ResultadoAvaliacao],
) -> dict[str, list[ComparacaoPareada]]:
    """
    Testes pareados dentro de cada categoria.

    Comparar modelos de categorias diferentes não faz sentido: prever pace
    (min/km) e prever tempo (s) são alvos distintos, com escalas distintas.
    """
    por_categoria: dict[str, dict[str, dict[int, float]]] = {}
    for r in resultados:
        por_categoria.setdefault(r.categoria, {})[r.nome] = r.erro_por_atleta

    return {
        categoria: comparar_todos(erros)
        for categoria, erros in por_categoria.items()
        if len(erros) > 1
    }


def tabela_comparacoes(
    comparacoes: dict[str, list[ComparacaoPareada]],
) -> pl.DataFrame:
    """Uma linha por par de modelos, com p-valor corrigido por Holm."""
    linhas = [
        {
            "categoria": categoria,
            "modelo_a": c.modelo_a,
            "modelo_b": c.modelo_b,
            "atletas": c.n_pares,
            "mediana_dif": round(c.mediana_diferenca, 4),
            "media_dif": round(c.media_diferenca, 4),
            "p_valor": float(f"{c.p_valor:.2e}"),
            "p_holm": float(f"{(c.p_ajustado if c.p_ajustado is not None else c.p_valor):.2e}"),
            "significativa": c.significativa,
            "vencedor": c.vencedor or "empate",
        }
        for categoria, lista in comparacoes.items()
        for c in lista
    ]
    return pl.DataFrame(linhas) if linhas else pl.DataFrame()
