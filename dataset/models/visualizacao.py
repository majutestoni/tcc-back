from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from dataset.constants import COL_ATHLETE
from dataset.models.previsao_treino.model import (
    exibir_exemplos_recomendacao,
    plotar_recomendacao_atleta,
    plotar_resumo_recomendacoes,
)
from dataset.models.resultado import ResultadoModelo


def tabela_metricas(resultados: list[ResultadoModelo]) -> pl.DataFrame:
    linhas = []
    for r in resultados:
        linha = {"modelo": r.nome, "categoria": r.categoria}
        linha.update(r.metricas)
        linhas.append(linha)
    return pl.DataFrame(linhas)


def exibir_detalhes_modelo(resultado: ResultadoModelo) -> None:
    print(f"\n--- {resultado.nome} ---")
    for chave, valor in resultado.metricas.items():
        print(f"  {chave}: {valor:.4f}")

    if "coeficientes" in resultado.detalhes:
        print("  Coeficientes:", resultado.detalhes["coeficientes"])
        print("  Intercepto:", resultado.detalhes.get("intercepto"))
    if "importancias" in resultado.detalhes:
        print("  Importâncias:")
        for nome, imp in resultado.detalhes["importancias"].items():
            print(f"    {nome}: {imp:.4f}")
    if "previsao_exemplo" in resultado.detalhes:
        print("  Previsão exemplo:", resultado.detalhes["previsao_exemplo"])
    if resultado.y_true_cls is not None and resultado.class_names is not None:
        print(classification_report(
            resultado.y_true_cls,
            resultado.y_pred_cls,
            target_names=resultado.class_names,
            digits=3,
        ))


def plotar_scatter_regressao(
    resultado: ResultadoModelo,
    xlabel: str,
    ylabel: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    y_test = resultado.y_test
    y_pred = resultado.y_pred
    ax.scatter(y_test, y_pred, alpha=0.3, s=8)
    lim_min = min(y_test.min(), y_pred.min())
    lim_max = max(y_test.max(), y_pred.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", label="Ideal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    r2 = resultado.metricas.get("r2", 0)
    ax.set_title(f"{resultado.nome}\nR²={r2:.3f}")
    ax.legend(fontsize=8)
    return ax


def plotar_comparacao_regressao(
    resultados: list[ResultadoModelo],
    titulo: str,
    xlabel: str,
    ylabel: str,
) -> None:
    n = len(resultados)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, resultados):
        plotar_scatter_regressao(res, xlabel, ylabel, ax=ax)

    fig.suptitle(titulo, fontsize=14)
    plt.tight_layout()
    plt.show()

    nomes = [r.nome for r in resultados]
    maes = [r.metricas.get("mae", 0) for r in resultados]
    r2s = [r.metricas.get("r2", 0) for r in resultados]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(nomes, maes, color="steelblue")
    ax1.set_title("MAE")
    ax1.tick_params(axis="x", rotation=15)
    ax2.bar(nomes, r2s, color="coral")
    ax2.set_title("R²")
    ax2.tick_params(axis="x", rotation=15)
    fig.suptitle(f"{titulo} — Métricas comparativas")
    plt.tight_layout()
    plt.show()


def plotar_matriz_confusao(resultado: ResultadoModelo, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ConfusionMatrixDisplay(
        confusion_matrix(resultado.y_true_cls, resultado.y_pred_cls),
        display_labels=resultado.class_names,
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de confusão — {resultado.nome}")
    return ax


def plotar_comparacao_treino(resultados: list[ResultadoModelo]) -> None:
    treino_results = [r for r in resultados if r.categoria == "treino"]
    if not treino_results:
        return

    nomes = [r.nome for r in treino_results]
    erros = [r.metricas.get("erro_distribuicao_medio", 0) for r in treino_results]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(nomes, erros, color="steelblue")
    ax.set_title("Erro médio de distribuição vs meta")
    ax.set_ylabel("Erro (menor = melhor)")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.show()

    rf_results = [r for r in treino_results if "f1_macro" in r.metricas]
    if rf_results:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar([r.nome for r in rf_results], [r.metricas["f1_macro"] for r in rf_results], color="coral")
        ax.set_title("F1-macro (classificação RF)")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.show()

        for res in rf_results:
            if res.y_true_cls is not None:
                plotar_matriz_confusao(res)
                plt.tight_layout()
                plt.show()


def exibir_teste_recomendacao(resultado: ResultadoModelo, n: int = 5, plotar: bool = True) -> None:
    if resultado.recomendacoes is None or resultado.df_exibir is None:
        print(f"\n[{resultado.nome}] Sem dados de recomendação.")
        return

    print(f"\n{'=' * 60}")
    print(f"TESTE DE RECOMENDAÇÃO — {resultado.nome}")
    print(f"{'=' * 60}")

    if resultado.parametros_calibrados:
        print("\nParâmetros calibrados:")
        for tipo, p in resultado.parametros_calibrados.items():
            print(f"  {tipo}: dist={p['distancia_km']} km, pace_offset={p['pace_offset_min']:+.2f}")

    erro_medio = resultado.metricas.get("erro_distribuicao_medio")
    if erro_medio is not None:
        print(f"\nErro médio de distribuição vs meta: {erro_medio:.3f}")

    print(f"\n=== Recomendação ({resultado.nome}) ===")
    print(resultado.recomendacoes.select(
        COL_ATHLETE, "tipo", "descricao",
        "distancia_km", "pace_alvo_min_km", "tempo_estimado_min", "erro_distribuicao",
    ))

    col_tipo = resultado.col_tipo or "tipo_treino"
    exibir_exemplos_recomendacao(
        resultado.df_exibir, resultado.recomendacoes, n=n, col_tipo=col_tipo,
    )

    if not plotar:
        return

    if resultado.recomendacoes.height > 0:
        plotar_recomendacao_atleta(
            resultado.df_exibir,
            resultado.recomendacoes,
            athlete_id=resultado.recomendacoes[COL_ATHLETE][0],
            col_tipo=col_tipo,
        )
    plotar_resumo_recomendacoes(resultado.recomendacoes)


def plotar_comparacao_recomendacoes(resultados: list[ResultadoModelo]) -> None:
    treino_results = [r for r in resultados if r.categoria == "treino" and r.recomendacoes is not None]
    if len(treino_results) < 2:
        return

    fig, axes = plt.subplots(1, len(treino_results), figsize=(6 * len(treino_results), 4))
    if len(treino_results) == 1:
        axes = [axes]

    cores = {"leve": "#4CAF50", "longao": "#2196F3", "ritmo": "#FF9800", "intervalado": "#F44336"}

    for ax, res in zip(axes, treino_results):
        contagem = (
            res.recomendacoes.group_by("tipo")
            .agg(pl.len().alias("qtd"))
            .sort("qtd", descending=True)
        )
        tipos = contagem["tipo"].to_list()
        qtds = contagem["qtd"].to_list()
        ax.bar(tipos, qtds, color=[cores.get(t, "gray") for t in tipos])
        ax.set_title(res.nome)
        ax.set_xlabel("Tipo recomendado")
        ax.set_ylabel("Nº de atletas")

    fig.suptitle("Comparação — tipos recomendados")
    plt.tight_layout()
    plt.show()


def resumo_melhores(resultados: list[ResultadoModelo]) -> None:
    print("\n" + "=" * 60)
    print("RESUMO — Melhor modelo por categoria")
    print("=" * 60)

    pace = [r for r in resultados if r.categoria == "pace"]
    if pace:
        melhor = min(pace, key=lambda r: r.metricas.get("mae", float("inf")))
        print(f"  Pace (menor MAE): {melhor.nome} — MAE={melhor.metricas['mae']:.4f}, R²={melhor.metricas['r2']:.4f}")

    tempo = [r for r in resultados if r.categoria == "tempo"]
    if tempo:
        melhor = min(tempo, key=lambda r: r.metricas.get("mae", float("inf")))
        print(f"  Tempo (menor MAE): {melhor.nome} — MAE={melhor.metricas['mae']:.4f}, R²={melhor.metricas['r2']:.4f}")

    treino = [r for r in resultados if r.categoria == "treino"]
    if treino:
        melhor = min(treino, key=lambda r: r.metricas.get("erro_distribuicao_medio", float("inf")))
        print(
            f"  Treino (menor erro distribuição): {melhor.nome} — "
            f"erro={melhor.metricas['erro_distribuicao_medio']:.4f}"
        )
