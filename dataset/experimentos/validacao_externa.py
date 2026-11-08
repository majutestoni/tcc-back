"""
Validação externa com desfecho de prova (ADR-018, item 3).

Três perguntas, em ordem de importância para o TCC:

1. **O treino prediz desempenho?** Se features de treino não explicam o tempo
   de prova, todo o resto perde sentido. É o teste de sanidade do desfecho.
2. **A composição de treino acrescenta algo além de volume e pace?** Aqui está
   a premissa de ADR-007: se a *distribuição* entre leve/longão/ritmo/
   intervalado não melhora a predição depois de controlar quanto e quão rápido
   o atleta treina, então recomendar "equilibre sua distribuição" não tem
   suporte nos dados.
3. **A aderência à `META` prediz desempenho?** Teste direto da constante.

Comparação de modelos encaixados (*nested*): modelo B contém todas as features
de A mais as de interesse. Se B não supera A, as features adicionais não
carregam informação — e como B é um superconjunto, a comparação é justa.

**Natureza observacional.** Estes dados são observados, não experimentais. Uma
associação entre composição de treino e desempenho não estabelece que mudar a
composição melhoraria o desempenho: atletas mais rápidos podem treinar
diferente *porque* são mais rápidos. O que o teste estabelece é mais fraco e
ainda assim útil: se nem a associação existe, a recomendação não tem base.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from dataset.avaliacao.estatistica import ic_bootstrap, wilcoxon_pareado
from dataset.constants import ALFA, COL_ATHLETE, COL_TEMPO_PROVA, N_FOLDS, SEED
from dataset.experimentos.desfecho import (
    FEATURES_COMPOSICAO,
    FEATURES_VOLUME,
)


@dataclass(frozen=True)
class ModeloEncaixado:
    nome: str
    features: list[str]


@dataclass
class ResultadoEncaixado:
    nome: str
    features: list[str]
    r2: float
    r2_ic: tuple[float, float]
    mae: float
    n: int
    erro_por_atleta: dict[int, float]


def _cv_predicoes(
    X: np.ndarray,
    y: np.ndarray,
    construir,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> np.ndarray:
    """
    Predições out-of-fold. `KFold` simples (sem agrupar) porque aqui cada linha
    é **um atleta** — as unidades já são independentes.
    """
    y_pred = np.empty_like(y, dtype=float)
    for treino, teste in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        modelo = construir()
        modelo.fit(X[treino], y[treino])
        y_pred[teste] = modelo.predict(X[teste])
    return y_pred


def avaliar_encaixado(
    tabela: pl.DataFrame,
    modelo: ModeloEncaixado,
    alvo: str = COL_TEMPO_PROVA,
    construir=LinearRegression,
    n_folds: int = N_FOLDS,
) -> ResultadoEncaixado:
    """Avalia um modelo de desfecho com CV e IC por bootstrap de atletas."""
    from sklearn.metrics import r2_score

    limpo = tabela.select([COL_ATHLETE, alvo, *modelo.features]).drop_nulls()
    X = limpo.select(modelo.features).to_numpy()
    y = limpo[alvo].to_numpy()
    atletas = limpo[COL_ATHLETE].to_numpy()

    y_pred = _cv_predicoes(X, y, construir, n_folds=n_folds)
    erro_absoluto = np.abs(y - y_pred)

    ic = ic_bootstrap(erro_absoluto)
    # IC do R² via reamostragem dos resíduos out-of-fold por atleta.
    rng = np.random.default_rng(SEED)
    r2s = [
        r2_score(y[idx], y_pred[idx])
        for idx in (rng.integers(0, len(y), len(y)) for _ in range(2000))
    ]

    return ResultadoEncaixado(
        nome=modelo.nome,
        features=modelo.features,
        r2=float(r2_score(y, y_pred)),
        r2_ic=(
            float(np.percentile(r2s, 100 * ALFA / 2)),
            float(np.percentile(r2s, 100 * (1 - ALFA / 2))),
        ),
        mae=ic.estimativa,
        n=len(y),
        erro_por_atleta={int(a): float(e) for a, e in zip(atletas, erro_absoluto)},
    )


def modelos_encaixados() -> list[ModeloEncaixado]:
    """
    Hierarquia de modelos. Cada um contém o anterior, então a diferença isola
    a contribuição do bloco de features adicionado.
    """
    return [
        ModeloEncaixado("A: só volume/pace", list(FEATURES_VOLUME)),
        ModeloEncaixado(
            "B: A + composição de treino", [*FEATURES_VOLUME, *FEATURES_COMPOSICAO]
        ),
        ModeloEncaixado(
            "C: A + aderência à META", [*FEATURES_VOLUME, "distancia_meta"]
        ),
    ]


def comparar_encaixados(
    tabela: pl.DataFrame,
    alvo: str = COL_TEMPO_PROVA,
    usar_random_forest: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Avalia a hierarquia e testa cada extensão contra o modelo base A.

    `usar_random_forest=True` repete com RF, para verificar que uma conclusão
    negativa não é artefato de linearidade — se a composição de treino
    importasse de forma não-linear, o RF acharia.
    """
    construir = (
        (lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1
        ))
        if usar_random_forest
        else LinearRegression
    )

    resultados = [
        avaliar_encaixado(tabela, m, alvo=alvo, construir=construir)
        for m in modelos_encaixados()
    ]

    linhas = [
        {
            "modelo": r.nome,
            "n_features": len(r.features),
            "r2": round(r.r2, 4),
            "r2_ic_inf": round(r.r2_ic[0], 4),
            "r2_ic_sup": round(r.r2_ic[1], 4),
            "mae_min": round(r.mae, 3),
            "n": r.n,
        }
        for r in resultados
    ]

    base = resultados[0]
    comparacoes = [
        {
            "extensao": r.nome,
            "delta_r2_vs_A": round(r.r2 - base.r2, 4),
            "delta_mae_vs_A": round(r.mae - base.mae, 4),
            "p_holm": float(f"{(c.p_ajustado or c.p_valor):.3g}"),
            "significativa": c.significativa,
            "conclusao": (
                "acrescenta informação"
                if c.significativa and r.mae < base.mae
                else "não acrescenta"
            ),
        }
        for r in resultados[1:]
        for c in [wilcoxon_pareado(r.nome, r.erro_por_atleta, base.nome, base.erro_por_atleta)]
    ]

    return pl.DataFrame(linhas), pl.DataFrame(comparacoes)


def correlacoes_desfecho(
    tabela: pl.DataFrame,
    alvo: str = COL_TEMPO_PROVA,
) -> pl.DataFrame:
    """Correlação de Pearson de cada feature com o desfecho, com IC."""
    y = tabela[alvo].to_numpy()
    colunas = [*FEATURES_VOLUME, *FEATURES_COMPOSICAO, "distancia_meta"]

    linhas = []
    for coluna in colunas:
        if coluna not in tabela.columns:
            continue
        x = tabela[coluna].to_numpy()
        mascara = ~(np.isnan(x) | np.isnan(y))
        if mascara.sum() < 3:
            continue

        r = float(np.corrcoef(x[mascara], y[mascara])[0, 1])
        n = int(mascara.sum())
        # IC de Fisher para o coeficiente de correlação.
        z = np.arctanh(np.clip(r, -0.999999, 0.999999))
        erro = 1.96 / np.sqrt(n - 3)

        linhas.append({
            "feature": coluna,
            "correlacao": round(r, 4),
            "ic_inferior": round(float(np.tanh(z - erro)), 4),
            "ic_superior": round(float(np.tanh(z + erro)), 4),
            "n": n,
        })

    return pl.DataFrame(linhas).sort("correlacao")
