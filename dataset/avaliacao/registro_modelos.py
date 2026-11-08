"""
Especificação dos 8 modelos do projeto para avaliação rigorosa (ADR-017).

As features e hiperparâmetros aqui espelham `dataset/models/`. As funções
`treina_*` de lá não são reaproveitadas porque cada uma faz o próprio
`train_test_split` internamente — é justamente o que ADR-017 corrige. Elas
seguem servindo à demonstração e ao Streamlit (ADR-008/ADR-011).
"""

from __future__ import annotations

import polars as pl
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from dataset.avaliacao.avaliador import EspecModelo
from dataset.constants import (
    COL_ATHLETE,
    COL_DISTANCE,
    COL_ELAPSED,
    COL_ELEVATION,
    COL_HR,
    COL_PACE,
    COL_TIPO_TREINO,
    SEED,
)

FEATURES_PACE = [COL_DISTANCE, COL_ELEVATION, COL_HR]
FEATURES_TEMPO = [COL_DISTANCE, COL_ELEVATION]

AVISO_ALVO_CIRCULAR = (
    "ATENÇÃO: o alvo `tipo_treino` é função determinística de pace relativo ao "
    "atleta e do percentil de distância (ADR-004) — as mesmas quantidades que "
    "geram as features. O modelo reaprende uma fórmula, então um F1 alto aqui "
    "não é evidência de capacidade preditiva."
)


def _rf_regressor() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)


def _preparar_treino_rf(df: pl.DataFrame) -> pl.DataFrame:
    from dataset.models.previsao_treino.random_forest import _preparar_features

    return _preparar_features(df)


def _preparar_treino_sem_fc(df: pl.DataFrame) -> pl.DataFrame:
    """Subconjunto de `_preparar_features` que não depende de FC nem elevação."""
    return df.with_columns([
        (pl.col(COL_DISTANCE) / 1000).alias("distancia_km"),
        (
            (pl.col(COL_PACE) - pl.col(COL_PACE).mean().over(COL_ATHLETE))
            / pl.col(COL_PACE).std().over(COL_ATHLETE)
        ).fill_null(0).alias("pace_relativo"),
    ])


def _xgb_regressor():
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=SEED,
        n_jobs=-1,
    )


def especificacoes() -> list[EspecModelo]:
    """Os 8 modelos, na mesma divisão por categoria de ADR-008."""
    from dataset.models.previsao_treino.random_forest import (
        FEATURES_RF,
        RF_BEST_PARAMS,
    )

    return [
        EspecModelo(
            nome="Regressão Linear (pace)",
            categoria="pace",
            tarefa="regressao",
            features=FEATURES_PACE,
            alvo=COL_PACE,
            construir=LinearRegression,
            exige_fc=True,
            exige_elevacao=True,
        ),
        EspecModelo(
            nome="Random Forest (pace)",
            categoria="pace",
            tarefa="regressao",
            features=FEATURES_PACE,
            alvo=COL_PACE,
            construir=_rf_regressor,
            exige_fc=True,
            exige_elevacao=True,
        ),
        EspecModelo(
            nome="RF Regressor (pace)",
            categoria="pace",
            tarefa="regressao",
            features=FEATURES_PACE,
            alvo=COL_PACE,
            construir=lambda: RandomForestRegressor(
                n_estimators=200, max_depth=12, random_state=SEED, n_jobs=-1
            ),
            exige_fc=True,
            exige_elevacao=True,
        ),
        EspecModelo(
            nome="Regressão Linear (tempo)",
            categoria="tempo",
            tarefa="regressao",
            features=FEATURES_TEMPO,
            alvo=COL_ELAPSED,
            construir=LinearRegression,
            exige_elevacao=True,
        ),
        EspecModelo(
            nome="Random Forest (tempo)",
            categoria="tempo",
            tarefa="regressao",
            features=FEATURES_TEMPO,
            alvo=COL_ELAPSED,
            construir=_rf_regressor,
            exige_elevacao=True,
        ),
        EspecModelo(
            nome="XGBoost (tempo)",
            categoria="tempo",
            tarefa="regressao",
            features=FEATURES_TEMPO,
            alvo=COL_ELAPSED,
            construir=_xgb_regressor,
            exige_elevacao=True,
        ),
        EspecModelo(
            nome="RF Classificador (tipo de treino)",
            categoria="treino",
            tarefa="classificacao",
            features=FEATURES_RF,
            alvo=COL_TIPO_TREINO,
            construir=lambda: RandomForestClassifier(
                **RF_BEST_PARAMS,
                random_state=SEED,
                n_jobs=-1,
                class_weight="balanced",
            ),
            preparar=_preparar_treino_rf,
            exige_fc=True,
            exige_elevacao=True,
            observacao=AVISO_ALVO_CIRCULAR,
        ),
        EspecModelo(
            nome="RF Classificador sem FC (tipo de treino)",
            categoria="treino",
            tarefa="classificacao",
            features=["distancia_km", "pace_relativo"],
            alvo=COL_TIPO_TREINO,
            construir=lambda: RandomForestClassifier(
                **RF_BEST_PARAMS,
                random_state=SEED,
                n_jobs=-1,
                class_weight="balanced",
            ),
            preparar=_preparar_treino_sem_fc,
            observacao=(
                "Variante aplicável a fontes sem FC/elevação (ADR-018). "
                + AVISO_ALVO_CIRCULAR
            ),
        ),
    ]


def especificacoes_aplicaveis(tem_fc: bool, tem_elevacao: bool) -> list[EspecModelo]:
    """Filtra os modelos que a fonte suporta."""
    return [e for e in especificacoes() if e.aplicavel(tem_fc, tem_elevacao)]
