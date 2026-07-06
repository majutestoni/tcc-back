import polars as pl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.preprocessing import LabelEncoder

from dataset.constants import *
from dataset.models.previsao_treino.model import (
    calibrar_parametros,
    _gerar_recomendacoes,
)
from dataset.models.resultado import ResultadoModelo
from dataset.models.visualizacao import exibir_teste_recomendacao

COL_TIPO_RF = "tipo_treino_rf"

FEATURES_RF = [
    "distancia_km",
    "elevacao_por_km",
    COL_HR,
    "pace_relativo",
    "fc_relativa",
]

RF_BEST_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [8, 12, 16, None],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt", 0.6],
}


def _preparar_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (pl.col(COL_DISTANCE) / 1000).alias("distancia_km"),
        (pl.col(COL_ELEVATION) / (pl.col(COL_DISTANCE) / 1000)).alias("elevacao_por_km"),
        (
            (pl.col(COL_PACE) - pl.col(COL_PACE).mean().over(COL_ATHLETE))
            / pl.col(COL_PACE).std().over(COL_ATHLETE)
        ).fill_null(0).alias("pace_relativo"),
        (
            (pl.col(COL_HR) - pl.col(COL_HR).median().over(COL_ATHLETE))
            / pl.col(COL_HR).std().over(COL_ATHLETE)
        ).fill_null(0).alias("fc_relativa"),
    ])


def _criar_modelo_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        **RF_BEST_PARAMS,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


def _treinar_classificador_rf(
    df: pl.DataFrame,
    usar_grid: bool = False,
    verbose: bool = True,
    plotar: bool = True,
):
    df_feat = _preparar_features(df)
    df_modelo = df_feat.select(FEATURES_RF + [COL_TIPO_TREINO]).drop_nulls()

    X = df_modelo.select(FEATURES_RF).to_numpy()
    y_raw = df_modelo.select(COL_TIPO_TREINO).to_numpy().ravel()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    best_params = RF_BEST_PARAMS
    if usar_grid:
        if verbose:
            print("\nBuscando melhores hiperparâmetros (GridSearchCV)...")
        grid = GridSearchCV(
            base,
            RF_PARAM_GRID,
            cv=3,
            scoring="f1_macro",
            n_jobs=1,
        )
        grid.fit(X_train, y_train)
        modelo = grid.best_estimator_
        best_params = grid.best_params_
        if verbose:
            print(f"Melhores parâmetros: {grid.best_params_}")
            print(f"F1-macro (CV): {grid.best_score_:.3f}")
    else:
        modelo = _criar_modelo_rf()
        modelo.fit(X_train, y_train)
        if verbose:
            print(f"\nHiperparâmetros fixos: {RF_BEST_PARAMS}")

    y_pred = modelo.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    importancias = dict(sorted(
        zip(FEATURES_RF, modelo.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    ))

    if verbose:
        print("\n=== Random Forest — Classificação de tipo de treino ===")
        print(classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            digits=3,
        ))
        print("\nImportância das features:")
        for nome, imp in importancias.items():
            print(f"  {nome}: {imp:.4f}")

    if plotar:
        _, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred),
            display_labels=le.classes_,
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Matriz de confusão — RF classificador")
        plt.tight_layout()
        plt.show()

    modelo.fit(X, y)
    return modelo, le, y_test, y_pred, list(le.classes_), f1, importancias, best_params


def _classificar_com_rf(df: pl.DataFrame, modelo, le: LabelEncoder) -> pl.DataFrame:
    df_feat = _preparar_features(df)
    X = df_feat.select(FEATURES_RF).to_numpy()
    preds = le.inverse_transform(modelo.predict(X))
    return df_feat.with_columns(pl.Series(COL_TIPO_RF, preds))


def treina_rf_recomendacao(
    df: pl.DataFrame,
    athlete_id: int | None = None,
    usar_grid: bool = False,
    plotar: bool = True,
    verbose: bool = True,
) -> ResultadoModelo:
    """
    1. Treina RF para classificar tipo de treino (hiperparâmetros em RF_BEST_PARAMS)
    2. Usa previsões do RF na janela de 6 semanas
    3. Recomenda próximo treino com parâmetros calibrados
    """
    modelo, le, y_test, y_pred, class_names, f1, importancias, best_params = _treinar_classificador_rf(
        df, usar_grid=usar_grid, verbose=verbose, plotar=plotar,
    )
    df_rf = _classificar_com_rf(df, modelo, le)

    parametros = calibrar_parametros(df_rf, col_tipo=COL_TIPO_RF)

    recomendacoes = _gerar_recomendacoes(
        df_rf,
        COL_TIPO_RF,
        parametros,
        athlete_id,
        metodo="random_forest",
    )

    erro_medio = recomendacoes["erro_distribuicao"].mean() if recomendacoes.height > 0 else None
    df_exibir = df_rf.with_columns(pl.col(COL_TIPO_RF).alias(COL_TIPO_TREINO))

    resultado_modelo = ResultadoModelo(
        nome="Random Forest (recomendação)",
        categoria="treino",
        metricas={
            "f1_macro": float(f1),
            "erro_distribuicao_medio": float(erro_medio) if erro_medio is not None else 0.0,
        },
        detalhes={"importancias": importancias, "hiperparametros": best_params},
        recomendacoes=recomendacoes,
        df_exibir=df_exibir,
        col_tipo=COL_TIPO_RF,
        parametros_calibrados=parametros,
        y_true_cls=y_test,
        y_pred_cls=y_pred,
        class_names=class_names,
    )

    if verbose or plotar:
        exibir_teste_recomendacao(resultado_modelo, n=5, plotar=plotar)

    return resultado_modelo
