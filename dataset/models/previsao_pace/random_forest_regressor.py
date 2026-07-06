import matplotlib.pyplot as plt
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from dataset.constants import *
from dataset.models.resultado import ResultadoModelo
from dataset.models.visualizacao import plotar_scatter_regressao


def _plotar_rf_regressor(resultado: ResultadoModelo) -> None:
    plotar_scatter_regressao(
        resultado,
        xlabel="Pace real (min/km)",
        ylabel="Pace previsto (min/km)",
    )
    plt.tight_layout()
    plt.show()


def treina_random_forest_regressor(
    df: pl.DataFrame,
    plotar: bool = True,
    verbose: bool = True,
) -> ResultadoModelo:
    features = [COL_DISTANCE, COL_ELEVATION, COL_HR]

    X = df.select(features).to_numpy()
    y = df.select(COL_PACE).to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    modelo = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=None)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    importancias = dict(zip(features, modelo.feature_importances_))
    novo_treino = [[5000.0, 50.0, 150.0]]
    pace_previsto = float(modelo.predict(novo_treino)[0])

    resultado = ResultadoModelo(
        nome="Random Forest Regressor (pace)",
        categoria="pace",
        metricas={"mae": mae, "r2": r2},
        detalhes={
            "importancias": importancias,
            "previsao_exemplo": {"entrada": novo_treino[0], "pace_min_km": pace_previsto},
        },
        y_test=y_test,
        y_pred=y_pred,
    )

    if verbose:
        print("MAE:", mae)
        print("R²:", r2)
        for coluna, imp in importancias.items():
            print(f"{coluna}: {imp:.4f}")
        print("Pace previsto (exemplo):", pace_previsto)

    if plotar:
        _plotar_rf_regressor(resultado)

    return resultado
