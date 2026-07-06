import matplotlib.pyplot as plt
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from dataset.constants import *
from dataset.models.resultado import ResultadoModelo
from dataset.models.visualizacao import plotar_scatter_regressao


def _plotar_rf(resultado: ResultadoModelo) -> None:
    plotar_scatter_regressao(
        resultado,
        xlabel="Pace real (min/km)",
        ylabel="Pace previsto (min/km)",
    )
    plt.tight_layout()
    plt.show()


def treina_random_forest(
    df: pl.DataFrame,
    plotar: bool = True,
    verbose: bool = True,
) -> ResultadoModelo:
    features = [COL_DISTANCE, COL_ELEVATION, COL_HR]
    target = COL_PACE

    df_modelo = df.select(features + [target]).drop_nulls()
    X = df_modelo.select(features).to_numpy()
    y = df_modelo.select(target).to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    importancias = dict(zip(features, modelo.feature_importances_))

    resultado = ResultadoModelo(
        nome="Random Forest (pace)",
        categoria="pace",
        metricas={"mae": mae, "mse": mse, "r2": r2},
        detalhes={"importancias": importancias},
        y_test=y_test,
        y_pred=y_pred,
    )

    if verbose:
        print("MAE:", mae)
        print("MSE:", mse)
        print("R²:", r2)
        for nome, imp in importancias.items():
            print(f"{nome}: {imp:.4f}")

    if plotar:
        _plotar_rf(resultado)

    return resultado
