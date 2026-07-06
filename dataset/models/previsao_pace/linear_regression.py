import matplotlib.pyplot as plt
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from dataset.constants import *
from dataset.models.resultado import ResultadoModelo
from dataset.models.visualizacao import plotar_scatter_regressao


def _plotar_lr(resultado: ResultadoModelo) -> None:
    plotar_scatter_regressao(
        resultado,
        xlabel="Pace real (min/km)",
        ylabel="Pace previsto (min/km)",
    )
    plt.tight_layout()
    plt.show()


def treina_linear_regression(
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

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    coeficientes = dict(zip(features, modelo.coef_))
    novo_treino = [[5000.0, 50.0, 150.0]]
    pace_previsto = float(modelo.predict(novo_treino)[0])

    resultado = ResultadoModelo(
        nome="Regressão Linear (pace)",
        categoria="pace",
        metricas={"mae": mae, "r2": r2},
        detalhes={
            "coeficientes": coeficientes,
            "intercepto": float(modelo.intercept_),
            "previsao_exemplo": {"entrada": novo_treino[0], "pace_min_km": pace_previsto},
        },
        y_test=y_test,
        y_pred=y_pred,
    )

    if verbose:
        print("MAE:", mae)
        print("R²:", r2)
        print("Coeficientes:", coeficientes)
        print("Intercepto:", modelo.intercept_)
        print("Pace previsto (exemplo):", pace_previsto)

    if plotar:
        _plotar_lr(resultado)

    return resultado
