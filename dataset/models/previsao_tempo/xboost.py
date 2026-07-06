import matplotlib.pyplot as plt
import polars as pl
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from dataset.constants import *
from dataset.models.resultado import ResultadoModelo
from dataset.models.visualizacao import plotar_scatter_regressao


def _plotar_tempo(resultado: ResultadoModelo) -> None:
    plotar_scatter_regressao(
        resultado,
        xlabel="Tempo real (s)",
        ylabel="Tempo previsto (s)",
    )
    plt.tight_layout()
    plt.show()

def treina_xboost_tempo(df: pl.DataFrame, plotar: bool = True, verbose: bool = True) -> ResultadoModelo:
    features = [COL_DISTANCE, COL_ELEVATION]

    df_modelo = df.select(features + [COL_ELAPSED]).drop_nulls()

    X = df_modelo.select(features).to_numpy()
    y = df_modelo.select(COL_ELAPSED).to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    modelo = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    importancias = dict(zip(features, modelo.feature_importances_))

    resultado = ResultadoModelo(
        nome="XBoost (tempo)",
        categoria="tempo",
        metricas={"mae": mae, "r2": r2},
        detalhes={
            "mae_minutos": mae / 60,
            "importancias": importancias,
            "hiperparametros": {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
            },
        },
        y_test=y_test,
        y_pred=y_pred,
    )

    if verbose:
        print("Erro médio em minutos:", mae / 60)
        print("R²:", r2)
        for nome, imp in importancias.items():
            print(f"  {nome}: {imp:.4f}")
    if plotar:
        _plotar_tempo(resultado)
    return resultado