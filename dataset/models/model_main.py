import polars as pl
from dataset.preprocessamento import preprocessar
from dataset.models.previsao_pace.random_forest import treina_random_forest
from dataset.models.previsao_pace.linear_regression import treina_linear_regression
from dataset.models.previsao_pace.random_forest_regressor import treina_random_forest_regressor
from dataset.models.previsao_tempo.random_forest import treina_random_tempo
from dataset.models.previsao_treino.model import treina_modelo
from dataset.models.previsao_treino.random_forest import treina_rf_recomendacao

df = pl.read_csv("dataset/raw-data-kaggle.csv", separator=";")
df = preprocessar(df)

## Previsão de pace ##
# treina_linear_regression(df)
# treina_random_forest(df)
# treina_random_forest_regressor(df)

## tempo
# treina_random_tempo(df)

## treino
treina_modelo(df)
# treina_rf_recomendacao(df, usar_grid=False)
