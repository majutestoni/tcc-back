import polars as pl

from dataset.categorizar.kmeans import treina_kmeans_por_genero
from dataset.categorizar.nomear_clusters import nomear_categorias
from dataset.constants import COL_DISTANCE, COL_PACE
from dataset.models.previsao_treino.recomendacao_cluster import (
    atletas_exemplo_por_perfil,
    exibir_card_recomendacao,
    gerar_recomendacoes_por_cluster,
)
from dataset.preprocessamento import preprocessar

df = pl.read_csv("dataset/raw-data-kaggle.csv", separator=";")
df = preprocessar(df)

## Recomendação de treino — só por cluster (ADR-024)
resultado = treina_kmeans_por_genero(
    df,
    features=[COL_DISTANCE, COL_PACE],
    n_clusters=2,
    plotar=False,
    verbose=False,
)
perfil = nomear_categorias(resultado.df_exibir)
recs = gerar_recomendacoes_por_cluster(df, perfil)
print(f"Recomendações por cluster: {recs.height} atletas")
for aid in atletas_exemplo_por_perfil(perfil):
    exibir_card_recomendacao(df, recs, aid)
