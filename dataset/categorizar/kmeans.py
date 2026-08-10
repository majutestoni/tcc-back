import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from dataset.categorizar.definir_perfil_atleta import montar_perfil_atletas
from dataset.constants import *
from dataset.models.resultado import ResultadoModelo


def _plotar_kmeans(resultado: ResultadoModelo) -> None:
    perfil = resultado.df_exibir
    if perfil is None or COL_CLUSTER_CORREDOR not in perfil.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    clusters = perfil[COL_CLUSTER_CORREDOR].to_numpy()
    dist_km = (perfil[COL_DISTANCE] / 1000).to_numpy()
    pace = perfil[COL_PACE].to_numpy()
    freq = perfil[COL_FREQUENCIA].to_numpy()

    scatter0 = axes[0].scatter(dist_km, pace, c=clusters, cmap="viridis", alpha=0.85, s=40)
    axes[0].set_xlabel("Distância mediana (km)")
    axes[0].set_ylabel("Pace mediano (min/km)")
    axes[0].set_title("Clusters: distância × pace")
    axes[0].grid(alpha=0.15)
    fig.colorbar(scatter0, ax=axes[0], label="Cluster")

    scatter1 = axes[1].scatter(freq, pace, c=clusters, cmap="viridis", alpha=0.85, s=40)
    axes[1].set_xlabel("Frequência (corridas/semana)")
    axes[1].set_ylabel("Pace mediano (min/km)")
    axes[1].set_title("Clusters: frequência × pace")
    axes[1].grid(alpha=0.15)
    fig.colorbar(scatter1, ax=axes[1], label="Cluster")

    fig.suptitle(resultado.nome, fontsize=13)
    plt.tight_layout()
    plt.show()


def treina_kmeans(
    df: pl.DataFrame,
    n_clusters: int = 3,
    plotar: bool = True,
    verbose: bool = True,
) -> ResultadoModelo:
    """
    Categoriza corredores com KMeans.

    Features por atleta (derivadas de COL_DISTANCE, COL_TIMESTAMP e COL_PACE):
    - distância mediana
    - pace mediano
    - frequência semanal (a partir do intervalo de timestamps)
    """
    features = [COL_DISTANCE, COL_PACE, COL_FREQUENCIA]

    perfil = montar_perfil_atletas(df)
    X = perfil.select(features).to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = modelo.fit_predict(X_scaled)

    inercia = float(modelo.inertia_)
    silhueta = float(silhouette_score(X_scaled, labels)) if n_clusters > 1 else 0.0

    centros = scaler.inverse_transform(modelo.cluster_centers_)
    centros_dict = {
        f"cluster_{i}": {
            COL_DISTANCE: float(centros[i, 0]),
            COL_PACE: float(centros[i, 1]),
            COL_FREQUENCIA: float(centros[i, 2]),
        }
        for i in range(n_clusters)
    }

    perfil = perfil.with_columns(pl.Series(COL_CLUSTER_CORREDOR, labels))
    contagem = (
        perfil.group_by(COL_CLUSTER_CORREDOR)
        .agg(pl.len().alias("atletas"))
        .sort(COL_CLUSTER_CORREDOR)
    )

    resultado = ResultadoModelo(
        nome=f"KMeans (corredores, k={n_clusters})",
        categoria="corredor",
        metricas={"inercia": inercia, "silhouette": silhueta},
        detalhes={
            "n_clusters": n_clusters,
            "features": features,
            "centros": centros_dict,
            "contagem_clusters": contagem.to_dicts(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        df_exibir=perfil,
        y_pred=np.asarray(labels),
        class_names=[f"cluster_{i}" for i in range(n_clusters)],
    )

    if verbose:
        print("Inércia:", round(inercia, 4))
        print("Silhouette:", round(silhueta, 4))
        print("Atletas clusterizados:", perfil.height)
        print("Contagem por cluster:")
        print(contagem)
        print("Centros (escala original):")
        for nome, centro in centros_dict.items():
            dist_km = centro[COL_DISTANCE] / 1000
            print(
                f"  {nome}: dist={dist_km:.2f} km | "
                f"pace={centro[COL_PACE]:.2f} min/km | "
                f"freq={centro[COL_FREQUENCIA]:.2f} corridas/semana"
            )

    if plotar:
        _plotar_kmeans(resultado)

    return resultado
