import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from dataset.categorizar.perfil_atletas import montar_perfil_atletas
from dataset.constants import *
from dataset.models.resultado import ResultadoModelo
from dataset.preprocessamento import formatar_pace

_LABELS = {
    COL_DISTANCE: "Distância mediana (km)",
    COL_PACE: "Pace mediano (min:seg/km)",
    COL_FREQUENCIA: "Frequência (corridas/semana)",
    COL_HR: "FC mediana (bpm)",
}


def _valor_exibir(col: str, valor: float) -> float:
    if col == COL_DISTANCE:
        return valor / 1000
    return valor


def _formatar_centro(centro: dict[str, float], features: list[str]) -> str:
    partes = []
    for col in features:
        valor = centro[col]
        if col == COL_DISTANCE:
            partes.append(f"dist={valor / 1000:.2f} km")
        elif col == COL_PACE:
            partes.append(f"pace={formatar_pace(valor)}")
        elif col == COL_FREQUENCIA:
            partes.append(f"freq={valor:.2f} corridas/semana")
        elif col == COL_HR:
            partes.append(f"fc={valor:.1f} bpm")
        else:
            partes.append(f"{col}={valor:.2f}")
    return " | ".join(partes)


def _nome_combinacao(features: list[str]) -> str:
    nomes = {
        COL_DISTANCE: "distância",
        COL_PACE: "pace",
        COL_FREQUENCIA: "frequência",
        COL_HR: "FC",
    }
    return " + ".join(nomes.get(col, col) for col in features)


def _plotar_kmeans(resultado: ResultadoModelo) -> None:
    perfil = resultado.df_exibir
    features = resultado.detalhes.get("features", [])
    if perfil is None or COL_CLUSTER_CORREDOR not in perfil.columns or len(features) < 2:
        return

    clusters = perfil[COL_CLUSTER_CORREDOR].to_numpy()
    pares = [(features[0], features[1])]
    if len(features) >= 3:
        pares.append((features[2], features[1]))

    fig, axes = plt.subplots(1, len(pares), figsize=(7 * len(pares), 5))
    axes = np.atleast_1d(axes)

    for ax, (x_col, y_col) in zip(axes, pares):
        x = np.array([_valor_exibir(x_col, v) for v in perfil[x_col].to_numpy()])
        y = np.array([_valor_exibir(y_col, v) for v in perfil[y_col].to_numpy()])
        scatter = ax.scatter(x, y, c=clusters, cmap="viridis", alpha=0.85, s=40)
        ax.set_xlabel(_LABELS.get(x_col, x_col))
        ax.set_ylabel(_LABELS.get(y_col, y_col))
        ax.set_title(f"{_LABELS.get(x_col, x_col)} × {_LABELS.get(y_col, y_col)}")
        ax.grid(alpha=0.15)
        fig.colorbar(scatter, ax=ax, label="Cluster")

    fig.suptitle(resultado.nome, fontsize=13)
    plt.tight_layout()
    plt.show()


def _aplicar_kmeans(
    perfil: pl.DataFrame,
    features: list[str],
    n_clusters: int,
) -> tuple[pl.DataFrame, float, float, dict[str, dict[str, float]], StandardScaler]:
    k = min(n_clusters, perfil.height)
    if k < 1:
        raise ValueError("Perfil vazio para clusterização.")

    X = perfil.select(features).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = modelo.fit_predict(X_scaled)

    inercia = float(modelo.inertia_)
    silhueta = float(silhouette_score(X_scaled, labels)) if k > 1 else 0.0

    centros = scaler.inverse_transform(modelo.cluster_centers_)
    centros_dict = {
        f"cluster_{i}": {features[j]: float(centros[i, j]) for j in range(len(features))}
        for i in range(k)
    }

    perfil = perfil.with_columns(pl.Series(COL_CLUSTER_CORREDOR, labels))
    return perfil, inercia, silhueta, centros_dict, scaler


def treina_kmeans(
    df: pl.DataFrame,
    features: list[str] | None = None,
    n_clusters: int = 3,
    plotar: bool = True,
    verbose: bool = True,
) -> ResultadoModelo:
    """
    Categoriza corredores com KMeans a partir de features agregadas por atleta.

    Cada feature deve existir no perfil retornado por `montar_perfil_atletas`.
    """
    if features is None:
        features = [COL_DISTANCE, COL_PACE]

    perfil = montar_perfil_atletas(df)
    perfil, inercia, silhueta, centros_dict, scaler = _aplicar_kmeans(
        perfil, features, n_clusters
    )
    contagem = (
        perfil.group_by(COL_CLUSTER_CORREDOR)
        .agg(pl.len().alias("atletas"))
        .sort(COL_CLUSTER_CORREDOR)
    )

    resultado = ResultadoModelo(
        nome=f"KMeans ({_nome_combinacao(features)}, k={n_clusters})",
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
        y_pred=perfil[COL_CLUSTER_CORREDOR].to_numpy(),
        class_names=[f"cluster_{i}" for i in range(n_clusters)],
    )

    if verbose:
        print("Features:", features)
        print("Inércia:", round(inercia, 4))
        print("Silhouette:", round(silhueta, 4))
        print("Atletas clusterizados:", perfil.height)
        print("Contagem por cluster:")
        print(contagem)
        print("Centros (escala original):")
        for nome, centro in centros_dict.items():
            print(f"  {nome}: {_formatar_centro(centro, features)}")

    if plotar:
        _plotar_kmeans(resultado)

    return resultado


def treina_kmeans_por_genero(
    df: pl.DataFrame,
    features: list[str] | None = None,
    n_clusters: int = 3,
    plotar: bool = True,
    verbose: bool = True,
) -> ResultadoModelo:
    """
    Categoriza corredores com KMeans **separado por gênero**.

    Roda k clusters para masculino e k para feminino, gerando até 2×k categorias
    (ex.: k=3 → 6 clusters: C0-M … C2-M, C0-F … C2-F).
    """
    if features is None:
        features = [COL_DISTANCE, COL_PACE]

    perfil = montar_perfil_atletas(df).with_columns(
        pl.col(COL_GENDER).cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    )

    blocos: list[pl.DataFrame] = []
    centros_por_categoria: dict[str, dict[str, float]] = {}
    contagem_list: list[dict] = []
    metricas_genero: dict[str, dict[str, float]] = {}
    inercia_total = 0.0
    silhueta_ponderada = 0.0
    total_atletas = 0

    for genero in ("M", "F"):
        subset = perfil.filter(pl.col(COL_GENDER) == genero)
        if subset.height == 0:
            continue

        k = min(n_clusters, subset.height)
        rotulado, inercia, silhueta, centros, scaler = _aplicar_kmeans(
            subset, features, k
        )
        rotulado = rotulado.with_columns(
            (
                pl.lit("C")
                + pl.col(COL_CLUSTER_CORREDOR).cast(pl.Utf8)
                + pl.lit("-")
                + pl.lit(genero)
            ).alias("categoria")
        )
        blocos.append(rotulado)

        for i in range(k):
            cat = f"C{i}-{genero}"
            centros_por_categoria[cat] = centros[f"cluster_{i}"]

        for row in (
            rotulado.group_by("categoria")
            .agg(pl.len().alias("atletas"))
            .sort("categoria")
            .iter_rows(named=True)
        ):
            contagem_list.append({
                "categoria": row["categoria"],
                "genero": genero,
                "cluster": int(row["categoria"].split("-")[0][1:]),
                "atletas": row["atletas"],
            })

        metricas_genero[genero] = {
            "n_atletas": subset.height,
            "k": k,
            "inercia": inercia,
            "silhouette": silhueta,
        }
        inercia_total += inercia
        silhueta_ponderada += silhueta * subset.height
        total_atletas += subset.height

    perfil_final = pl.concat(blocos).sort("categoria", COL_PACE)
    silhueta_media = silhueta_ponderada / total_atletas if total_atletas else 0.0

    resultado = ResultadoModelo(
        nome=f"KMeans ({_nome_combinacao(features)}, k={n_clusters}/gênero)",
        categoria="corredor",
        metricas={
            "inercia": inercia_total,
            "silhouette": silhueta_media,
        },
        detalhes={
            "n_clusters_por_genero": n_clusters,
            "n_categorias": perfil_final["categoria"].n_unique(),
            "features": features,
            "centros": centros_por_categoria,
            "contagem_clusters": contagem_list,
            "metricas_por_genero": metricas_genero,
            "separado_por_genero": True,
        },
        df_exibir=perfil_final,
        y_pred=perfil_final[COL_CLUSTER_CORREDOR].to_numpy(),
        class_names=sorted(perfil_final["categoria"].unique().to_list()),
    )

    if verbose:
        print("Features:", features)
        print("Separação: masculino e feminino clusterizados separadamente")
        print(f"Categorias geradas: {perfil_final['categoria'].n_unique()} (k={n_clusters} por gênero)")
        print("Inércia total:", round(inercia_total, 4))
        print("Silhouette média ponderada:", round(silhueta_media, 4))
        print("Atletas clusterizados:", perfil_final.height)
        print("\nMétricas por gênero:")
        for genero, m in metricas_genero.items():
            rotulo = "Masculino" if genero == "M" else "Feminino"
            print(
                f"  {rotulo}: {m['n_atletas']} atletas | k={m['k']} | "
                f"silhouette={m['silhouette']:.4f} | inercia={m['inercia']:.4f}"
            )
        print("\nContagem por categoria:")
        print(pl.DataFrame(contagem_list))
        print("\nCentros por categoria:")
        for cat, centro in sorted(centros_por_categoria.items()):
            print(f"  {cat}: {_formatar_centro(centro, features)}")

    if plotar and perfil_final.height > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, genero in zip(axes, ("M", "F")):
            sub = perfil_final.filter(pl.col(COL_GENDER) == genero)
            if sub.height == 0:
                ax.axis("off")
                continue
            rotulo = "Masculino" if genero == "M" else "Feminino"
            x = (sub[COL_DISTANCE] / 1000).to_numpy()
            y = sub[COL_PACE].to_numpy()
            c = sub[COL_CLUSTER_CORREDOR].to_numpy()
            scatter = ax.scatter(x, y, c=c, cmap="viridis", alpha=0.85, s=40)
            ax.set_xlabel(_LABELS[COL_DISTANCE])
            ax.set_ylabel(_LABELS[COL_PACE])
            ax.set_title(f"{rotulo} — k={min(n_clusters, sub.height)}")
            ax.grid(alpha=0.15)
            fig.colorbar(scatter, ax=ax, label="Cluster")
        fig.suptitle(resultado.nome, fontsize=13)
        plt.tight_layout()
        plt.show()

    return resultado
