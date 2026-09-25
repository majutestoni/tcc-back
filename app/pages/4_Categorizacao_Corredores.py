import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import streamlit as st

from app.shared import (
    COMBINACOES_CLUSTER,
    clusters_por_genero,
    comparacao_features_k,
    fichas_atletas,
    grid_k_por_genero,
    silhouette_amostras,
    tabela_clusters,
    validacao_clusters,
)
from dataset.categorizar.tabela_fichas import tabela_fichas
from dataset.categorizar.visualizacao_fichas import (
    plotar_ficha_atleta,
    plotar_fichas_atletas,
)
from dataset.constants import COL_ATHLETE, COL_DISTANCE, COL_GENDER, COL_PACE
from dataset.preprocessamento import formatar_pace

st.set_page_config(page_title="Categorização de Corredores", page_icon="🎯", layout="wide")

CORES_CAT = {
    "C0-M": "#2E86AB",
    "C1-M": "#E84855",
    "C0-F": "#1B998B",
    "C1-F": "#F4A261",
}

st.title("🎯 Categorização de Corredores — KMeans")
st.markdown(
    """
    Clusterizamos atletas por **distância mediana + pace mediano**, separando
    **homens e mulheres** em grupos distintos, com **k=2** por gênero → **4 categorias**.
    Cada categoria recebe um nome de negócio (**Volume alto** / **Volume moderado**).
    Portado do notebook `categorizacao_corredores.ipynb`.
    """
)

perfil, detalhes = clusters_por_genero()
resumo = tabela_clusters()

st.caption(f"Corridas categorizadas: {perfil.height} atletas · {perfil['categoria'].n_unique()} categorias")

# --- Testes que comprovam a escolha (features × k) --------------------------
st.header("Escolha de features e k — testes comparativos")
st.markdown(
    """
    Antes de fixar **`dist_pace`** (distância + pace) com **k=2** por gênero,
    comparamos **4 combinações de features** × **k ∈ {2, 3, 4}** em masculino e
    feminino separadamente. Critérios: **Silhouette** (↑ melhor), **Davies-Bouldin**
    (↓ melhor) e **min_cluster** (evitar grupos minúsculos).
    """
)
grid_cmp, sugestoes_cmp = comparacao_features_k()

with st.expander("Grid completo — combinação × k × gênero", expanded=False):
    st.dataframe(
        grid_cmp.sort("genero", "combinacao", "k").to_pandas(),
        width="stretch",
        hide_index=True,
    )

st.subheader("Sugestão automática por gênero (melhor silhouette · min_cluster ≥ 5)")
st.dataframe(sugestoes_cmp.to_pandas(), width="stretch", hide_index=True)

cores_cmp = {
    "dist_pace": "#2E86AB",
    "dist_pace_freq": "#1B7F4E",
    "dist_pace_fc_freq": "#E84855",
    "dist_pace_fc_freq_ncorr": "#6A4C93",
}
fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
for ax, genero in zip(axes_cmp, ["Masculino", "Feminino"]):
    sub = grid_cmp.filter(pl.col("genero") == genero)
    for comb in COMBINACOES_CLUSTER:
        curva = sub.filter(pl.col("combinacao") == comb).sort("k")
        if curva.height == 0:
            continue
        ax.plot(
            curva["k"].to_numpy(), curva["silhouette"].to_numpy(),
            marker="o", lw=2, label=comb, color=cores_cmp.get(comb, "gray"),
        )
    ax.set_title(genero)
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax.set_xticks([2, 3, 4])
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
fig_cmp.suptitle("Silhouette por k — cada combinação de features", fontsize=12)
fig_cmp.tight_layout()
st.pyplot(fig_cmp)

st.info(
    "**Escolha final:** `dist_pace` + k=2. Melhor (ou quase) silhouette entre as "
    "combinações com grupos viáveis; incluir FC, frequência ou nº de corridas piora "
    "a separação e k≥3 fragmenta o feminino."
)

# --- Tabela descritiva ------------------------------------------------------
st.header("Tabela descritiva das categorias")
st.dataframe(
    resumo.select(
        "categoria", "rotulo_cluster", "atletas",
        "dist_km_med", "pace_med", "freq_semana_med", "descricao_cluster",
    ).to_pandas(),
    width="stretch",
    hide_index=True,
)

# --- Scatter com centróides -------------------------------------------------
st.header("Distância × pace por gênero (clusters + centróides)")
centros = detalhes["centros"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, genero in zip(axes, ["M", "F"]):
    sub = perfil.filter(pl.col(COL_GENDER) == genero)
    titulo = "Masculino" if genero == "M" else "Feminino"
    for cat in sorted(sub["categoria"].unique().to_list()):
        pts = sub.filter(pl.col("categoria") == cat)
        ax.scatter(
            (pts[COL_DISTANCE] / 1000).to_numpy(),
            pts[COL_PACE].to_numpy(),
            c=CORES_CAT.get(cat, "gray"),
            label=f"{cat} · {pts['nome_cluster'][0]} (n={pts.height})",
            alpha=0.8, s=50, edgecolors="white", linewidths=0.4,
        )
        centro = centros[cat]
        ax.scatter(
            centro[COL_DISTANCE] / 1000, centro[COL_PACE],
            marker="*", s=320, c="black", zorder=5,
        )
        ax.annotate(
            cat, (centro[COL_DISTANCE] / 1000, centro[COL_PACE]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=9, fontweight="bold",
        )
    ax.set_xlabel("Distância mediana (km)")
    ax.set_ylabel("Pace mediano (min/km)")
    ax.invert_yaxis()
    ax.set_title(titulo)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
fig.suptitle("Clusters independentes por gênero (★ = centróide)", fontsize=13)
fig.tight_layout()
st.pyplot(fig)

# --- Espaço original vs padronizado (onde o KMeans decide) ------------------
st.header("Original vs espaço padronizado do KMeans")
st.caption(
    "O KMeans decide no espaço padronizado (z-score). As elipses tracejadas "
    "mostram a dispersão (covariância) de cada cluster nesse espaço."
)
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler

features = [COL_DISTANCE, COL_PACE]
fig_esc, axes_esc = plt.subplots(2, 2, figsize=(13, 10))
for col_ax, genero in enumerate(["M", "F"]):
    sub = perfil.filter(pl.col(COL_GENDER) == genero)
    rotulo = "Masculino" if genero == "M" else "Feminino"
    X = sub.select(features).to_numpy()
    Xs = StandardScaler().fit_transform(X)
    ax_orig, ax_scaled = axes_esc[0, col_ax], axes_esc[1, col_ax]
    for cat in sorted(sub["categoria"].unique().to_list()):
        mask = (sub["categoria"] == cat).to_numpy()
        cor = CORES_CAT.get(cat, "gray")
        nome = sub.filter(pl.col("categoria") == cat)["nome_cluster"][0]
        ax_orig.scatter(X[mask, 0] / 1000, X[mask, 1], c=cor, alpha=0.8, s=45, label=f"{cat} · {nome}")
        ax_scaled.scatter(Xs[mask, 0], Xs[mask, 1], c=cor, alpha=0.8, s=45, label=f"{cat} · {nome}")
        if mask.sum() >= 3:
            cov = np.cov(Xs[mask].T)
            if np.all(np.isfinite(cov)):
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                w, h = 2 * np.sqrt(np.maximum(vals, 1e-9))
                m = Xs[mask].mean(axis=0)
                ax_scaled.add_patch(
                    Ellipse(m, w, h, angle=angle, fill=False, edgecolor=cor, lw=2, ls="--")
                )
    for ax, titulo, xlab, ylab, inv in [
        (ax_orig, f"{rotulo} — original", "Distância (km)", "Pace (min/km)", True),
        (ax_scaled, f"{rotulo} — KMeans (padronizado)", "Dist z-score", "Pace z-score", False),
    ]:
        ax.set_title(titulo)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
        if inv:
            ax.invert_yaxis()
fig_esc.suptitle("Original vs espaço escalado do KMeans", fontsize=12)
fig_esc.tight_layout()
st.pyplot(fig_esc)

# --- Boxplots por categoria -------------------------------------------------
st.header("Distribuição de distância e pace por categoria")
cats_box = ["C0-M", "C1-M", "C0-F", "C1-F"]
dados_box = [
    {
        "cat": cat,
        "dist": (perfil.filter(pl.col("categoria") == cat)[COL_DISTANCE] / 1000).to_numpy(),
        "pace": perfil.filter(pl.col("categoria") == cat)[COL_PACE].to_numpy(),
    }
    for cat in cats_box
    if perfil.filter(pl.col("categoria") == cat).height
]
fig_box, axes_box = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, key, titulo in zip(axes_box, ["dist", "pace"], ["Distância (km)", "Pace (min/km)"]):
    bp = ax.boxplot(
        [d[key] for d in dados_box],
        tick_labels=[d["cat"] for d in dados_box],
        patch_artist=True,
    )
    for patch, d in zip(bp["boxes"], dados_box):
        patch.set_facecolor(CORES_CAT.get(d["cat"], "gray"))
        patch.set_alpha(0.7)
    if key == "pace":
        ax.invert_yaxis()
    ax.set_title(titulo)
    ax.grid(alpha=0.2, axis="y")
fig_box.suptitle("Distribuição por categoria", fontsize=12)
fig_box.tight_layout()
st.pyplot(fig_box)

# --- Nomes e descrições -----------------------------------------------------
st.header("Categorias e descrições")
for row in resumo.sort("categoria").iter_rows(named=True):
    st.markdown(
        f"**{row['categoria']} — {row['rotulo_cluster']}** ({row['atletas']} atletas)  \n"
        f"Distância mediana: {row['dist_km_med']} km · Pace: {row['pace_med']} · "
        f"Freq.: {row['freq_semana_med']}/sem  \n"
        f"_{row['descricao_cluster']}_"
    )

# --- Validação de clusters (interna + estabilidade) -------------------------
st.header("Validação de clusters — interna e estabilidade")
det_val = validacao_clusters()

st.subheader("Validação interna por gênero")
st.caption("Silhouette ↑ melhor · Davies-Bouldin ↓ melhor · Calinski-Harabasz ↑ melhor")
linhas_int = []
for grupo, m in det_val["validacao_interna"].items():
    linhas_int.append({
        "grupo": grupo,
        "atletas": m["n_atletas"],
        "k": m["k"],
        "silhouette": round(m["silhouette"], 4),
        "davies_bouldin": round(m["davies_bouldin"], 4) if m.get("davies_bouldin") is not None else None,
        "calinski_harabasz": round(m["calinski_harabasz"], 1) if m.get("calinski_harabasz") is not None else None,
        "min_cluster": m["min_cluster"],
        "max_cluster": m["max_cluster"],
    })
st.dataframe(pl.DataFrame(linhas_int).to_pandas(), width="stretch", hide_index=True)

st.subheader("Estabilidade — ARI entre execuções consecutivas")
st.caption("ARI próximo de 1 = agrupamento robusto a diferentes seeds.")
linhas_estab = []
for grupo, m in det_val["validacao_estabilidade"].items():
    linhas_estab.append({
        "grupo": grupo,
        "ari_medio": round(m["ari_medio"], 4),
        "ari_std": round(m["ari_std"], 4),
        "n_runs": int(m["n_runs"]),
    })
st.dataframe(pl.DataFrame(linhas_estab).to_pandas(), width="stretch", hide_index=True)

st.subheader("Grid de k (k=2..7) — justificativa do k final")
grids_k = grid_k_por_genero()
fig_k, axes_k = plt.subplots(2, 3, figsize=(14, 7))
for linha_ax, genero in zip(axes_k, ["M", "F"]):
    grid = grids_k[genero]
    if grid.height == 0:
        continue
    cor = "#2E86AB" if genero == "M" else "#E84855"
    rotulo = "Masculino" if genero == "M" else "Feminino"
    for ax, col, titulo in zip(
        linha_ax,
        ["silhouette", "davies_bouldin", "calinski_harabasz"],
        ["Silhouette (↑)", "Davies-Bouldin (↓)", "Calinski-Harabasz (↑)"],
    ):
        ax.plot(grid["k"].to_numpy(), grid[col].to_numpy(), marker="o", color=cor)
        ax.axvline(2, ls="--", color="gray", alpha=0.7, label="k=2")
        ax.set_title(f"{rotulo} — {titulo}")
        ax.set_xlabel("k")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
fig_k.suptitle("Métricas internas por k — distância + pace", fontsize=13)
fig_k.tight_layout()
st.pyplot(fig_k)

st.subheader("Silhouette por atleta")
st.caption("Barras vermelhas (silhouette < 0) indicam atletas em possível fronteira entre clusters.")
sil_df = silhouette_amostras()
cats_ord = ["C0-M", "C1-M", "C0-F", "C1-F"]
fig_sil, ax_sil = plt.subplots(figsize=(14, 4))
pos = 0
xticks, xlabels = [], []
for cat in cats_ord:
    sub = sil_df.filter(pl.col("categoria") == cat).sort("silhouette_amostra")
    if sub.height == 0:
        continue
    n = sub.height
    xs = np.arange(pos, pos + n)
    vals = sub["silhouette_amostra"].to_numpy()
    ax_sil.bar(
        xs, vals,
        color=["#C0392B" if v < 0 else CORES_CAT.get(cat, "gray") for v in vals],
        width=0.9, alpha=0.85,
    )
    ax_sil.axvline(pos + n - 0.5, color="gray", ls=":", lw=0.8)
    xticks.append(pos + n / 2 - 0.5)
    xlabels.append(f"{cat}\n(n={n})")
    pos += n
media_sil = float(sil_df["silhouette_amostra"].mean())
ax_sil.axhline(0, color="black", lw=0.8)
ax_sil.axhline(media_sil, color="#555", ls="--", label=f"média={media_sil:.3f}")
ax_sil.set_xticks(xticks)
ax_sil.set_xticklabels(xlabels)
ax_sil.set_ylabel("Silhouette")
ax_sil.set_title("Silhouette por atleta (vermelho = possível fronteira)")
ax_sil.legend()
ax_sil.grid(alpha=0.2, axis="y")
fig_sil.tight_layout()
st.pyplot(fig_sil)
neg = sil_df.filter(pl.col("silhouette_amostra") < 0).height
st.caption(f"Atletas com silhouette negativo: {neg} / {sil_df.height}")

# --- Validação externa (vs perfil por regras) ------------------------------
st.header("Validação externa — concordância com perfis por regras")
st.markdown(
    """
    Cruzamos os clusters KMeans com os **perfis por regras** interpretáveis
    (perfil do atleta, nível de pace, volume e distância). Métricas: **ARI**,
    **NMI**, **homogeneidade**, **completude** e **V-measure** (todas ↑ melhor).
    """
)
val_ext = det_val.get("validacao_externa")
if not val_ext:
    st.warning("Validação externa indisponível.")
else:
    resumo_ext = []
    for ref, vals in val_ext["metricas"].items():
        resumo_ext.append({
            "referencia": ref,
            "ari": round(vals["ari"], 4),
            "nmi": round(vals["nmi"], 4),
            "homogeneidade": round(vals["homogeneity"], 4),
            "completude": round(vals["completeness"], 4),
            "v_measure": round(vals["v_measure"], 4),
        })
    st.dataframe(pl.DataFrame(resumo_ext).to_pandas(), width="stretch", hide_index=True)

    with st.expander("Tabelas de contingência (categoria × referência)"):
        for ref, linhas in val_ext["contingencias"].items():
            st.markdown(f"**{ref}**")
            st.dataframe(pl.DataFrame(linhas).to_pandas(), width="stretch", hide_index=True)

    if val_ext.get("metricas_por_genero"):
        st.subheader("Validação externa por gênero")
        for genero, metricas in val_ext["metricas_por_genero"].items():
            rotulo = "Masculino" if genero == "M" else "Feminino"
            st.markdown(f"**{rotulo}**")
            linhas_g = [
                {"referencia": ref, **{k: round(v, 4) for k, v in vals.items()}}
                for ref, vals in metricas.items()
            ]
            st.dataframe(pl.DataFrame(linhas_g).to_pandas(), width="stretch", hide_index=True)

# --- Fichas -----------------------------------------------------------------
st.header("Fichas de atletas — posição no pace da categoria")
st.caption(
    "A barra indica a posição do atleta no pace da categoria (cluster + gênero): "
    "esquerda = melhor pace, direita = pior pace."
)
fichas = fichas_atletas()

with st.expander("Tabela completa de fichas"):
    st.dataframe(tabela_fichas(fichas).to_pandas(), width="stretch", hide_index=True)

for categoria in sorted(fichas["categoria"].unique().to_list()):
    subset = fichas.filter(pl.col("categoria") == categoria)
    rotulo = subset["rotulo_cluster"][0]
    melhor = formatar_pace(subset["pace_melhor_categoria"][0])
    pior = formatar_pace(subset["pace_pior_categoria"][0])
    st.markdown(
        f"**{categoria} — {rotulo}** · melhor: {melhor} · pior: {pior} · "
        f"{subset.height} atletas"
    )
    st.dataframe(
        tabela_fichas(subset).head(5).to_pandas(), width="stretch", hide_index=True
    )

st.subheader("Fichas visuais (amostra)")
st.pyplot(plotar_fichas_atletas(fichas, n=6, seed=42))

st.subheader("Ficha individual")
ids = fichas[COL_ATHLETE].to_list()
athlete_id = st.selectbox("Atleta", ids, index=0)
st.pyplot(plotar_ficha_atleta(fichas, athlete_id))
