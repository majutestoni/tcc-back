from __future__ import annotations

import hashlib

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from dataset.constants import *

NOMES_M = [
    "Lucas Silva", "Pedro Almeida", "Rafael Costa", "Bruno Martins",
    "Thiago Rocha", "Felipe Souza", "Gustavo Lima", "André Barbosa",
    "Diego Nunes", "Ricardo Freitas", "Marcelo Dias", "Caio Mendes",
    "Henrique Pinto", "Leandro Azevedo", "Vinícius Cardoso", "Mateus Lopes",
    "Paulo Henrique", "Rodrigo Campos", "Eduardo Batista", "Fábio Melo",
    "Alexandre Reis", "Igor Fonseca", "Daniel Cunha", "Murilo Prado",
    "Samuel Borges", "Otávio Xavier", "Renato Peixoto", "Vitor Antunes",
    "Wagner Siqueira", "Yuri Magalhães", "Anderson Tavares", "César Brandão",
    "Emerson Guimarães", "Fernando Assis", "Gilberto Pacheco", "Hugo Vasconcelos",
    "Ítalo Moura", "João Pedro", "Kevin Santana", "Luiz Fernando",
    "Márcio Bastos", "Nelson Queiroz", "Orlando Pires", "Patrick Neves",
    "Quésio Faro", "Ruan Figueiredo", "Sérgio Alencar", "Túlio Barreto",
    "Ubirajara Luz", "Valter Coelho", "William Dantas", "Zeca Portela",
    "Arthur Nogueira", "Breno Caldas", "Cauã Ferraz", "Davi Sales",
    "Enzo Pimentel", "Fabrício Guedes", "Gabriel Torres", "Heitor Valente",
    "Ian Cordeiro", "Jonas Farias", "Kaique Beltrão", "Luan Escobar",
    "Miguel Paiva", "Nicolas Dutra", "Otto Seabra", "Pietro Lacerda",
    "Ravi Montenegro", "Simon Aguiar", "Theo Cavalcante", "Ulisses Parente",
    "Vicente Amorim", "Wesley Quirino", "Álvaro Serpa", "Bento Maciel",
    "Ciro Albuquerque", "Dênis Fontes", "Érico Jardim", "Flávio Belém",
    "Geraldo Passos", "Hélio Crispim", "Ícaro Bonfim", "Júlio César",
]
NOMES_F = [
    "Ana Oliveira", "Juliana Santos", "Camila Ferreira", "Beatriz Ribeiro",
    "Larissa Carvalho", "Mariana Gomes", "Patricia Monteiro", "Fernanda Araújo",
    "Amanda Teixeira", "Bruna Correia", "Letícia Moreira", "Carolina Vieira",
    "Isabela Duarte", "Natália Ramos", "Sofia Castro", "Helena Borges",
    "Laura Mendes", "Manuela Pinto", "Valentina Lopes", "Alice Souza",
    "Clara Martins", "Luiza Costa", "Júlia Almeida", "Lorena Dias",
    "Melissa Rocha", "Nicole Barbosa", "Priscila Nunes", "Rafaela Freitas",
    "Sabrina Lima", "Tatiane Cardoso", "Úrsula Azevedo", "Vitória Campos",
    "Yasmin Batista", "Zoe Melo", "Adriana Reis", "Bianca Fonseca",
    "Cíntia Cunha", "Débora Prado", "Elaine Borges", "Flávia Xavier",
    "Gisele Peixoto", "Heloísa Antunes", "Ingrid Siqueira", "Joana Magalhães",
    "Karina Tavares", "Lívia Brandão", "Mônica Guimarães", "Nina Assis",
    "Olívia Pacheco", "Paula Vasconcelos", "Queila Moura", "Renata Santana",
    "Simone Bastos", "Tânia Queiroz", "Vera Pires", "Wanda Neves",
]

CORES_PERFIL = {
    "avancado": "#1B7F4E",
    "fundista": "#2E86AB",
    "intermediario": "#E09F3E",
    "iniciante": "#E84855",
}


def _atribuir_nomes_unicos(athlete_ids: list[int], genders: list[str]) -> list[str]:
    """Atribui nomes fictícios estáveis e únicos por atleta."""
    usados: set[str] = set()
    nomes: list[str] = []

    for athlete_id, gender in zip(athlete_ids, genders):
        digest = hashlib.md5(str(athlete_id).encode()).hexdigest()
        idx = int(digest[:8], 16)
        genero = str(gender).strip().upper()
        lista = NOMES_F if genero == "F" else NOMES_M

        nome = lista[idx % len(lista)]
        tentativas = 0
        while nome in usados and tentativas < len(lista):
            tentativas += 1
            nome = lista[(idx + tentativas) % len(lista)]
        if nome in usados:
            nome = f"{nome} ({athlete_id % 1000})"

        usados.add(nome)
        nomes.append(nome)

    return nomes


def montar_fichas_atletas(
    perfil_regras: pl.DataFrame,
    perfil_kmeans: pl.DataFrame,
) -> pl.DataFrame:
    """
    Une perfil por regras + cluster KMeans e calcula distância
    até o melhor e o pior pace da mesma categoria (perfil_atleta).
    """
    fichas = (
        perfil_regras.join(
            perfil_kmeans.select(COL_ATHLETE, COL_CLUSTER_CORREDOR),
            on=COL_ATHLETE,
            how="inner",
        )
        .with_columns(
            pl.col(COL_GENDER).cast(pl.Utf8).str.strip_chars().str.to_uppercase()
        )
    )

    limites = fichas.group_by(COL_PERFIL_ATLETA).agg(
        pl.col(COL_PACE).min().alias("pace_melhor_categoria"),
        pl.col(COL_PACE).max().alias("pace_pior_categoria"),
        pl.len().alias("atletas_na_categoria"),
    )

    fichas = (
        fichas.join(limites, on=COL_PERFIL_ATLETA, how="left")
        .with_columns(
            (pl.col(COL_PACE) - pl.col("pace_melhor_categoria"))
            .round(3)
            .alias("dist_pace_melhor"),
            (pl.col("pace_pior_categoria") - pl.col(COL_PACE))
            .round(3)
            .alias("dist_pace_pior"),
            (pl.col(COL_PACE).rank("ordinal").over(COL_PERFIL_ATLETA))
            .alias("ranking_na_categoria"),
        )
        .sort(COL_PERFIL_ATLETA, COL_PACE)
    )

    nomes = _atribuir_nomes_unicos(
        fichas[COL_ATHLETE].to_list(),
        fichas[COL_GENDER].to_list(),
    )
    return fichas.with_columns(pl.Series("nome_ficticio", nomes))


def _desenhar_ficha(ax: Axes, row: dict) -> None:
    perfil = row[COL_PERFIL_ATLETA]
    cor = CORES_PERFIL.get(perfil, "#555555")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    card = FancyBboxPatch(
        (0.15, 0.3),
        9.7,
        9.4,
        boxstyle="round,pad=0.15,rounding_size=0.4",
        facecolor="#FAFBFC",
        edgecolor=cor,
        linewidth=2.2,
    )
    ax.add_patch(card)

    ax.add_patch(
        FancyBboxPatch(
            (0.15, 8.55),
            9.7,
            1.15,
            boxstyle="round,pad=0.02,rounding_size=0.35",
            facecolor=cor,
            edgecolor=cor,
            linewidth=0,
        )
    )

    genero_txt = "Masculino" if row[COL_GENDER] == "M" else "Feminino"
    ax.text(
        0.45, 9.15,
        f"{row['nome_ficticio']}  ·  {genero_txt}",
        color="white", fontsize=11, fontweight="bold", va="center",
    )
    ax.text(
        9.55, 9.15,
        f"ID {row[COL_ATHLETE]}",
        color="white", fontsize=8, ha="right", va="center", alpha=0.9,
    )

    linhas = [
        f"Corridas no dataset: {int(row[COL_N_CORRIDAS])}",
        f"Pace mediano: {row[COL_PACE]:.2f} min/km",
        f"Distância mediana: {row[COL_DISTANCE] / 1000:.1f} km",
        f"Frequência: {row[COL_FREQUENCIA]:.2f} corridas/semana",
        f"KMeans: cluster {row[COL_CLUSTER_CORREDOR]}",
        f"Perfil (regras): {perfil}",
        (
            f"Eixos: pace={row[COL_NIVEL_PACE]} · "
            f"volume={row[COL_NIVEL_VOLUME]} · "
            f"dist={row[COL_NIVEL_DISTANCIA]}"
        ),
        (
            f"Ranking na categoria: "
            f"{int(row['ranking_na_categoria'])}/{int(row['atletas_na_categoria'])}"
        ),
    ]
    y = 7.85
    for texto in linhas:
        ax.text(0.5, y, texto, fontsize=8.5, va="center", color="#222")
        y -= 0.72

    # Gauge melhor ← atleta → pior (eixo de pace)
    pace = float(row[COL_PACE])
    melhor = float(row["pace_melhor_categoria"])
    pior = float(row["pace_pior_categoria"])
    span = max(pior - melhor, 1e-6)
    pos = (pace - melhor) / span

    ax.text(0.5, 1.85, "Posição no pace da categoria", fontsize=8, color="#444")
    ax.plot([0.7, 9.3], [1.15, 1.15], color="#DDDDDD", lw=8, solid_capstyle="round")
    ax.plot([0.7, 9.3], [1.15, 1.15], color=cor, lw=3, solid_capstyle="round", alpha=0.35)
    x_atleta = 0.7 + pos * 8.6
    ax.plot(x_atleta, 1.15, "o", color=cor, markersize=10, zorder=5)

    ax.text(0.7, 0.55, f"Melhor\n{melhor:.2f}", fontsize=7, ha="left", color="#1B7F4E")
    ax.text(9.3, 0.55, f"Pior\n{pior:.2f}", fontsize=7, ha="right", color="#E84855")
    ax.text(
        5.0, 0.55,
        f"+{row['dist_pace_melhor']:.2f} do melhor  |  "
        f"+{row['dist_pace_pior']:.2f} do pior",
        fontsize=7.5, ha="center", color="#333",
    )


def plotar_fichas_atletas(
    fichas: pl.DataFrame,
    n: int = 6,
    seed: int = 42,
) -> Figure:
    """Desenha fichas visuais de até n atletas (amostra estratificada por perfil)."""
    amostras = []
    for perfil in fichas[COL_PERFIL_ATLETA].unique().to_list():
        subset = fichas.filter(pl.col(COL_PERFIL_ATLETA) == perfil)
        amostras.append(subset.sample(min(1, subset.height), seed=seed))

    base = pl.concat(amostras)
    faltam = max(0, n - base.height)
    if faltam > 0:
        resto = fichas.join(
            base.select(COL_ATHLETE),
            on=COL_ATHLETE,
            how="anti",
        )
        if resto.height > 0:
            base = pl.concat([
                base,
                resto.sample(min(faltam, resto.height), seed=seed),
            ])

    amostra = base.head(n)
    cols = min(3, amostra.height)
    rows = int(np.ceil(amostra.height / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for i, row in enumerate(amostra.iter_rows(named=True)):
        _desenhar_ficha(axes_flat[i], row)

    for j in range(amostra.height, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "Fichas de atletas — KMeans × perfil por regras",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()
    return fig


def plotar_ficha_atleta(fichas: pl.DataFrame, athlete_id: int) -> Figure:
    """Desenha a ficha de um atleta específico."""
    subset = fichas.filter(pl.col(COL_ATHLETE) == athlete_id)
    if subset.height == 0:
        raise ValueError(f"Atleta {athlete_id} não encontrado nas fichas.")

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    _desenhar_ficha(ax, subset.row(0, named=True))
    fig.suptitle("Ficha do atleta", fontsize=13)
    plt.tight_layout()
    return fig


def tabela_fichas(fichas: pl.DataFrame) -> pl.DataFrame:
    """Versão tabular resumida das fichas."""
    return (
        fichas.select(
            pl.col(COL_ATHLETE).alias("id"),
            pl.col("nome_ficticio").alias("nome"),
            pl.col(COL_GENDER).alias("genero"),
            pl.col(COL_N_CORRIDAS).alias("corridas"),
            pl.col(COL_CLUSTER_CORREDOR).alias("kmeans"),
            pl.col(COL_PERFIL_ATLETA).alias("perfil"),
            pl.col(COL_PACE).round(2).alias("pace"),
            pl.col("dist_pace_melhor").alias("delta_melhor"),
            pl.col("dist_pace_pior").alias("delta_pior"),
            pl.col("ranking_na_categoria").alias("rank"),
            pl.col("atletas_na_categoria").alias("n_categoria"),
        )
        .sort("perfil", "pace")
    )
