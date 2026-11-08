"""
Protocolo de avaliação de recomendadores de treino (ADR-017).

**O que este protocolo mede, e o que não mede.** Ele usa origem rolante
(*rolling origin*): para cada uma das últimas `n_teste` corridas de um atleta,
o recomendador vê apenas o que veio antes dela e tenta acertar o tipo daquela
corrida. Isso mede **imitação** — quão bem o sistema reproduz o que o atleta
de fato fez em seguida. Não mede se o conselho é *bom*: um atleta que treina
mal é imitado com sucesso por um sistema que recomenda mal.

Para medir qualidade do conselho é preciso um desfecho, que é o que
`desfecho.py` e `aprender_meta.py` fazem (ADR-018/ADR-019).

Mesmo assim, imitação é informativa como piso: se a regra de ADR-007 não
supera "repita o último treino", ela não está usando informação que um
baseline trivial já não tenha.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.metrics import f1_score

from dataset.constants import (
    COL_ATHLETE,
    COL_PACE,
    COL_TIMESTAMP,
    COL_TIPO_TREINO,
    SEED,
    SEMANAS_JANELA,
)
from dataset.models.previsao_treino.model import META

FORMATO_TIMESTAMP = "%d/%m/%Y %H:%M"
TIPOS = tuple(META.keys())


@dataclass(frozen=True)
class ContextoAtleta:
    """
    Histórico de um atleta até (exclusive) o ponto a prever.

    `tipos_janela` já vem restrito às últimas `SEMANAS_JANELA` semanas, que é a
    janela de ADR-007; `tipos` é o histórico completo, usado pelos baselines
    que não têm noção de janela.
    """

    athlete_id: int
    tipos: np.ndarray
    tipos_janela: np.ndarray
    pace_medio: float

    def distribuicao(self) -> dict[str, float]:
        """Distribuição de tipos na janela de ADR-007."""
        if self.tipos_janela.size == 0:
            return dict.fromkeys(TIPOS, 0.0)
        return {
            tipo: float(np.mean(self.tipos_janela == tipo)) for tipo in TIPOS
        }

    @property
    def ultimo_tipo(self) -> str | None:
        return str(self.tipos[-1]) if self.tipos.size else None


Recomendador = Callable[[ContextoAtleta], str]


@dataclass
class ResultadoRecomendador:
    nome: str
    n_casos: int
    n_atletas: int
    acerto: float
    f1_macro: float
    distribuicao_recomendada: dict[str, float]
    acerto_por_atleta: dict[int, float]
    y_true: np.ndarray
    y_pred: np.ndarray

    @property
    def erro_por_atleta(self) -> dict[int, float]:
        """Formato esperado pelos testes pareados de `avaliacao.estatistica`."""
        return {a: 1.0 - v for a, v in self.acerto_por_atleta.items()}


def _historico_por_atleta(df: pl.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    ordenado = (
        df.with_columns(
            pl.col(COL_TIMESTAMP)
            .str.strptime(pl.Datetime, format=FORMATO_TIMESTAMP, strict=False)
            .alias("_ts")
        )
        .drop_nulls("_ts")
        .sort([COL_ATHLETE, "_ts"])
    )

    historico = {}
    for (athlete_id,), grupo in ordenado.group_by([COL_ATHLETE], maintain_order=True):
        historico[int(athlete_id)] = {
            "ts": grupo["_ts"].to_numpy(),
            "tipo": grupo[COL_TIPO_TREINO].to_numpy(),
            "pace": grupo[COL_PACE].to_numpy(),
        }
    return historico


def _contextos(
    df: pl.DataFrame,
    n_teste: int,
    min_historico: int,
    semanas_janela: int,
) -> list[tuple[ContextoAtleta, str]]:
    """
    Monta os casos de teste: (contexto até t, tipo real em t).

    Um atleta só entra se tiver `min_historico` corridas antes do primeiro
    ponto de teste — sem histórico, qualquer recomendador está adivinhando e o
    caso não discrimina entre eles.
    """
    janela_ns = np.timedelta64(semanas_janela * 7, "D")
    casos = []

    for athlete_id, h in _historico_por_atleta(df).items():
        total = h["tipo"].size
        if total < min_historico + n_teste:
            continue

        for i in range(total - n_teste, total):
            ts_corte = h["ts"][i - 1]
            na_janela = h["ts"][:i] >= (ts_corte - janela_ns)

            casos.append((
                ContextoAtleta(
                    athlete_id=athlete_id,
                    tipos=h["tipo"][:i],
                    tipos_janela=h["tipo"][:i][na_janela],
                    pace_medio=float(np.mean(h["pace"][:i])),
                ),
                str(h["tipo"][i]),
            ))

    return casos


def avaliar_recomendador(
    casos: list[tuple[ContextoAtleta, str]],
    nome: str,
    recomendador: Recomendador,
) -> ResultadoRecomendador:
    """Roda um recomendador em todos os casos e mede acerto e F1-macro."""
    if not casos:
        raise ValueError("nenhum caso de teste; reduza min_historico ou n_teste")

    y_true, y_pred, atletas = [], [], []
    for contexto, real in casos:
        y_true.append(real)
        y_pred.append(recomendador(contexto))
        atletas.append(contexto.athlete_id)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    atletas = np.array(atletas)

    acerto_por_atleta = {
        int(a): float(np.mean(y_true[atletas == a] == y_pred[atletas == a]))
        for a in np.unique(atletas)
    }

    return ResultadoRecomendador(
        nome=nome,
        n_casos=len(y_true),
        n_atletas=len(acerto_por_atleta),
        acerto=float(np.mean(y_true == y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", labels=list(TIPOS), zero_division=0)),
        distribuicao_recomendada={
            tipo: float(np.mean(y_pred == tipo)) for tipo in TIPOS
        },
        acerto_por_atleta=acerto_por_atleta,
        y_true=y_true,
        y_pred=y_pred,
    )


def montar_casos(
    df: pl.DataFrame,
    n_teste: int = 5,
    min_historico: int = 20,
    semanas_janela: int = SEMANAS_JANELA,
) -> list[tuple[ContextoAtleta, str]]:
    """Casos de teste por origem rolante, compartilhados por todos os recomendadores."""
    return _contextos(df, n_teste, min_historico, semanas_janela)


def tabela_recomendadores(
    resultados: list[ResultadoRecomendador],
    *,
    em_percentual: bool = False,
) -> pl.DataFrame:
    """
    Uma linha por recomendador.

    `f1_macro` importa mais que `acerto` aqui: um recomendador degenerado que
    sempre responde a classe mais comum obtém acerto razoável e F1-macro baixo.

    Com `em_percentual=True`, acerto/F1/`rec_*` saem em % (0–100) para a UI.
    """
    if em_percentual:
        return pl.DataFrame([
            {
                "recomendador": r.nome,
                "acerto_%": round(r.acerto * 100, 1),
                "f1_macro_%": round(r.f1_macro * 100, 1),
                "casos": r.n_casos,
                "atletas": r.n_atletas,
                **{
                    f"rec_{t}_%": round(r.distribuicao_recomendada[t] * 100, 1)
                    for t in TIPOS
                },
            }
            for r in resultados
        ]).sort("f1_macro_%", descending=True)

    return pl.DataFrame([
        {
            "recomendador": r.nome,
            "acerto": round(r.acerto, 4),
            "f1_macro": round(r.f1_macro, 4),
            "casos": r.n_casos,
            "atletas": r.n_atletas,
            **{f"rec_{t}": round(r.distribuicao_recomendada[t], 3) for t in TIPOS},
        }
        for r in resultados
    ]).sort("f1_macro", descending=True)


def matriz_confusao(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple[str, ...] = TIPOS,
) -> pl.DataFrame:
    """Matriz real × predito (linhas = real, colunas = predito)."""
    from sklearn.metrics import confusion_matrix

    matriz = confusion_matrix(y_true, y_pred, labels=list(labels))
    return pl.DataFrame(
        {
            "real \\ predito": list(labels),
            **{lab: matriz[:, i].tolist() for i, lab in enumerate(labels)},
        }
    )


def tabela_amostra_casos(
    casos: list[tuple[ContextoAtleta, str]],
    y_pred: np.ndarray,
    n: int = 15,
    seed: int = SEED,
) -> pl.DataFrame:
    """Amostra de casos para inspeção na UI (histórico curto → real vs predito)."""
    if not casos:
        return pl.DataFrame()

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(casos), size=min(n, len(casos)), replace=False)
    linhas = []
    for i in sorted(int(x) for x in indices):
        contexto, real = casos[i]
        ultimos = contexto.tipos[-5:].tolist() if contexto.tipos.size else []
        predito = str(y_pred[i])
        linhas.append(
            {
                "athlete_id": contexto.athlete_id,
                "ultimos_5": " → ".join(str(t) for t in ultimos),
                "real": real,
                "predito": predito,
                "acertou": real == predito,
            }
        )
    return pl.DataFrame(linhas)
