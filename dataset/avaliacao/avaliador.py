"""
Avaliador genérico com validação cruzada agrupada por atleta (ADR-017).

Produz **predições out-of-fold** para todas as corridas: cada corrida é
prevista por um modelo que nunca viu nenhuma corrida daquele atleta. Isso dá
duas coisas de uma vez:

1. métricas globais honestas (sem otimismo de vazamento);
2. um erro por atleta, que é a unidade de análise dos testes pareados de
   `estatistica.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from dataset.avaliacao.estatistica import (
    IntervaloConfianca,
    ic_bootstrap,
    ic_bootstrap_metrica,
)
from dataset.avaliacao.particionamento import folds_por_atleta
from dataset.constants import COL_ATHLETE, N_BOOTSTRAP, N_FOLDS

TipoTarefa = Literal["regressao", "classificacao"]


@dataclass(frozen=True)
class EspecModelo:
    """
    Descreve um modelo de forma suficiente para avaliá-lo sob qualquer
    protocolo. Separado das funções `treina_*` de `dataset/models/` de propósito:
    aquelas fazem o fit de demonstração e devolvem `ResultadoModelo` (ADR-008,
    consumido pelo Streamlit); esta descreve só o que a avaliação precisa.
    """

    nome: str
    categoria: str
    tarefa: TipoTarefa
    features: list[str]
    alvo: str
    construir: Callable[[], object]
    preparar: Callable[[pl.DataFrame], pl.DataFrame] | None = None
    exige_fc: bool = False
    exige_elevacao: bool = False
    observacao: str = ""

    def aplicavel(self, tem_fc: bool, tem_elevacao: bool) -> bool:
        return (tem_fc or not self.exige_fc) and (tem_elevacao or not self.exige_elevacao)


@dataclass
class ResultadoAvaliacao:
    nome: str
    categoria: str
    tarefa: TipoTarefa
    protocolo: str
    n_corridas: int
    n_atletas: int
    metricas: dict[str, IntervaloConfianca]
    erro_por_atleta: dict[int, float]
    observacao: str = ""
    detalhes: dict = field(default_factory=dict)

    def metrica_principal(self) -> tuple[str, IntervaloConfianca]:
        chave = "mae" if self.tarefa == "regressao" else "f1_macro"
        return chave, self.metricas[chave]


def _matriz(df: pl.DataFrame, espec: EspecModelo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preparado = espec.preparar(df) if espec.preparar else df
    colunas = espec.features + [espec.alvo, COL_ATHLETE]

    faltando = [c for c in colunas if c not in preparado.columns]
    if faltando:
        raise ValueError(f"{espec.nome}: colunas ausentes {faltando}")

    limpo = preparado.select(colunas).drop_nulls()
    if limpo.height == 0:
        raise ValueError(f"{espec.nome}: nenhuma linha após remover nulos")

    X = limpo.select(espec.features).to_numpy()
    y = limpo.select(espec.alvo).to_numpy().ravel()
    grupos = limpo[COL_ATHLETE].to_numpy()
    return X, y, grupos


def _erro_por_atleta(
    y: np.ndarray,
    y_pred: np.ndarray,
    grupos: np.ndarray,
    tarefa: TipoTarefa,
) -> dict[int, float]:
    """MAE por atleta (regressão) ou taxa de erro por atleta (classificação)."""
    erros: dict[int, float] = {}
    for atleta in np.unique(grupos):
        mascara = grupos == atleta
        if tarefa == "regressao":
            erros[int(atleta)] = float(np.mean(np.abs(y[mascara] - y_pred[mascara])))
        else:
            erros[int(atleta)] = float(np.mean(y[mascara] != y_pred[mascara]))
    return erros


def _codificar(y: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mapeia rótulos para inteiros.

    Necessário por desempenho: `f1_score` sobre arrays de string refaz
    `np.unique` em strings a cada chamada, e o bootstrap chama a métrica
    milhares de vezes. Codificar uma vez reduz a avaliação dos classificadores
    de ~200 s para poucos segundos.
    """
    classes = np.unique(np.concatenate([y, y_pred]))
    codigos = {classe: i for i, classe in enumerate(classes)}
    traduzir = np.vectorize(codigos.__getitem__, otypes=[np.int32])
    return traduzir(y), traduzir(y_pred), np.arange(len(classes))


def _metricas(
    y: np.ndarray,
    y_pred: np.ndarray,
    grupos: np.ndarray,
    tarefa: TipoTarefa,
    erro_por_atleta: dict[int, float],
    n_reamostras: int = N_BOOTSTRAP,
) -> dict[str, IntervaloConfianca]:
    por_atleta = np.array(list(erro_por_atleta.values()), dtype=float)

    if tarefa == "regressao":
        return {
            "mae": ic_bootstrap(por_atleta, n_reamostras=n_reamostras),
            "rmse": ic_bootstrap_metrica(
                y, y_pred, grupos, root_mean_squared_error, n_reamostras=n_reamostras
            ),
            "r2": ic_bootstrap_metrica(
                y, y_pred, grupos, r2_score, n_reamostras=n_reamostras
            ),
        }

    y_cod, pred_cod, rotulos = _codificar(y, y_pred)

    return {
        "f1_macro": ic_bootstrap_metrica(
            y_cod,
            pred_cod,
            grupos,
            lambda a, b: f1_score(a, b, average="macro", labels=rotulos, zero_division=0),
            n_reamostras=n_reamostras,
        ),
        "acuracia": ic_bootstrap_metrica(
            y_cod, pred_cod, grupos, accuracy_score, n_reamostras=n_reamostras
        ),
        "erro_por_atleta": ic_bootstrap(por_atleta, n_reamostras=n_reamostras),
    }


def avaliar(
    df: pl.DataFrame,
    espec: EspecModelo,
    n_folds: int = N_FOLDS,
    n_reamostras: int = N_BOOTSTRAP,
) -> ResultadoAvaliacao:
    """
    Avalia um modelo com CV agrupada por atleta e predições out-of-fold.

    `n_reamostras` controla o bootstrap dos ICs; reduzir alarga levemente os
    intervalos em troca de tempo, útil em contexto interativo.
    """
    X, y, grupos = _matriz(df, espec)

    y_pred = np.empty_like(y)
    indices = pl.DataFrame({COL_ATHLETE: grupos})
    for treino, teste in folds_por_atleta(indices, n_folds=n_folds):
        modelo = espec.construir()
        modelo.fit(X[treino], y[treino])
        y_pred[teste] = modelo.predict(X[teste])

    erro_por_atleta = _erro_por_atleta(y, y_pred, grupos, espec.tarefa)

    return ResultadoAvaliacao(
        nome=espec.nome,
        categoria=espec.categoria,
        tarefa=espec.tarefa,
        protocolo=f"GroupKFold({n_folds}) por atleta, bootstrap={n_reamostras}",
        n_corridas=len(y),
        n_atletas=len(np.unique(grupos)),
        metricas=_metricas(
            y, y_pred, grupos, espec.tarefa, erro_por_atleta, n_reamostras=n_reamostras
        ),
        erro_por_atleta=erro_por_atleta,
        observacao=espec.observacao,
    )


def avaliar_protocolo_antigo(
    df: pl.DataFrame,
    espec: EspecModelo,
    seed: int = 42,
) -> dict[str, float]:
    """
    Reproduz o protocolo original (`train_test_split` aleatório por corrida)
    para quantificar o otimismo. Existe só como termo de comparação.
    """
    from sklearn.model_selection import train_test_split

    X, y, _ = _matriz(df, espec)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)

    modelo = espec.construir()
    modelo.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_te)

    if espec.tarefa == "regressao":
        return {
            "mae": float(mean_absolute_error(y_te, y_pred)),
            "rmse": float(root_mean_squared_error(y_te, y_pred)),
            "r2": float(r2_score(y_te, y_pred)),
        }
    return {
        "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "acuracia": float(accuracy_score(y_te, y_pred)),
    }
