from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl


@dataclass
class ResultadoModelo:
    nome: str
    categoria: Literal["pace", "tempo", "treino", "corredor"]
    metricas: dict[str, float]
    detalhes: dict = field(default_factory=dict)
    recomendacoes: pl.DataFrame | None = None
    df_exibir: pl.DataFrame | None = None
    col_tipo: str | None = None
    parametros_calibrados: dict | None = None
    y_test: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    y_true_cls: np.ndarray | None = None
    y_pred_cls: np.ndarray | None = None
    class_names: list[str] | None = None
