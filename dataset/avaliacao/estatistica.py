"""
Inferência estatística para comparação de modelos (ADR-017).

O protocolo antigo reportava uma métrica pontual por modelo e declarava
vencedor o de maior número. Sem medida de incerteza não é possível dizer se a
diferença é real. Aqui:

- **IC por bootstrap de atletas** (cluster bootstrap): as corridas de um atleta
  não são independentes, então reamostra-se atleta, não corrida. Reamostrar
  corrida subestimaria o intervalo.
- **Wilcoxon pareado** para comparar dois modelos nos *mesmos* atletas. Pareado
  porque a variação entre atletas (uns são simplesmente mais difíceis de
  prever) é muito maior que a diferença entre modelos; o teste pareado a
  cancela. Não paramétrico porque erros de predição não são normais.
- **Correção de Holm** porque comparar 8 modelos entre si gera 28 testes, e a
  chance de um falso positivo cresce com o número de comparações.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon

from dataset.constants import ALFA, N_BOOTSTRAP, SEED


@dataclass(frozen=True)
class IntervaloConfianca:
    estimativa: float
    inferior: float
    superior: float
    nivel: float

    def __str__(self) -> str:
        return f"{self.estimativa:.4f} [{self.inferior:.4f}, {self.superior:.4f}]"


@dataclass(frozen=True)
class ComparacaoPareada:
    modelo_a: str
    modelo_b: str
    n_pares: int
    mediana_diferenca: float
    media_diferenca: float
    estatistica: float
    p_valor: float
    p_ajustado: float | None = None

    @property
    def significativa(self) -> bool:
        p = self.p_ajustado if self.p_ajustado is not None else self.p_valor
        return bool(p < ALFA)

    @property
    def vencedor(self) -> str | None:
        """
        Modelo com menor erro, se a diferença for significativa.

        A direção vem da **média** das diferenças, não da mediana: com poucos
        casos por atleta a mediana é frequentemente exatamente zero mesmo
        quando o Wilcoxon (que usa postos) detecta deslocamento, e usar a
        mediana reportaria "empate" numa comparação significativa.
        """
        if not self.significativa or self.media_diferenca == 0:
            return None
        return self.modelo_a if self.media_diferenca < 0 else self.modelo_b


def ic_bootstrap(
    valores: np.ndarray,
    n_reamostras: int = N_BOOTSTRAP,
    alfa: float = ALFA,
    seed: int = SEED,
) -> IntervaloConfianca:
    """
    IC percentil por bootstrap da média de `valores`.

    Cada valor deve ser a métrica de **um atleta** (uma unidade independente),
    e não de uma corrida — daí o nome "bootstrap de atletas".
    """
    valores = np.asarray(valores, dtype=float)
    valores = valores[~np.isnan(valores)]
    if valores.size == 0:
        raise ValueError("sem valores válidos para bootstrap")
    if valores.size == 1:
        v = float(valores[0])
        return IntervaloConfianca(v, v, v, 1 - alfa)

    rng = np.random.default_rng(seed)
    amostras = rng.choice(valores, size=(n_reamostras, valores.size), replace=True)
    medias = amostras.mean(axis=1)

    return IntervaloConfianca(
        estimativa=float(valores.mean()),
        inferior=float(np.percentile(medias, 100 * alfa / 2)),
        superior=float(np.percentile(medias, 100 * (1 - alfa / 2))),
        nivel=1 - alfa,
    )


def ic_bootstrap_metrica(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grupos: np.ndarray,
    metrica,
    n_reamostras: int = N_BOOTSTRAP,
    alfa: float = ALFA,
    seed: int = SEED,
) -> IntervaloConfianca:
    """
    IC de uma métrica global (ex.: R²) reamostrando **atletas** inteiros.

    Necessário para métricas que não decompõem por atleta: R² calculado por
    atleta não equivale ao R² global, então não dá para usar `ic_bootstrap`.
    """
    y_true, y_pred, grupos = map(np.asarray, (y_true, y_pred, grupos))
    atletas = np.unique(grupos)
    indices_por_atleta = {a: np.flatnonzero(grupos == a) for a in atletas}

    rng = np.random.default_rng(seed)
    valores = []
    for _ in range(n_reamostras):
        sorteados = rng.choice(atletas, size=atletas.size, replace=True)
        idx = np.concatenate([indices_por_atleta[a] for a in sorteados])
        valores.append(metrica(y_true[idx], y_pred[idx]))

    valores = np.asarray(valores, dtype=float)
    return IntervaloConfianca(
        estimativa=float(metrica(y_true, y_pred)),
        inferior=float(np.percentile(valores, 100 * alfa / 2)),
        superior=float(np.percentile(valores, 100 * (1 - alfa / 2))),
        nivel=1 - alfa,
    )


def wilcoxon_pareado(
    nome_a: str,
    erros_a: dict[int, float],
    nome_b: str,
    erros_b: dict[int, float],
) -> ComparacaoPareada:
    """
    Compara dois modelos pelos erros nos atletas em comum.

    `erros_*` mapeia atleta -> erro (menor = melhor). Sinal de
    `mediana_diferenca` negativo favorece o modelo A.
    """
    comuns = sorted(set(erros_a) & set(erros_b))
    if not comuns:
        raise ValueError(f"{nome_a} e {nome_b} não têm atletas em comum")

    a = np.array([erros_a[i] for i in comuns], dtype=float)
    b = np.array([erros_b[i] for i in comuns], dtype=float)
    diferenca = a - b

    if np.allclose(diferenca, 0.0):
        return ComparacaoPareada(nome_a, nome_b, len(comuns), 0.0, 0.0, 0.0, 1.0)

    estatistica, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return ComparacaoPareada(
        modelo_a=nome_a,
        modelo_b=nome_b,
        n_pares=len(comuns),
        mediana_diferenca=float(np.median(diferenca)),
        media_diferenca=float(np.mean(diferenca)),
        estatistica=float(estatistica),
        p_valor=float(p),
    )


def correcao_holm(comparacoes: list[ComparacaoPareada]) -> list[ComparacaoPareada]:
    """
    Aplica Holm-Bonferroni aos p-valores, preenchendo `p_ajustado`.

    Holm em vez de Bonferroni puro porque é uniformemente mais poderoso
    mantendo o mesmo controle do erro tipo I familiar.
    """
    if not comparacoes:
        return []

    ordem = sorted(range(len(comparacoes)), key=lambda i: comparacoes[i].p_valor)
    m = len(comparacoes)
    ajustados = [0.0] * m
    maximo_corrente = 0.0

    for posicao, indice in enumerate(ordem):
        ajustado = min(1.0, (m - posicao) * comparacoes[indice].p_valor)
        maximo_corrente = max(maximo_corrente, ajustado)
        ajustados[indice] = maximo_corrente

    return [
        ComparacaoPareada(
            modelo_a=c.modelo_a,
            modelo_b=c.modelo_b,
            n_pares=c.n_pares,
            mediana_diferenca=c.mediana_diferenca,
            media_diferenca=c.media_diferenca,
            estatistica=c.estatistica,
            p_valor=c.p_valor,
            p_ajustado=ajustados[i],
        )
        for i, c in enumerate(comparacoes)
    ]


def comparar_todos(erros_por_modelo: dict[str, dict[int, float]]) -> list[ComparacaoPareada]:
    """Todas as comparações par a par, já corrigidas por Holm."""
    nomes = sorted(erros_por_modelo)
    comparacoes = [
        wilcoxon_pareado(a, erros_por_modelo[a], b, erros_por_modelo[b])
        for i, a in enumerate(nomes)
        for b in nomes[i + 1:]
    ]
    return correcao_holm(comparacoes)
