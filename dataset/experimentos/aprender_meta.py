"""
Aprendizado da meta de distribuição de treino (ADR-019, item 4).

A `META` de ADR-007 (leve 60%, longão 20%, ritmo 12%, intervalado 8%) foi
afirmada, não medida. Aqui ela é **estimada a partir do desfecho**: qual
distribuição de tipos de treino se associa ao melhor tempo de **meia maratona**
(21,0975 km), mantendo volume e pace constantes.

Meia maratona e não maratona porque o público do projeto é o corredor amador
(ADR seção 1), e 21 km é a distância de prova dominante nesse público. Otimizar
para maratona produziria uma meta calibrada para um perfil que não é o do
usuário do sistema.

## Duas metas, e por que a restrita é a que vale

`META` tem um defeito estrutural que a ablação de ADR-017 expôs: o
classificador de ADR-004 é **relativo ao atleta** — `longao` é tudo acima do
percentil 75 de distância, o que fixa ~25% das corridas nessa classe por
construção. Nenhum atleta pode atingir "60% leve, 20% longão". O déficit de
`leve` fica positivo para todos, e a regra recomenda `leve` quase sempre.

Por isso o módulo estima duas metas:

- **irrestrita** — o ótimo no simplexo, ignorando o que o classificador
  consegue produzir. Serve de diagnóstico: mostra para onde os dados apontam.
- **factível** — o ótimo restrito ao intervalo observado de cada tipo. É a
  meta utilizável, porque um atleta pode de fato alcançá-la.

## O que isto não estabelece

Associação observacional, não causalidade. A meta aprendida descreve a
distribuição dos atletas que correm mais rápido com volume e pace comparáveis;
não prova que um atleta que mude sua distribuição vai melhorar. Um ensaio
controlado responderia isso, e está fora do alcance de dados observacionais.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from dataset.constants import ALFA, COL_TEMPO_PROVA, N_FOLDS, SEED
from dataset.experimentos.desfecho import (
    FEATURES_COMPOSICAO,
    FEATURES_VOLUME,
    distancia_a_meta,
)
from dataset.models.previsao_treino.model import META

TIPOS = [c.removeprefix("prop_") for c in FEATURES_COMPOSICAO]

# Percentis que delimitam a região factível de cada tipo.
PERCENTIL_INFERIOR = 5
PERCENTIL_SUPERIOR = 95


@dataclass
class MetaAprendida:
    nome: str
    meta: dict[str, float]
    tempo_previsto_min: float
    restrita: bool
    limites: dict[str, tuple[float, float]] = field(default_factory=dict)
    ic: dict[str, tuple[float, float]] = field(default_factory=dict)

    TOLERANCIA_FRONTEIRA = 1e-3

    def na_fronteira(self, tipo: str) -> bool:
        """
        Indica se o ótimo daquele tipo ficou colado num limite.

        Importa para a interpretação: componente na fronteira significa que o
        modelo quer *mais* (ou menos) do que qualquer atleta da amostra
        pratica — o ótimo verdadeiro está fora da região observada, e a meta
        aprendida é o melhor alcançável, não o melhor possível.
        """
        if tipo not in self.limites:
            return False
        inferior, superior = self.limites[tipo]
        valor = self.meta[tipo]
        return (
            abs(valor - inferior) < self.TOLERANCIA_FRONTEIRA
            or abs(valor - superior) < self.TOLERANCIA_FRONTEIRA
        )

    LIMITE_AMPLITUDE_ESTAVEL = 0.10

    def amplitude_ic(self, tipo: str) -> float:
        """Largura do IC bootstrap daquele componente."""
        if tipo not in self.ic:
            return float("nan")
        inferior, superior = self.ic[tipo]
        return superior - inferior

    def bem_identificada(self, tipo: str) -> bool:
        """
        Indica se o componente está bem determinado pelos dados.

        Amplitude de IC acima de ~10 pontos percentuais significa que
        reamostrar a base move o ótimo daquele tipo por uma margem maior que a
        própria diferença em relação à `META` de ADR-007 — ou seja, o número
        aprendido não é confiável o suficiente para substituir o afirmado.
        """
        amplitude = self.amplitude_ic(tipo)
        return bool(amplitude == amplitude and amplitude < self.LIMITE_AMPLITUDE_ESTAVEL)

    def como_tabela(self) -> pl.DataFrame:
        return pl.DataFrame([
            {
                "tipo": tipo,
                "meta_aprendida": round(self.meta[tipo], 4),
                "meta_global": META[tipo],
                "diferenca": round(self.meta[tipo] - META[tipo], 4),
                "ic_inferior": round(self.ic.get(tipo, (float("nan"),) * 2)[0], 4),
                "ic_superior": round(self.ic.get(tipo, (float("nan"),) * 2)[1], 4),
                "amplitude_ic": round(self.amplitude_ic(tipo), 4),
                "bem_identificada": self.bem_identificada(tipo),
                "na_fronteira": self.na_fronteira(tipo),
                "limite_inferior": round(self.limites.get(tipo, (0.0, 1.0))[0], 4),
                "limite_superior": round(self.limites.get(tipo, (0.0, 1.0))[1], 4),
            }
            for tipo in TIPOS
        ])


PARES = [
    (i, j) for i in range(len(TIPOS)) for j in range(i + 1, len(TIPOS))
]


def _termos_scheffe(proporcoes: np.ndarray) -> np.ndarray:
    """
    Expande as proporções nos termos do polinômio quadrático de Scheffé:
    os 4 termos lineares mais os 6 produtos cruzados.

    Um modelo **linear** nas proporções tem seu mínimo necessariamente num
    vértice do simplexo — ele só consegue dizer "faça 100% de um tipo". Como as
    4 proporções somam 1, os termos quadráticos puros (xi²) são redundantes com
    os cruzados, e a forma canônica para dados de mistura é a de Scheffé:
    E[y] = Σ bi·xi + Σ bij·xi·xj. Com os cruzados o ótimo pode ser interior,
    que é o que uma meta de distribuição precisa ser.
    """
    proporcoes = np.atleast_2d(proporcoes)
    cruzados = np.stack(
        [proporcoes[:, i] * proporcoes[:, j] for i, j in PARES], axis=1
    )
    return np.hstack([proporcoes, cruzados])


def _matriz(tabela: pl.DataFrame, alvo: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Desenho do modelo: features de volume (lineares) + Scheffé nas proporções.

    As colunas de volume vêm primeiro, para que `aprender_meta` possa fixá-las
    na mediana ao otimizar.
    """
    colunas = [*FEATURES_VOLUME, *FEATURES_COMPOSICAO]
    limpo = tabela.select([alvo, *colunas]).drop_nulls()

    volume = limpo.select(FEATURES_VOLUME).to_numpy()
    proporcoes = limpo.select(FEATURES_COMPOSICAO).to_numpy()

    return np.hstack([volume, _termos_scheffe(proporcoes)]), limpo[alvo].to_numpy()


def _desenho(volume_fixo: np.ndarray, proporcoes: np.ndarray) -> np.ndarray:
    """Monta uma linha do desenho a partir do volume fixo e das proporções."""
    return np.hstack([volume_fixo.reshape(1, -1), _termos_scheffe(proporcoes)])


def _ajustar(X: np.ndarray, y: np.ndarray) -> Ridge:
    """
    Ridge e não OLS: as 4 proporções somam 1 e os termos cruzados são funções
    delas, o que torna o desenho fortemente colinear por construção. Sem
    regularização os coeficientes ficam instáveis e o ótimo salta entre
    reamostras do bootstrap.
    """
    modelo = Ridge(alpha=1.0, random_state=SEED)
    modelo.fit(X, y)
    return modelo


def _limites_observados(tabela: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Região factível: o que o classificador de ADR-004 de fato produz."""
    return {
        tipo: (
            float(np.percentile(tabela[f"prop_{tipo}"].to_numpy(), PERCENTIL_INFERIOR)),
            float(np.percentile(tabela[f"prop_{tipo}"].to_numpy(), PERCENTIL_SUPERIOR)),
        )
        for tipo in TIPOS
    }


def _otimizar(
    modelo: Ridge,
    volume_fixo: np.ndarray,
    limites: dict[str, tuple[float, float]] | None,
) -> tuple[dict[str, float], float]:
    """
    Minimiza o tempo previsto sobre o simplexo das 4 proporções.

    SLSQP porque o problema tem a restrição de igualdade `soma = 1` mais
    caixas por variável. O ponto inicial é a `META` atual projetada nos
    limites, e o resultado é verificado contra múltiplos inícios aleatórios
    para reduzir risco de ótimo local.
    """
    def prever(proporcoes: np.ndarray) -> float:
        return float(modelo.predict(_desenho(volume_fixo, proporcoes))[0])

    caixas = (
        [limites[tipo] for tipo in TIPOS] if limites else [(0.0, 1.0)] * len(TIPOS)
    )
    restricao = {"type": "eq", "fun": lambda p: float(np.sum(p) - 1.0)}

    inicios = [np.array([META[t] for t in TIPOS], dtype=float)]
    rng = np.random.default_rng(SEED)
    for _ in range(8):
        bruto = rng.dirichlet(np.ones(len(TIPOS)))
        inicios.append(bruto)

    melhor_p, melhor_valor = None, np.inf
    for inicio in inicios:
        recortado = np.clip(inicio, [c[0] for c in caixas], [c[1] for c in caixas])
        if recortado.sum() > 0:
            recortado = recortado / recortado.sum()

        resultado = minimize(
            prever,
            recortado,
            method="SLSQP",
            bounds=caixas,
            constraints=[restricao],
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if resultado.success and resultado.fun < melhor_valor:
            melhor_valor, melhor_p = float(resultado.fun), resultado.x

    if melhor_p is None:
        raise RuntimeError("otimização da meta não convergiu em nenhum início")

    melhor_p = np.clip(melhor_p, 0.0, 1.0)
    melhor_p = melhor_p / melhor_p.sum()

    return {tipo: float(p) for tipo, p in zip(TIPOS, melhor_p)}, melhor_valor


def aprender_meta(
    tabela: pl.DataFrame,
    alvo: str = COL_TEMPO_PROVA,
    restrita: bool = True,
    n_bootstrap: int = 200,
) -> MetaAprendida:
    """
    Estima a meta ótima de distribuição para o desfecho `alvo`.

    Volume e pace são fixados na **mediana** da amostra: sem isso a otimização
    responderia "treine mais e mais rápido", que é verdade e inútil — a
    pergunta é como distribuir um volume dado.
    """
    X, y = _matriz(tabela, alvo)
    modelo = _ajustar(X, y)

    n_volume = len(FEATURES_VOLUME)
    volume_fixo = np.median(X[:, :n_volume], axis=0)
    limites = _limites_observados(tabela) if restrita else None

    meta, tempo = _otimizar(modelo, volume_fixo, limites)

    # IC por bootstrap: reajusta modelo e reotimiza em cada reamostra.
    rng = np.random.default_rng(SEED)
    amostras: dict[str, list[float]] = {tipo: [] for tipo in TIPOS}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        try:
            meta_b, _tempo_b = _otimizar(
                _ajustar(X[idx], y[idx]), volume_fixo, limites
            )
        except RuntimeError:
            continue
        for tipo, valor in meta_b.items():
            amostras[tipo].append(valor)

    ic = {
        tipo: (
            float(np.percentile(valores, 100 * ALFA / 2)),
            float(np.percentile(valores, 100 * (1 - ALFA / 2))),
        )
        for tipo, valores in amostras.items()
        if valores
    }

    return MetaAprendida(
        nome="factível (restrita ao observado)" if restrita else "irrestrita (simplexo)",
        meta=meta,
        tempo_previsto_min=tempo,
        restrita=restrita,
        limites=limites or {},
        ic=ic,
    )


def validar_meta_fora_da_amostra(
    tabela: pl.DataFrame,
    alvo: str = COL_TEMPO_PROVA,
    n_folds: int = N_FOLDS,
    restrita: bool = True,
) -> pl.DataFrame:
    """
    Valida a meta aprendida **fora da amostra**.

    Em cada fold a meta é aprendida só no treino e a aderência a ela é medida
    no teste. Sem isso a meta aprendida venceria por construção, já que foi
    ajustada nos mesmos dados em que é avaliada.

    Compara três metas pelo poder da aderência em explicar o desfecho:
    a de ADR-007, a aprendida e a uniforme.
    """
    X, y = _matriz(tabela, alvo)
    indices = np.arange(len(y))

    correlacoes: dict[str, list[float]] = {
        "meta global": [],
        "aprendida": [],
        "uniforme": [],
    }
    uniforme = dict.fromkeys(TIPOS, 1.0 / len(TIPOS))

    for treino, teste in KFold(n_folds, shuffle=True, random_state=SEED).split(indices):
        meta_fold = aprender_meta(
            tabela[treino], alvo=alvo, restrita=restrita, n_bootstrap=0
        ).meta

        subset = tabela[teste]
        for nome, meta in (
            ("meta global", META),
            ("aprendida", meta_fold),
            ("uniforme", uniforme),
        ):
            com_distancia = distancia_a_meta(subset, meta=meta)
            d = com_distancia["distancia_meta"].to_numpy()
            alvo_teste = com_distancia[alvo].to_numpy()
            if np.std(d) > 0:
                correlacoes[nome].append(
                    float(np.corrcoef(d, alvo_teste)[0, 1])
                )

    return pl.DataFrame([
        {
            "meta": nome,
            "corr_media_fora_amostra": round(float(np.mean(valores)), 4),
            "corr_desvio": round(float(np.std(valores)), 4),
            "folds": len(valores),
            "interpretacao": (
                "aderência prediz desempenho"
                if np.mean(valores) > 0.05
                else "aderência não prediz desempenho"
            ),
        }
        for nome, valores in correlacoes.items()
        if valores
    ]).sort("corr_media_fora_amostra", descending=True)


def tabela_regiao_factivel(tabela: pl.DataFrame) -> pl.DataFrame:
    """
    Compara a `META` de ADR-007 com o que o classificador de ADR-004 produz.

    Evidência direta do defeito estrutural: `dentro_da_regiao=false` significa
    que nenhum atleta pode atingir aquele componente da meta.
    """
    limites = _limites_observados(tabela)

    return pl.DataFrame([
        {
            "tipo": tipo,
            "meta_global": META[tipo],
            "observado_mediana": round(
                float(tabela[f"prop_{tipo}"].median()), 4
            ),
            "observado_p5": round(limites[tipo][0], 4),
            "observado_p95": round(limites[tipo][1], 4),
            "dentro_da_regiao": bool(
                limites[tipo][0] <= META[tipo] <= limites[tipo][1]
            ),
        }
        for tipo in TIPOS
    ])
