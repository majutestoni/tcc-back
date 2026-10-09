"""
Experimentos do projeto (ADR-017, ADR-018, ADR-019).

Quatro blocos, executáveis por `python -m dataset.experimentos <bloco>`:

- `protocolo` — reavalia os 8 modelos sob particionamento agrupado por atleta,
  com IC e testes pareados (ADR-017).
- `baselines` — compara a regra de ADR-007 com recomendadores triviais e roda
  a ablação dos seus componentes (ADR-017).
- `desfecho` — validação externa com o tempo de meia maratona como desfecho
  (ADR-018).
- `meta` — estima a meta de distribuição a partir do desfecho (ADR-019).

Imports lazy (ADR-012): este pacote arrasta sklearn, scipy e a fonte externa.
"""

from __future__ import annotations

__all__ = [
    "aprender_meta",
    "avaliar_recomendador",
    "baselines_padrao",
    "comparar_encaixados",
    "correlacoes_desfecho",
    "montar_casos",
    "montar_tabela_desfecho",
    "rodar_ablacao",
    "tabela_recomendadores",
    "tabela_regiao_factivel",
    "validar_meta_fora_da_amostra",
]

_MAPA = {
    "aprender_meta": "aprender_meta",
    "tabela_regiao_factivel": "aprender_meta",
    "validar_meta_fora_da_amostra": "aprender_meta",
    "baselines_padrao": "baselines",
    "rodar_ablacao": "ablacao",
    "avaliar_recomendador": "protocolo_recomendacao",
    "montar_casos": "protocolo_recomendacao",
    "tabela_recomendadores": "protocolo_recomendacao",
    "montar_tabela_desfecho": "desfecho",
    "comparar_encaixados": "validacao_externa",
    "correlacoes_desfecho": "validacao_externa",
}


def __getattr__(nome: str):
    if nome not in _MAPA:
        raise AttributeError(f"module {__name__!r} has no attribute {nome!r}")

    from importlib import import_module

    modulo = import_module(f"{__name__}.{_MAPA[nome]}")
    return getattr(modulo, nome)


def __dir__() -> list[str]:
    return sorted(__all__)
