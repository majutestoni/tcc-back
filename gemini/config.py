from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_ENV_PADRAO = Path(".env")

MODELO_PADRAO = "gemini-3.6-flash"
TEMPERATURA_PADRAO = 0.3


def _carregar_env(caminho: Path = _ENV_PADRAO) -> None:
    """Carrega variáveis de um arquivo .env simples, sem dependências externas."""
    if not caminho.exists():
        return
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, valor = linha.split("=", 1)
        os.environ.setdefault(chave.strip(), valor.strip().strip('"').strip("'"))


TENTATIVAS_PADRAO = 3
ESPERA_BASE_PADRAO = 2.0


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str | None
    modelo: str = MODELO_PADRAO
    temperatura: float = TEMPERATURA_PADRAO
    max_output_tokens: int | None = None
    tentativas_max: int = TENTATIVAS_PADRAO
    espera_base_s: float = ESPERA_BASE_PADRAO

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        _carregar_env()
        max_tokens = os.environ.get("GEMINI_MAX_OUTPUT_TOKENS")
        return cls(
            api_key=os.environ.get("GEMINI_API_KEY"),
            modelo=os.environ.get("GEMINI_MODEL", MODELO_PADRAO),
            temperatura=float(os.environ.get("GEMINI_TEMPERATURE", TEMPERATURA_PADRAO)),
            max_output_tokens=int(max_tokens) if max_tokens else None,
            tentativas_max=int(os.environ.get("GEMINI_MAX_TENTATIVAS", TENTATIVAS_PADRAO)),
            espera_base_s=float(os.environ.get("GEMINI_ESPERA_BASE_S", ESPERA_BASE_PADRAO)),
        )

    def validar(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY não definida. Configure a variável de ambiente "
                "ou o arquivo .env (ver .env.example)."
            )
