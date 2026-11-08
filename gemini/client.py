from __future__ import annotations

import logging
import time
from typing import Type, TypeVar

from pydantic import BaseModel

from gemini.config import GeminiConfig

T = TypeVar("T", bound=BaseModel)

CODIGOS_TRANSITORIOS = (429, 500, 502, 503, 504)

logging.getLogger("google_genai").setLevel(logging.ERROR)


class GeminiClient:
    """Wrapper fino sobre o SDK google-genai para saída estruturada (JSON)."""

    def __init__(self, config: GeminiConfig | None = None) -> None:
        self.config = config or GeminiConfig.from_env()
        self.config.validar()
        self._cliente = None

    def _obter_cliente(self):
        if self._cliente is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "Pacote google-genai não instalado. Rode: pip install google-genai"
                ) from exc
            self._cliente = genai.Client(api_key=self.config.api_key)
        return self._cliente

    def gerar_estruturado(
        self,
        contents: str,
        schema: Type[T],
        system_instruction: str | None = None,
    ) -> T:
        from google.genai import errors, types

        cliente = self._obter_cliente()
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.config.temperatura,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=self.config.max_output_tokens,
        )

        ultimo_erro: Exception | None = None
        for tentativa in range(self.config.tentativas_max):
            try:
                resposta = cliente.models.generate_content(
                    model=self.config.modelo,
                    contents=contents,
                    config=config,
                )
                parsed = resposta.parsed
                if parsed is None:
                    raise RuntimeError(
                        "Resposta do Gemini não pôde ser parseada. "
                        f"Texto bruto: {resposta.text}"
                    )
                return parsed
            except errors.APIError as erro:
                ultimo_erro = erro
                transitorio = getattr(erro, "code", None) in CODIGOS_TRANSITORIOS
                ultima_tentativa = tentativa == self.config.tentativas_max - 1
                if transitorio and not ultima_tentativa:
                    time.sleep(self.config.espera_base_s * (2**tentativa))
                    continue
                if transitorio:
                    raise RuntimeError(
                        f"Gemini indisponível (código {erro.code}) após "
                        f"{self.config.tentativas_max} tentativas. Tente novamente "
                        "em instantes ou troque o modelo via GEMINI_MODEL."
                    ) from erro
                raise

        raise ultimo_erro  # type: ignore[misc]
