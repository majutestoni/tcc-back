"""
Coloca a raiz do repositório no `sys.path` para os testes.

Sem isto, `pytest tests/` insere apenas `tests/` no path (modo de import
`prepend`) e `from dataset...` falha.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
