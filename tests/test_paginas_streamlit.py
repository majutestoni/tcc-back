"""
Teste de fumaça das páginas do Streamlit.

Não executa as páginas (os loaders de avaliação levam ~1 min na primeira
vez, o que é caro demais para um teste de unidade). Verifica o que quebra
silenciosamente: página com erro de sintaxe, loader renomeado e a checagem de
base externa ausente.
"""

import ast
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
APP = RAIZ / "app"

LOADERS_ESPERADOS = {
    "avaliacao_protocolo",
    "comparacao_baselines",
    "ablacao_adr007",
    "recomendacoes_por_cluster",
    "recomendacao_cluster_atleta",
    "fichas_atletas",
    "perfil_multidim",
}


def _paginas() -> list[Path]:
    return sorted([APP / "Home.py", *(APP / "pages").glob("*.py")])


def test_todas_as_paginas_compilam():
    for pagina in _paginas():
        ast.parse(pagina.read_text(encoding="utf-8"), filename=str(pagina))


def test_paginas_esperadas_existem():
    nomes = {p.name for p in (APP / "pages").glob("*.py")}

    assert "6_Recomendacao_Treino.py" in nomes
    assert "7_Historico_Recomendacoes.py" in nomes
    assert "8_Validacao_da_Recomendacao.py" in nomes
    assert "8_Protocolo_Experimental.py" not in nomes
    assert "7_Recomendacao_por_Cluster.py" not in nomes
    assert "10_Desfecho_e_Meta.py" not in nomes
    assert "9_Desfecho_e_Meta.py" not in nomes


def test_shared_expoe_os_loaders_usados():
    from app import shared

    for nome in LOADERS_ESPERADOS:
        assert callable(getattr(shared, nome)), nome


def test_paginas_so_importam_loaders_existentes():
    from app import shared

    for nome in ("6_Recomendacao_Treino.py", "8_Validacao_da_Recomendacao.py"):
        arvore = ast.parse((APP / "pages" / nome).read_text(encoding="utf-8"))
        importados = [
            alias.name
            for no in ast.walk(arvore)
            if isinstance(no, ast.ImportFrom) and no.module == "app.shared"
            for alias in no.names
        ]

        assert importados, f"{nome} não importa nada de app.shared"
        for simbolo in importados:
            assert hasattr(shared, simbolo), f"{nome} importa {simbolo}, que não existe"


def test_fonte_externa_indisponivel_nao_levanta(monkeypatch=None):
    import os

    from app.shared import fonte_externa_disponivel

    anterior = os.environ.get("DIRETORIO_DADOS_EXTERNOS")
    os.environ["DIRETORIO_DADOS_EXTERNOS"] = "/caminho/que/nao/existe"
    try:
        disponivel, diretorio = fonte_externa_disponivel()
        assert disponivel is False
        assert "nao/existe" in diretorio
    finally:
        if anterior is None:
            os.environ.pop("DIRETORIO_DADOS_EXTERNOS", None)
        else:
            os.environ["DIRETORIO_DADOS_EXTERNOS"] = anterior


def test_comparacao_modelos_avisa_sobre_protocolo_antigo():
    texto = (APP / "pages" / "3_Comparacao_Modelos.py").read_text(encoding="utf-8")

    assert "Validação da recomendação" in texto
    assert "ADR-017" in texto


def test_home_lista_telas_atuais():
    texto = (APP / "Home.py").read_text(encoding="utf-8")

    assert "Recomendação de Treino" in texto
    assert "Validação da recomendação" in texto
    assert "Protocolo Experimental" not in texto
    assert "Recomendação por Cluster" not in texto
    assert "Desfecho e Meta" not in texto


def test_recomendacao_treino_e_por_cluster_com_ficha():
    texto = (APP / "pages" / "6_Recomendacao_Treino.py").read_text(encoding="utf-8")

    assert "recomendacao_cluster_atleta" in texto
    assert "recomendacao_regras" not in texto
    assert "Posição relativa" in texto
    assert "Na amostra global" in texto
    assert "No cluster" in texto
    assert "Ficha do cluster" in texto
    assert "plotar_ficha_atleta" in texto
    assert "perfil_multidim_com_cluster" in texto
    assert "corridas_janela_atleta" in texto
    assert "Janela deslizante" in texto


def test_validacao_recomendacao_prioriza_desfecho():
    texto = (APP / "pages" / "8_Validacao_da_Recomendacao.py").read_text(
        encoding="utf-8"
    )

    assert "Validação da recomendação" in texto
    assert "A regra funciona?" in texto
    assert "eficacia_por_desfecho" in texto
    assert "Eficácia (desfecho)" in texto
    assert "Imitação (contraste)" in texto
    assert "**Déficit**" in texto
    assert "**Baseline**" in texto
    assert "Conclusão em 30 segundos" in texto
    idx_ef = texto.index("Eficácia (desfecho)")
    idx_imit = texto.index("Imitação (contraste)")
    idx_modelos = texto.index("Modelos auxiliares")
    assert idx_ef < idx_imit < idx_modelos
