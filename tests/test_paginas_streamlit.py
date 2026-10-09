"""
Teste de fumaça das páginas do Streamlit.

Não executa as páginas (os loaders de ADR-017/018 levam ~1 min na primeira
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
    "fonte_externa_disponivel",
    "desfecho_meia",
    "validacao_encaixada",
    "meta_aprendida",
    "recomendacoes_por_cluster",
    "comparacao_global_vs_cluster",
    "recomendacao_cluster_atleta",
}


def _paginas() -> list[Path]:
    return sorted([APP / "Home.py", *(APP / "pages").glob("*.py")])


def test_todas_as_paginas_compilam():
    for pagina in _paginas():
        ast.parse(pagina.read_text(encoding="utf-8"), filename=str(pagina))


def test_paginas_novas_existem():
    nomes = {p.name for p in (APP / "pages").glob("*.py")}

    assert "7_Recomendacao_por_Cluster.py" in nomes
    assert "8_Historico_Recomendacoes.py" in nomes
    assert "9_Protocolo_Experimental.py" in nomes
    assert "10_Desfecho_e_Meta.py" in nomes


def test_shared_expoe_os_loaders_das_paginas_novas():
    from app import shared

    for nome in LOADERS_ESPERADOS:
        assert callable(getattr(shared, nome)), nome


def test_paginas_novas_so_importam_loaders_existentes():
    from app import shared

    for nome in (
        "7_Recomendacao_por_Cluster.py",
        "9_Protocolo_Experimental.py",
        "10_Desfecho_e_Meta.py",
    ):
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

    assert "Protocolo Experimental" in texto
    assert "ADR-017" in texto


def test_home_lista_as_telas_novas():
    texto = (APP / "Home.py").read_text(encoding="utf-8")

    assert "Recomendação por Cluster" in texto
    assert "Protocolo Experimental" in texto
    assert "Desfecho e Meta" in texto


def test_protocolo_experimental_prioriza_recomendacao():
    texto = (APP / "pages" / "9_Protocolo_Experimental.py").read_text(encoding="utf-8")

    assert "Validação da recomendação" in texto
    assert "Conclusão em 30 segundos" in texto
    assert "Matriz de confusão" in texto
    assert "Amostra de casos" in texto
    idx_rec = texto.index("Validação da recomendação")
    idx_modelos = texto.index("Modelos auxiliares")
    assert idx_rec < idx_modelos
