import importlib

_EXPORTS = {
    "treina_kmeans": "kmeans",
    "treina_kmeans_por_genero": "kmeans",
    "nomear_categorias": "nomear_clusters",
    "tabela_descritiva_clusters": "nomear_clusters",
    "definir_perfil_atleta": "perfil_atletas",
    "montar_perfil_atletas": "perfil_atletas",
    "tabela_fichas": "tabela_fichas",
    "avaliar_k": "validacao_cluster",
    "comparar_com_perfil_regras": "validacao_cluster",
    "estabilidade_kmeans": "validacao_cluster",
    "metricas_externas": "validacao_cluster",
    "metricas_internas": "validacao_cluster",
    "silhouette_por_amostra": "validacao_cluster",
    "validar_resultado_kmeans": "validacao_cluster",
    "montar_fichas_atletas": "visualizacao_fichas",
    "plotar_ficha_atleta": "visualizacao_fichas",
    "plotar_fichas_atletas": "visualizacao_fichas",
    "comparar_combinacoes_features": "comparar_features",
    "comparar_combinacoes_e_k": "comparar_features",
    "sugerir_melhor_config": "comparar_features",
}

__all__ = list(_EXPORTS)


def __getattr__(nome):
    modulo = _EXPORTS.get(nome)
    if modulo is None:
        raise AttributeError(f"module {__name__!r} has no attribute {nome!r}")
    modulo_importado = importlib.import_module(f"{__name__}.{modulo}")
    valor = getattr(modulo_importado, nome)
    globals()[nome] = valor
    return valor


def __dir__():
    return sorted(__all__)
