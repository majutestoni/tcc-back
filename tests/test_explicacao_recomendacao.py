"""Testes da justificativa descritiva da recomendação por cluster."""

import polars as pl

from dataset.models.previsao_treino.recomendacao_cluster import explicar_recomendacao


def test_explicar_recomendacao_menciona_tipo_distancia_pace_tempo():
    rec = {
        "tipo": "longao",
        "descricao": "Longão em ritmo confortável",
        "distancia_km": 14.0,
        "pace_alvo_min_km": 6.2,
        "tempo_estimado_min": 86.8,
        "perfil_cluster": "volume_alto",
        "rotulo_cluster": "Volume alto · M",
    }
    atual = {"leve": 0.80, "longao": 0.05, "ritmo": 0.10, "intervalado": 0.05}
    meta = {"leve": 0.50, "longao": 0.25, "ritmo": 0.15, "intervalado": 0.10}
    deficits = {t: meta[t] - atual[t] for t in meta}
    janela = pl.DataFrame(
        {
            "tipo": ["leve", "leve", "longao", "ritmo"],
            "distancia_km": [7.0, 8.0, 12.5, 9.0],
            "pace_min_km": [5.8, 5.9, 6.1, 5.5],
        }
    )

    texto = explicar_recomendacao(
        rec=rec,
        atual=atual,
        meta=meta,
        deficits=deficits,
        ultimos=["leve", "ritmo"],
        corridas_janela=janela,
        info_cluster={
            "perfil_cluster": "volume_alto",
            "rotulo_cluster": "Volume alto · M",
        },
        parametros_tipo={
            "distancia_km": (12.0, 16.0),
            "pace_offset_min": 0.4,
            "descricao": "Longão em ritmo confortável",
        },
    )

    assert "longão" in texto.lower() or "longao" in texto.lower()
    assert "14.0 km" in texto or "14 km" in texto
    assert "Pace alvo" in texto
    assert "Tempo estimado" in texto
    assert "Volume alto" in texto
    assert "12.5 km" in texto  # mediana do longão do atleta na janela
    assert "anti-fadiga" in texto.lower()


def test_explicar_recomendacao_antifadiga_quando_tipo_muda():
    rec = {
        "tipo": "leve",
        "descricao": "Corrida leve / regenerativa",
        "distancia_km": 6.5,
        "pace_alvo_min_km": 6.5,
        "tempo_estimado_min": 42.0,
    }
    atual = {"leve": 0.50, "longao": 0.05, "ritmo": 0.10, "intervalado": 0.35}
    meta = {"leve": 0.50, "longao": 0.25, "ritmo": 0.15, "intervalado": 0.10}
    deficits = {t: meta[t] - atual[t] for t in meta}
    # maior déficit ponderado tende a longao; ultimo = intervalado → antifadiga

    texto = explicar_recomendacao(
        rec=rec,
        atual=atual,
        meta=meta,
        deficits=deficits,
        ultimos=["intervalado"],
        corridas_janela=pl.DataFrame(
            {"tipo": ["intervalado"], "distancia_km": [6.0], "pace_min_km": [5.5]}
        ),
        info_cluster={"perfil_cluster": "volume_moderado", "rotulo_cluster": "Mod · F"},
    )

    assert "anti-fadiga" in texto.lower()
    assert "leve" in texto.lower()


def test_explicar_recomendacao_vazia_sem_rec():
    assert explicar_recomendacao(
        rec={},
        atual={},
        meta={},
        deficits={},
        ultimos=[],
    ) == ""
