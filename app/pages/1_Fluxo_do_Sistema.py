"""
Mapa do fluxo científico e da tomada de decisão — espelho de FLUXO.md.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Fluxo do sistema", page_icon="🗺️", layout="wide")

st.title("Fluxo do sistema")
st.caption(
    "Do Strava à validação da recomendação — o mesmo mapa de `FLUXO.md`, "
    "navegável nesta demo."
)

ETAPAS = [
    {
        "id": "coleta",
        "nome": "1 · Coleta",
        "resumo": "CSV Kaggle (snapshot Strava). Futuro: OAuth via backend.",
        "por_que": (
            "Volume histórico estável e reprodutível para treinar, clusterizar "
            "e validar. API ao vivo fica no orquestrador (SPEC)."
        ),
        "detalhe": """
- Base primária: `dataset/raw-data-kaggle.csv`
- Fonte externa (desfecho 21 km): long-distance — `bash scripts/baixar_long_distance.sh`
- Sem desfecho na base Kaggle → só dá para medir *imitação* de hábito (ADR-018)
        """,
        "tela": None,
        "adr": "ADR-018",
    },
    {
        "id": "limpeza",
        "nome": "2 · Limpeza + engenharia",
        "resumo": "Gênero → FC → filtros → pace → tipo_treino relativo.",
        "por_que": (
            "Polars + limiares só em `constants.py` (ADR-001/002). Pipeline "
            "fixo e determinístico (ADR-003)."
        ),
        "detalhe": """
Ordem em `preprocessar(df)`:

1. Gênero válido (`M` / `F`)
2. Imputar FC (mediana)
3. Corridas válidas (distância, pace, FC)
4. Pace em min/km
5. Tipo de treino relativo ao atleta
        """,
        "tela": "pages/2_Visualizacao_Dataset.py",
        "adr": "ADR-001…004",
    },
    {
        "id": "tipo",
        "nome": "3 · Tipo de treino",
        "resumo": "longão · intervalado · leve · ritmo (relativo ao atleta).",
        "por_que": (
            "“Longo” para quem corre 5 km ≠ para quem corre 20 km. "
            "Percentis/desvios por atleta (ADR-004)."
        ),
        "detalhe": """
Prioridade:

1. **Longão** — distância ≥ p75 do atleta
2. **Intervalado** — pace bem abaixo da média (+ FC se houver)
3. **Leve** — pace bem acima da média
4. **Ritmo** — perto da média
5. Fallback → `leve`

Efeito: ~25% longão *por construção* → meta global 60% leve ficou inalcançável.
        """,
        "tela": "pages/2_Visualizacao_Dataset.py",
        "adr": "ADR-004",
    },
    {
        "id": "categ",
        "nome": "4 · Categorização",
        "resumo": "Regras (iniciante…avançado) + KMeans por gênero → perfil_cluster.",
        "por_que": (
            "Cluster encontra estrutura; regras explicam em português. "
            "KMeans separado por gênero (ADR-005/006)."
        ),
        "detalhe": """
- `perfil_atleta`: iniciante | intermediario | fundista | avancado
- `perfil_cluster`: **volume_alto** | **volume_moderado** (fallback)
- Features KMeans: distância + pace; `StandardScaler`; k por gênero
        """,
        "tela": "pages/4_Categorizacao_Corredores.py",
        "adr": "ADR-005/006",
    },
    {
        "id": "modelos",
        "nome": "5 · Modelos auxiliares",
        "resumo": "Pace/tempo (LinReg, RF, XGBoost) — não são o produto.",
        "por_que": (
            "Parte do TCC. Avaliação correta = GroupKFold por atleta (ADR-017). "
            "A tela Comparação mantém o protocolo antigo de propósito."
        ),
        "detalhe": """
Contrato único: `ResultadoModelo` (ADR-008).

Números defensáveis: tela **Validação da recomendação** / CLI `dataset.experimentos`.
        """,
        "tela": "pages/3_Comparacao_Modelos.py",
        "adr": "ADR-008/017",
    },
    {
        "id": "rec",
        "nome": "6 · Recomendação",
        "resumo": "A) Déficit vs meta do cluster  ·  B) Gemini (mesmo JSON).",
        "por_que": (
            "Meta por cluster alcançável (ADR-024/026). Regras explicáveis; "
            "Gemini só redige (ADR-010)."
        ),
        "detalhe": """
**Produto:** `META_POR_PERFIL` + déficit ponderado + anti-fadiga.

| Perfil | leve | longão | ritmo | intervalado |
|--------|------|--------|-------|-------------|
| volume_alto | 45% | 28% | 17% | 10% |
| volume_moderado | 42% | 22% | 22% | 14% |

Detalhe da tomada de decisão → aba **Tomada de decisão**.
        """,
        "tela": "pages/6_Recomendacao_Treino.py",
        "adr": "ADR-010/024/026",
    },
    {
        "id": "val",
        "nome": "7 · Validação",
        "resumo": "Eficácia (21 km) = veredito · F1 baixo ≠ nota do produto.",
        "por_que": (
            "Imitar hábito Strava não prova conselho bom. Desfecho externo "
            "mede utilidade (ADR-018/026)."
        ),
        "detalhe": """
1. **Eficácia por desfecho** — aderência à meta × tempo de meia (veredito)
2. **Imitação** — “por que F1 baixo não importa”
3. Ablação da regra + modelos auxiliares
        """,
        "tela": "pages/8_Validacao_da_Recomendacao.py",
        "adr": "ADR-017…026",
    },
]

PASSOS_DECISAO = [
    {
        "n": 1,
        "titulo": "Identificar o perfil_cluster",
        "texto": "`volume_alto` ou `volume_moderado` (fallback se não houver cluster).",
    },
    {
        "n": 2,
        "titulo": "Carregar a meta do perfil",
        "texto": "Tabela `META_POR_PERFIL` — metas **dentro** da região factível (~42–45% leve).",
    },
    {
        "n": 3,
        "titulo": "Distribuição atual (6 semanas)",
        "texto": "Proporção de leve / longão / ritmo / intervalado na janela `SEMANAS_JANELA`.",
    },
    {
        "n": 4,
        "titulo": "Calcular déficits",
        "texto": "`déficit = meta − atual` por tipo. Quem está mais abaixo da meta sobe na fila.",
    },
    {
        "n": 5,
        "titulo": "Déficit ponderado",
        "texto": "Pesos: intervalado 1,2 · longão 1,1 · resto 1,0. Escolhe o maior déficit ponderado.",
    },
    {
        "n": 6,
        "titulo": "Anti-fadiga",
        "texto": (
            "≥ 2 duros nos últimos 3 → `leve`. Último intervalado e pediria duro → `leve`. "
            "Último longão e pediria longão → `leve`."
        ),
    },
    {
        "n": 7,
        "titulo": "Distância alvo",
        "texto": "55% mediana do cluster + 45% mediana do atleta no tipo; longão sobe no máx. ~12%.",
    },
    {
        "n": 8,
        "titulo": "Pace alvo",
        "texto": "Pace médio do atleta + offset do tipo (leve mais lento, intervalado mais rápido).",
    },
    {
        "n": 9,
        "titulo": "Tempo estimado",
        "texto": "`distância × pace`.",
    },
    {
        "n": 10,
        "titulo": "Texto humano (+ Gemini opcional)",
        "texto": "`explicar_recomendacao(...)`. Gemini recebe o **mesmo** JSON e só redige.",
    },
]

DECISOES = [
    ("Polars + `constants.py`", "Performance e uma fonte de verdade para colunas/limiares"),
    ("Tipo de treino relativo", "“Longo/rápido” só faz sentido no histórico da pessoa"),
    ("KMeans por gênero + regras", "Cluster encontra estrutura; regras explicam"),
    ("Meta por cluster (não 60% leve)", "Meta global era inalcançável e viciava em `leve`"),
    ("Déficit ponderado + anti-fadiga", "Corrige desequilíbrio sem empilhar carga dura"),
    ("Distância misturada cluster×atleta", "Plano do grupo, dose individual"),
    ("Gemini só no texto", "Features confiáveis; justificativa rica"),
    ("GroupKFold por atleta", "Caso de uso real = atleta novo"),
    ("Segunda base com 21 km", "Validar utilidade, não só imitação"),
    ("Streamlit + notebooks", "Demo para banca; notebooks para exploração"),
    ("Sem API ainda", "Foco científico neste repo; HTTP no SPEC (ADR-009)"),
]


def _caixa(titulo: str, detalhe: str = "", *, destaque: str | None = None) -> None:
    """Bloco visual de uma etapa (substitui Mermaid — sem erro de sintaxe)."""
    border = {
        "produto": "#2e7d32",
        "validacao": "#1565c0",
        "decisao": "#ef6c00",
        None: "#90a4ae",
    }[destaque]
    fundo = {
        "produto": "#e8f5e9",
        "validacao": "#e3f2fd",
        "decisao": "#fff3e0",
        None: "#fafafa",
    }[destaque]
    detalhe_html = (
        f'<div style="font-size:0.85rem;opacity:0.85;margin-top:0.25rem;">{detalhe}</div>'
        if detalhe
        else ""
    )
    st.markdown(
        f"""
        <div style="
            border: 2px solid {border};
            background: {fundo};
            border-radius: 10px;
            padding: 0.75rem 1rem;
            margin: 0.15rem 0;
            text-align: center;
        ">
            <div style="font-weight:600;">{titulo}</div>
            {detalhe_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _seta(texto: str = "↓") -> None:
    st.markdown(
        f"<div style='text-align:center;font-size:1.4rem;line-height:1.2;"
        f"color:#546e7a;margin:0.1rem 0;'>{texto}</div>",
        unsafe_allow_html=True,
    )


def _pipeline_repo() -> None:
    _caixa("1 · Coleta", "CSV Kaggle (snapshot Strava)")
    _seta()
    _caixa("2 · Limpeza + engenharia", "gênero → FC → pace → tipo_treino")
    _seta()
    _caixa("3 · Categorização", "regras + KMeans → perfil_cluster")
    _seta()
    c1, c2 = st.columns(2)
    with c1:
        _caixa("4 · Modelos auxiliares", "pace / tempo (não é o produto)")
    with c2:
        _caixa(
            "5 · Recomendação (produto)",
            "déficit vs meta do cluster",
            destaque="produto",
        )
        _seta()
        _caixa("5b · Gemini (opcional)", "mesmo JSON, só redige o texto")
    _seta()
    _caixa(
        "6 · Validação",
        "eficácia 21 km = veredito · F1 baixo ≠ nota",
        destaque="validacao",
    )


def _fluxo_decisao() -> None:
    passos = [
        ("perfil_cluster", None),
        ("META_POR_PERFIL", None),
        ("Distribuição nas 6 semanas", None),
        ("Déficit = meta − atual", None),
        ("Déficit ponderado", None),
    ]
    for titulo, _ in passos:
        _caixa(titulo)
        _seta()

    _caixa("Anti-fadiga?", "2 duros recentes / repetição de carga", destaque="decisao")
    e1, e2 = st.columns(2)
    with e1:
        st.caption("sim →")
        _caixa("Força leve", destaque="produto")
    with e2:
        st.caption("não →")
        _caixa("Tipo escolhido")
    _seta()
    for titulo in (
        "Distância (cluster × atleta)",
        "Pace + offset do tipo",
        "Tempo = dist × pace",
        "explicar_recomendacao",
        "Gemini opcional",
    ):
        _caixa(titulo)
        if titulo != "Gemini opcional":
            _seta()


aba_mapa, aba_etapas, aba_decisao, aba_por_que = st.tabs(
    [
        "Mapa do fluxo",
        "Etapas detalhadas",
        "Tomada de decisão",
        "Por que cada escolha",
    ]
)


with aba_mapa:
    st.subheader("Pipeline neste repo")
    st.markdown(
        "Hoje a demo roda **só** aqui (CSV + Streamlit). A API HTTP ainda é proposta."
    )
    _pipeline_repo()


with aba_etapas:
    st.subheader("Percorra cada etapa")
    nomes = [e["nome"] for e in ETAPAS]
    escolha = st.radio("Etapa", nomes, horizontal=True, label_visibility="collapsed")
    etapa = next(e for e in ETAPAS if e["nome"] == escolha)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"### {etapa['nome']}")
        st.info(etapa["resumo"])
        st.markdown(etapa["detalhe"])
    with c2:
        st.markdown("**Por quê?**")
        st.success(etapa["por_que"])
        st.caption(f"Referência: {etapa['adr']}")
        if etapa["tela"]:
            st.page_link(etapa["tela"], label="Abrir tela relacionada", icon="🔗")

    st.divider()
    st.markdown("**Visão compacta de todas as etapas**")
    for e in ETAPAS:
        with st.expander(e["nome"]):
            st.write(e["resumo"])
            st.caption(e["por_que"])


with aba_decisao:
    st.subheader("Como a regra escolhe o próximo treino")
    st.markdown(
        """
        Caminho principal do produto (`recomendacao_cluster.py`).  
        Cada passo alimenta o seguinte — o tipo só muda de verdade no **anti-fadiga**.
        """
    )

    _fluxo_decisao()

    passo = st.slider("Passo da decisão", 1, 10, 1)
    atual = PASSOS_DECISAO[passo - 1]
    st.markdown(f"### Passo {atual['n']} — {atual['titulo']}")
    st.write(atual["texto"])

    with st.expander("Ver os 10 passos de uma vez"):
        for p in PASSOS_DECISAO:
            st.markdown(f"**{p['n']}. {p['titulo']}** — {p['texto']}")

    st.page_link(
        "pages/6_Recomendacao_Treino.py",
        label="Ver a recomendação ao vivo para um atleta",
        icon="🏃",
    )


with aba_por_que:
    st.subheader("Grandes escolhas do TCC")
    st.markdown("Resumo da seção 9 de `FLUXO.md` — o *porquê* por trás do *como*.")
    st.dataframe(
        [{"Escolha": a, "Por quê": b} for a, b in DECISOES],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Detalhe formal: `ADR.md`. Conclusões de banca: `GUIA-EXPERIMENTOS.md`. "
        "Texto narrativo completo: `FLUXO.md`."
    )
