import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import streamlit as st

from app.shared import (
    desfecho_meia,
    fonte_externa_disponivel,
    meta_aprendida,
    validacao_encaixada,
)

st.set_page_config(page_title="Desfecho e Meta", page_icon="🎯", layout="wide")

st.title("🎯 Desfecho de Prova e Meta Aprendida")
st.markdown(
    """
    Até aqui a recomendação de treino só podia ser avaliada contra a própria
    premissa — o erro em relação à `META` —, o que é circular. Esta tela usa uma
    **variável de desfecho** vinda da base externa (ADR-018): o melhor tempo de
    **meia maratona** de cada atleta.

    21 km e não maratona porque o público do projeto é o corredor **amador**, e
    essa é a distância de prova dominante nesse público.
    """
)

with st.expander("📖 O que esta tela prova, em 4 linhas"):
    st.markdown(
        """
        1. O treino **prediz** o tempo de prova — explica ~62% da variação.
        2. A **composição** do treino (quanto de leve/longão/ritmo/intervalado)
           acrescenta informação além de volume e pace. Ou seja, **a premissa da
           regra por déficit vs meta global está certa**.
        3. Mas a meta pede **60% de treino leve**, e nenhum atleta chega lá —
           o percentil 95 é 48%. A meta é **inalcançável por construção**.
        4. Tentei aprender a meta correta dos dados e **não consegui com
           precisão suficiente** para substituir os números. O que fica provado é
           o defeito e a direção da correção, não uma meta nova.

        Explicação completa dos termos (modelos encaixados, polinômio de
        Scheffé, região factível) em **`GUIA-EXPERIMENTOS.md`** na raiz.
        """
    )

disponivel, diretorio = fonte_externa_disponivel()
if not disponivel:
    st.error(
        f"""
        **Base externa não encontrada** em `{diretorio}`.

        São 2,3 GB e não são versionadas. Para baixar (o endpoint público do
        Kaggle dispensa credenciais):

        ```bash
        bash scripts/baixar_long_distance.sh
        ```

        Ou aponte `DIRETORIO_DADOS_EXTERNOS` para onde os arquivos já estão.
        """
    )
    st.stop()

max_atletas = st.sidebar.select_slider(
    "Atletas carregados da base externa",
    options=[3000, 6000, 12000, 24000, 36412],
    value=12000,
    help="Mais atletas = mais poder estatístico e mais tempo na 1ª execução.",
)

aba_desfecho, aba_composicao, aba_meta = st.tabs(
    ["3 · Desfecho", "3 · A composição importa?", "4 · Meta aprendida"]
)


with aba_desfecho:
    st.header("O desfecho: melhor meia maratona")

    with st.spinner("Extraindo desfecho da base externa (cacheado)…"):
        dados = desfecho_meia(max_atletas)

    tabela = dados["tabela"]
    if tabela.height == 0:
        st.warning("Nenhum atleta com desfecho. Aumente o número de atletas.")
        st.stop()

    fonte = dados["fonte"]
    colunas = st.columns(4)
    colunas[0].metric("Atletas com desfecho", f"{tabela.height:,}".replace(",", "."))
    colunas[1].metric("Corridas válidas", f"{dados['corridas_validas']:,}".replace(",", "."))
    colunas[2].metric("Tempo mediano", f"{dados['resumo']['tempo_mediano_min']:.0f} min")
    colunas[3].metric(
        "Volume semanal mediano",
        f"{dados['resumo']['volume_semanal_mediano_km']:.0f} km",
    )

    st.caption(
        f"Fonte `{fonte['fonte']}` · {fonte['atletas']:,} atletas carregados · "
        f"sem FC e sem elevação (o pipeline detecta e adapta, ADR-018)".replace(",", ".")
    )

    resumo = dados["resumo"]
    st.markdown(
        f"""
        Distribuição dos tempos: **p10 = {resumo['tempo_p10_min']:.0f} min**,
        **mediana = {resumo['tempo_mediano_min']:.0f} min**,
        **p90 = {resumo['tempo_p90_min']:.0f} min**. Faixa amadora plausível, que
        é o teste de sanidade do desfecho.
        """
    )

    fig, eixos = plt.subplots(1, 2, figsize=(12, 4))
    eixos[0].hist(tabela["tempo_prova_min"].to_numpy(), bins=50, color="#2196F3")
    eixos[0].set_xlabel("Tempo de meia maratona (min)")
    eixos[0].set_ylabel("Atletas")
    eixos[0].set_title("Desfecho")

    eixos[1].scatter(
        tabela["volume_semanal_km"].to_numpy(),
        tabela["tempo_prova_min"].to_numpy(),
        s=6, alpha=0.25, color="#4CAF50",
    )
    eixos[1].set_xlabel("Volume semanal no preparo (km)")
    eixos[1].set_ylabel("Tempo de prova (min)")
    eixos[1].set_title("Treino × desempenho")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "**Limitação registrada.** O melhor esforço de 21 km pode ter sido um "
        "treino intenso, não uma prova — é proxy de capacidade, não resultado "
        "oficial. A checagem de robustez usa maratonas verificadas pela coluna "
        "`major` cruzada com a data oficial da prova (ADR-018)."
    )

    with st.expander("Amostra da tabela de desfecho + preparo"):
        st.dataframe(tabela.head(50), width="stretch")


with aba_composicao:
    st.header("A composição de treino acrescenta algo além de volume e pace?")
    st.markdown(
        """
        Esta é a premissa da regra por déficit vs meta global. Se a
        **distribuição** entre leve/longão/ritmo/intervalado não melhora a
        predição depois de controlar *quanto* e *quão rápido* o atleta treina,
        então recomendar "equilibre sua distribuição" não tem suporte nos dados.

        Comparação de modelos **encaixados**: B contém todas as features de A
        mais as de interesse, então a diferença isola a contribuição.
        """
    )

    with st.spinner("Ajustando modelos encaixados…"):
        validacao = validacao_encaixada(max_atletas)

    if not validacao:
        st.warning("Sem dados suficientes.")
        st.stop()

    st.subheader("Correlação de cada feature com o tempo de prova")
    correlacoes = validacao["correlacoes"]
    st.dataframe(correlacoes, width="stretch")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    nomes = correlacoes["feature"].to_list()
    valores = correlacoes["correlacao"].to_list()
    erros = [
        [v - i for v, i in zip(valores, correlacoes["ic_inferior"].to_list())],
        [s - v for v, s in zip(valores, correlacoes["ic_superior"].to_list())],
    ]
    cores = ["#F44336" if v > 0 else "#4CAF50" for v in valores]
    ax.barh(nomes, valores, xerr=erros, color=cores, capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlação com o tempo de prova (negativo = mais rápido)")
    ax.set_title("Verde: associado a prova mais rápida")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        "`prop_leve` negativo e `prop_intervalado` positivo: **mais corrida leve "
        "associada a prova mais rápida** — a direção que a `META` supõe, e "
        "consistente com a literatura de treino polarizado."
    )

    for rotulo, chave, chave_comp in (
        ("Regressão Linear", "linear", "comp_linear"),
        ("Random Forest", "floresta", "comp_floresta"),
    ):
        st.subheader(f"Modelos encaixados — {rotulo}")
        st.dataframe(validacao[chave], width="stretch")
        st.dataframe(validacao[chave_comp], width="stretch")

    st.success(
        "A conclusão se mantém com **Random Forest**, então não é artefato de "
        "linearidade: a composição de treino acrescenta informação real, embora "
        "o ganho seja pequeno (ΔR² da ordem de 0,02)."
    )

    st.warning(
        "**Natureza observacional.** Associação, não causalidade — atletas mais "
        "rápidos podem treinar diferente *porque* são mais rápidos. O que os "
        "dados estabelecem é o mais fraco e ainda útil: se nem a associação "
        "existisse, a recomendação não teria base."
    )


with aba_meta:
    st.header("A meta global de distribuição, medida em vez de afirmada")

    with st.spinner("Estimando a meta ótima (otimização no simplexo)…"):
        meta = meta_aprendida(max_atletas)

    if not meta:
        st.warning("Sem dados suficientes.")
        st.stop()

    st.caption(f"{meta['atletas']:,} atletas com desfecho".replace(",", "."))

    st.subheader("A META atual é alcançável?")
    regiao = meta["regiao"]
    st.dataframe(regiao, width="stretch")

    inalcancaveis = regiao.filter(~pl.col("dentro_da_regiao"))
    if inalcancaveis.height:
        linhas = "\n".join(
            f"- **{r['tipo']}**: a meta pede {r['meta_global']:.0%}, mas o "
            f"observado vai de {r['observado_p5']:.1%} a {r['observado_p95']:.1%} "
            f"(mediana {r['observado_mediana']:.1%})"
            for r in inalcancaveis.iter_rows(named=True)
        )
        st.error(
            f"""
            **Defeito estrutural da meta global.** Estes componentes estão
            **fora** do que o classificador de tipo de treino (relativo ao
            atleta) consegue produzir:

{linhas}

            A causa: `longao` é definido pelo percentil 75 de distância *do
            próprio atleta*, o que fixa ~25% das corridas nessa classe por
            construção. Logo nenhum atleta atinge 60% de `leve`, o déficit de
            `leve` fica positivo para todos, e a regra recomenda `leve` quase
            sempre — é por isso que ela perde de `sempre_leve` na tela
            **Protocolo Experimental**.
            """
        )

    fig, ax = plt.subplots(figsize=(9, 4))
    tipos = regiao["tipo"].to_list()
    posicoes = np.arange(len(tipos))
    ax.barh(
        posicoes,
        [s - i for i, s in zip(regiao["observado_p5"], regiao["observado_p95"])],
        left=regiao["observado_p5"].to_list(),
        color="#B0BEC5", label="região observada (p5–p95)",
    )
    ax.scatter(
        regiao["meta_global"].to_list(), posicoes,
        color="#F44336", zorder=3, s=90, label="meta global (60/20/12/8)",
    )
    ax.scatter(
        meta["factivel"]["meta_aprendida"].to_list(), posicoes,
        color="#2196F3", zorder=3, s=90, marker="D", label="meta aprendida (factível)",
    )
    ax.set_yticks(posicoes)
    ax.set_yticklabels(tipos)
    ax.set_xlabel("Proporção do treino")
    ax.set_title("A META cai dentro do que o classificador produz?")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader(f"Meta aprendida — {meta['factivel_nome']}")
    st.caption(
        f"Tempo de 21 km previsto na meta: {meta['factivel_tempo']:.1f} min · "
        "volume e pace fixados na mediana (senão a resposta seria "
        "'treine mais e mais rápido', verdadeira e inútil)"
    )
    st.dataframe(meta["factivel"], width="stretch")

    with st.expander("Variante irrestrita (só diagnóstico)"):
        st.markdown(
            """
            Ótimo no simplexo sem restringir ao observável. Serve para mostrar
            para onde os dados apontam, **não** como meta utilizável: um atleta
            precisa poder alcançar a meta.

            O problema foi modelado como **experimento de mistura** (as 4
            proporções somam 1), com o polinômio quadrático de **Scheffé** —
            termos lineares mais os 6 produtos cruzados. Um modelo linear nas
            proporções tem mínimo obrigatoriamente num *vértice* do simplexo,
            isto é, só consegue responder "faça 100% de um tipo".
            """
        )
        st.dataframe(meta["irrestrita"], width="stretch")
        st.caption(f"Tempo previsto: {meta['irrestrita_tempo']:.1f} min")

    st.subheader("Validação fora da amostra")
    st.markdown(
        """
        Em cada fold a meta é aprendida **só no treino** e a aderência a ela é
        medida no teste. Sem isso a meta aprendida venceria por construção.
        """
    )
    st.dataframe(meta["validacao"], width="stretch")

    if meta["factivel_instaveis"]:
        st.error(
            f"""
            **Por que os valores aprendidos NÃO substituíram a `META`.**

            Componentes mal identificados: {', '.join(f'`{t}`' for t in meta['factivel_instaveis'])}.
            O IC bootstrap deles é mais largo que a própria diferença em relação
            à meta global — reamostrar a base move o ótimo mais do que a suposta
            melhoria. Compare também as metas na tabela acima com a coluna
            `corr_desvio`: se a diferença entre elas for menor que o desvio, elas
            **empatam**.

            Conclusão: a **direção** da meta global está certa (mais leve, menos
            intervalado); o erro é o **nível** de `leve`, fixado num valor que o
            classificador não pode produzir. Trocar 60/20/12/8 por outros quatro
            números seria substituir um chute por outro com precisão falsa. A
            meta em `model.py` segue inalterada.
            """
        )
