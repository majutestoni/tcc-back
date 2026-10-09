import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import streamlit as st

from app.shared import ablacao_adr007, avaliacao_protocolo, comparacao_baselines
from dataset.experimentos.baselines import (
    NOME_REGRA_META_CLUSTER,
    NOME_REGRA_META_GLOBAL,
)

st.set_page_config(page_title="Protocolo Experimental", page_icon="🔬", layout="wide")

st.title("🔬 Protocolo Experimental")
st.markdown(
    """
    **Foco desta tela:** validar a **recomendação de treino** — a regra por
    **déficit vs meta global** e a por **déficit vs meta do cluster** — contra
    baselines triviais.

    Os modelos de pace/tempo ficam na última aba (são auxiliares).
    """
)

with st.spinner("Avaliando recomendadores (cacheado após a 1ª vez)…"):
    baselines = comparacao_baselines()

veredito = baselines["veredito"]
pct_leve = veredito["pct_leve_global"] * 100
derrotas = veredito["derrotas"]
nome_global = baselines.get("nome_regra_global", NOME_REGRA_META_GLOBAL)

st.success(
    f"""
**Conclusão em 30 segundos**

1. A regra **{nome_global}** recomenda **leve em {pct_leve:.0f}%** dos casos
   (F1-macro **{veredito["f1_global"] * 100:.1f}%**, acerto
   **{veredito["acerto_global"] * 100:.1f}%**).
2. O melhor no ranking é **`{veredito["melhor_nome"]}`**
   (F1 **{veredito["melhor_f1"] * 100:.1f}%**) — a regra por meta global
   {"**perde** de: " + ", ".join(f"`{d}`" for d in derrotas) if derrotas else "não perde com significância dos listados"}.
3. Isso mede **imitação** (acertou o que o atleta fez), não se o conselho é bom.
   Qualidade do conselho → tela **Desfecho e Meta**.
    """
)

with st.expander("📖 Termos rápidos (bootstrap, Wilcoxon, F1…)"):
    st.markdown(
        """
        | Termo | Em uma frase |
        |---|---|
        | **Origem rolante** | Para cada treino recente, o sistema só vê o histórico anterior e tenta acertar o tipo daquele treino. |
        | **F1-macro** | Média do acerto em **cada** um dos 4 tipos. Quem sempre fala "leve" tem acerto ok e F1 baixo. |
        | **`rec_leve_%`** | % das recomendações que foram "leve". Se ~80%, a regra virou constante. |
        | **`p_holm`** | Chance de o resultado ser sorte, já corrigida por várias comparações. &lt; 0,05 = diferença real. |
        | **GroupKFold** | (aba de modelos) atleta nunca aparece em treino e teste ao mesmo tempo. |

        Detalhes em `GUIA-EXPERIMENTOS.md`.
        """
    )

aba_rec, aba_ablacao, aba_modelos = st.tabs(
    [
        "1 · Validação da recomendação",
        "2 · Ablação da regra",
        "3 · Modelos auxiliares (pace/tempo)",
    ]
)


with aba_rec:
    st.header("A regra supera o trivial?")
    st.markdown(
        f"""
        Protocolo: **origem rolante** — {baselines["casos"]} casos em
        {baselines["atletas"]} atletas. O recomendador vê só o passado e tenta
        acertar o tipo da próxima corrida que o atleta **de fato fez**.
        """
    )
    st.info(
        "**O que medimos:** imitação. **O que não medimos:** se o conselho é "
        "bom para a prova (isso está em **Desfecho e Meta**)."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Acerto (meta global)", f"{veredito['acerto_global'] * 100:.1f}%")
    m2.metric("F1-macro (meta global)", f"{veredito['f1_global'] * 100:.1f}%")
    m3.metric("% recomenda leve", f"{pct_leve:.0f}%")
    if veredito["f1_cluster"] is not None:
        m4.metric(
            "F1 (meta do cluster)",
            f"{veredito['f1_cluster'] * 100:.1f}%",
            delta=f"{(veredito['f1_cluster'] - veredito['f1_global']) * 100:.1f} pp",
        )

    st.subheader("Ranking: regras do projeto vs baselines")
    st.caption(
        "Valores em **%**. Destaque vermelho = regras do TCC "
        f"(`{NOME_REGRA_META_GLOBAL}` e `{NOME_REGRA_META_CLUSTER}`). "
        "Olhe sobretudo **f1_macro_%** e **rec_leve_%**."
    )
    tabela = baselines["tabela"]
    st.dataframe(tabela, width="stretch")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    nomes = tabela["recomendador"].to_list()
    col_f1 = "f1_macro_%" if "f1_macro_%" in tabela.columns else "f1_macro"
    cores = [
        "#F44336"
        if n in (NOME_REGRA_META_GLOBAL, NOME_REGRA_META_CLUSTER)
        else "#90A4AE"
        for n in nomes
    ]
    ax.barh(nomes, tabela[col_f1].to_list(), color=cores)
    ax.set_xlabel("F1-macro (%)")
    ax.set_title("Regras do projeto (vermelho) vs baselines")
    ax.invert_yaxis()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader(f"Confrontos diretos com `{nome_global}`")
    confrontos = baselines["confrontos"]
    st.dataframe(confrontos, width="stretch")
    if derrotas:
        st.error(
            f"A regra **{nome_global}** **perde** com significância estatística para: "
            + ", ".join(f"`{r}`" for r in derrotas)
            + ". Causa raiz: meta de 60% leve inalcançável — ver **Desfecho e Meta**."
        )
    else:
        st.warning("Nenhuma derrota significativa listada nos confrontos filtrados.")

    st.subheader("Matriz de confusão — déficit vs meta global")
    st.caption("Linhas = o que o atleta fez · Colunas = o que o sistema recomendou.")
    st.dataframe(baselines["matriz_global"], width="stretch")

    if baselines["matriz_cluster"].height:
        with st.expander("Matriz de confusão — déficit vs meta do cluster"):
            st.dataframe(baselines["matriz_cluster"], width="stretch")
            if veredito["pct_leve_cluster"] is not None:
                st.caption(
                    f"Meta do cluster recomenda leve em "
                    f"**{veredito['pct_leve_cluster'] * 100:.0f}%** dos casos "
                    f"(acerto {veredito['acerto_cluster'] * 100:.1f}%, "
                    f"F1 {veredito['f1_cluster'] * 100:.1f}%)."
                )

    st.subheader("Amostra de casos — déficit vs meta global")
    st.caption(
        "Cada linha: últimos 5 tipos do histórico → o que aconteceu (`real`) "
        "vs o que a regra pediu (`predito`)."
    )
    st.dataframe(baselines["amostra_global"], width="stretch")

    if baselines["amostra_cluster"].height:
        with st.expander("Amostra de casos — déficit vs meta do cluster"):
            st.dataframe(baselines["amostra_cluster"], width="stretch")


with aba_ablacao:
    st.header("Quais partes da regra realmente importam?")
    st.markdown(
        """
        Removemos um componente por vez. Se a métrica **não muda**, o componente
        é complexidade sem retorno. Se **melhora** ao remover, o componente atrapalha.
        """
    )

    with st.spinner("Rodando ablação…"):
        ablacao = ablacao_adr007()

    cols_pct = [c for c in ("f1_macro", "delta_f1_vs_completa") if c in ablacao.columns]
    ablacao_exibir = ablacao.with_columns(
        [(pl.col(c) * 100).round(2).alias(f"{c}_%") for c in cols_pct]
    )
    st.caption("Colunas `*_ %` em percentual.")
    st.dataframe(ablacao_exibir, width="stretch")
    st.markdown(
        "**`delta_f1_vs_completa_%` > 0** = remover o componente _melhorou_ o resultado."
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    variantes = ablacao["variante"].to_list()
    deltas = (ablacao["delta_f1_vs_completa"] * 100).to_list()
    cores = ["#4CAF50" if d > 0 else "#F44336" if d < 0 else "#90A4AE" for d in deltas]
    ax.barh(variantes, deltas, color=cores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ F1-macro vs configuração completa (pontos percentuais)")
    ax.set_title("Verde = remover o componente melhorou")
    ax.invert_yaxis()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    inertes = ablacao.filter(pl.col("delta_f1_vs_completa").abs() < 0.002)
    if inertes.height:
        st.warning(
            "Componentes **inertes** (efeito &lt; 0,2 pp de F1-macro): "
            + ", ".join(
                f"`{r}`" for r in inertes["componente_removido"].to_list() if r != "—"
            )
        )


with aba_modelos:
    st.header("Modelos auxiliares — particionamento por atleta")
    st.markdown(
        """
        Aqui estão pace/tempo/classificadores sob o protocolo correto
        (`GroupKFold` por atleta). **Não é a validação da recomendação** —
        serve para mostrar que as métricas antigas (tela Comparação de Modelos)
        eram otimistas.
        """
    )

    with st.spinner("Avaliando 8 modelos com CV agrupada (cacheado após a 1ª vez)…"):
        protocolo = avaliacao_protocolo()

    st.caption(f"Protocolo: {protocolo['protocolo']}")
    st.dataframe(protocolo["comparativa"], width="stretch")

    st.markdown(
        """
        **Como ler.** `valor` = métrica no protocolo correto (com IC).
        `protocolo_antigo` = split aleatório por corrida.
        **`otimismo` > 0** = o protocolo antigo inflava o número.
        """
    )

    comparativa = protocolo["comparativa"]
    otimismos = comparativa.filter(pl.col("otimismo").is_not_null())
    if otimismos.height:
        fig, ax = plt.subplots(figsize=(9, 4))
        cores = ["#F44336" if v > 0 else "#4CAF50" for v in otimismos["otimismo"]]
        ax.barh(
            otimismos["modelo"].to_list(),
            otimismos["otimismo"].to_list(),
            color=cores,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Otimismo do protocolo antigo")
        ax.set_title("Quanto o split por corrida inflava cada modelo")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Testes pareados (Wilcoxon + Holm)")
    pareados = protocolo["pareados"]
    if pareados.height:
        st.dataframe(pareados, width="stretch")
        empates = pareados.filter(~pl.col("significativa"))
        if empates.height:
            st.warning(
                "Pares **sem diferença estatística**:\n\n"
                + "\n".join(
                    f"- {r['modelo_a']} × {r['modelo_b']} "
                    f"(`p_holm = {r['p_holm']:.3g}`)"
                    for r in empates.iter_rows(named=True)
                )
            )

    if protocolo["ressalvas"]:
        st.subheader("Ressalvas metodológicas")
        for nome, aviso in protocolo["ressalvas"].items():
            st.error(f"**{nome}** — {aviso}")
