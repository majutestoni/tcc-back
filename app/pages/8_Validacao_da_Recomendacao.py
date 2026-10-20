import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import streamlit as st

from app.shared import (
    ablacao_adr007,
    avaliacao_protocolo,
    comparacao_baselines,
    eficacia_por_desfecho,
)
from dataset.experimentos.baselines import NOME_REGRA_META_CLUSTER
from dataset.models.previsao_treino.recomendacao_cluster import META_POR_PERFIL

st.set_page_config(
    page_title="Validação da recomendação", page_icon="🔬", layout="wide"
)

st.title("Validação da recomendação")
st.subheader("A regra funciona?")
st.markdown(
    """
    **Foco:** eficácia **prescritiva** — atletas mais alinhados à meta do cluster
    têm melhor tempo de meia maratona? (desfecho externo).

    A imitação do próximo treino no histórico Strava (~26% de acerto) **não** é
    o veredito do produto: o dataset não é um plano padronizado. Gerar o
    próximo treino de um atleta → tela **Recomendação de Treino**.
    """
)

st.info(
    """
**Déficit** — `meta − atual` por tipo na janela; maior déficit = tipo em falta.

**Baseline** — recomendador trivial só para o experimento de imitação
(sempre leve, repete o último…). **Não** define se o conselho é bom.

**Aderência** — quão perto a distribuição de treino fica da meta do cluster
(menor distância = mais aderente). Relacionamos isso ao **tempo de prova**.
    """
)

with st.spinner("Carregando evidência de desfecho (se a fonte externa existir)…"):
    eficacia = eficacia_por_desfecho()

resumo_ef = eficacia.get("resumo_padrao") or {}
corr = resumo_ef.get("corr_distancia_tempo")
n_ef = eficacia.get("n") or 0

if eficacia.get("disponivel") and n_ef > 0 and corr is not None:
    t_alta = resumo_ef.get("tempo_mediano_alta_aderencia_min")
    t_baixa = resumo_ef.get("tempo_mediano_baixa_aderencia_min")
    delta = (
        round(t_baixa - t_alta, 1)
        if t_alta is not None and t_baixa is not None
        else None
    )
    st.success(
        f"""
**Conclusão em 30 segundos (eficácia)**

1. Em **{n_ef}** atletas com desfecho de ~21 km, a correlação entre
   *distância à meta do cluster* e *tempo de prova* é **{corr:.3f}**
   ({resumo_ef.get("interpretacao", "—")}).
2. Tempo mediano: alta aderência **{t_alta} min** vs baixa aderência
   **{t_baixa} min**
   {f"(Δ ≈ **{delta} min** a favor de quem treina mais perto da meta)." if delta is not None else "."}
3. As metas de produto (`volume_alto` / `volume_moderado`) ficam na **região
   factível** do classificador — não pedem 60–70% de leve inalcançável.
4. O teste de **imitação** (aba 2) continua só como contraste negativo:
   acertar o hábito do Strava ≠ bom plano.
        """
    )
elif not eficacia.get("disponivel"):
    st.warning(
        f"""
**Desfecho externo ainda não baixado** (`{eficacia.get("diretorio")}`).

Rode `bash scripts/baixar_long_distance.sh` para habilitar a evidência de
eficácia (ADR-018). Enquanto isso, a aba de imitação mostra o experimento
antigo — **não** use o F1 ~26% como nota da recomendação.
        """
    )
else:
    st.warning("Tabela de desfecho vazia — não foi possível calcular aderência.")

with st.expander("📖 Termos rápidos"):
    st.markdown(
        """
        | Termo | Em uma frase |
        |---|---|
        | **Aderência** | Quão perto a mistura leve/longão/ritmo/intervalado fica da meta. |
        | **Desfecho** | Melhor tempo ~21 km do atleta (proxy de capacidade). |
        | **Déficit** | `meta − atual`; guia o *próximo* treino na demo. |
        | **Imitação** | Acertar o próximo tipo que o atleta já fez no histórico. |
        | **Baseline** | Chute trivial no teste de imitação. |
        | **F1-macro** | Métrica de imitação — não de qualidade do conselho. |

        Detalhes em `GUIA-EXPERIMENTOS.md`.
        """
    )

aba_ef, aba_imit, aba_ablacao, aba_modelos = st.tabs(
    [
        "1 · Eficácia (desfecho)",
        "2 · Imitação (contraste)",
        "3 · Ablação da regra",
        "4 · Modelos auxiliares (pace/tempo)",
    ]
)


with aba_ef:
    st.header("Aderência à meta do cluster × tempo de prova")
    st.markdown(
        """
        Fonte externa (long-distance). Menor `distancia_meta` = treino mais
        alinhado à meta. Correlação **positiva** com tempo = quem desvia mais
        da meta tende a prova **mais lenta**.
        """
    )
    st.markdown("**Metas de produto (alcançáveis)**")
    st.write(
        {
            chave: {t: f"{p * 100:.0f}%" for t, p in meta.items()}
            for chave, meta in META_POR_PERFIL.items()
        }
    )

    if not eficacia.get("disponivel"):
        st.info("Baixe a fonte externa para ver gráficos e tabelas desta aba.")
    elif n_ef == 0:
        st.info("Sem linhas de desfecho.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Atletas (21 km)", n_ef)
        m2.metric(
            "Corr(distância→tempo)",
            f"{corr:.3f}" if corr is not None else "—",
        )
        m3.metric(
            "Tempo · alta aderência",
            f"{resumo_ef.get('tempo_mediano_alta_aderencia_min')} min",
        )
        m4.metric(
            "Tempo · baixa aderência",
            f"{resumo_ef.get('tempo_mediano_baixa_aderencia_min')} min",
        )

        st.subheader("Comparação de metas no desfecho")
        st.caption(
            "Maior correlação = aderência a essa meta prediz melhor o tempo."
        )
        st.dataframe(eficacia["comparacao"], width="stretch", hide_index=True)

        st.subheader("Metas dentro da região observada (p5–p95)?")
        regiao = eficacia["regiao"]
        st.dataframe(regiao, width="stretch", hide_index=True)
        fora = regiao.filter(~pl.col("dentro_da_regiao")) if regiao.height else regiao
        if fora.height:
            st.error(
                "Componentes fora da região: "
                + ", ".join(
                    f"{r['perfil']}/{r['tipo']}={r['meta']}"
                    for r in fora.iter_rows(named=True)
                )
            )
        else:
            st.success("Todas as metas de cluster estão dentro do p5–p95 observado.")


with aba_imit:
    st.header("Imitação — contraste negativo")
    st.warning(
        """
        Este protocolo pergunta: “acertamos o próximo tipo que o atleta **já
        fez**?”. No Strava sem plano, isso mede hábito, não qualidade do
        conselho. Baselines que **copiam** o histórico ganham — e isso é
        esperado. **Não** use F1/acerto daqui como nota do produto.
        """
    )

    with st.spinner("Avaliando imitação (cacheado após a 1ª vez)…"):
        baselines = comparacao_baselines()

    veredito = baselines["veredito"]
    pct_leve = veredito["pct_leve_cluster"] * 100
    derrotas = veredito["derrotas"]
    nome_regra = baselines.get("nome_regra_cluster", NOME_REGRA_META_CLUSTER)

    st.markdown(
        f"""
        Regra **{nome_regra}**: F1-macro
        **{veredito["f1_cluster"] * 100:.1f}%**, acerto
        **{veredito["acerto_cluster"] * 100:.1f}%**, recomenda leve em
        **{pct_leve:.0f}%**. Melhor no ranking de imitação:
        `{veredito["melhor_nome"]}` (F1 {veredito["melhor_f1"] * 100:.1f}%).
        """
    )

    st.subheader("Ranking (imitação)")
    tabela = baselines["tabela"]
    st.dataframe(tabela, width="stretch")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    nomes = tabela["recomendador"].to_list()
    col_f1 = "f1_macro_%" if "f1_macro_%" in tabela.columns else "f1_macro"
    cores = [
        "#F44336" if n == NOME_REGRA_META_CLUSTER else "#90A4AE" for n in nomes
    ]
    ax.barh(nomes, tabela[col_f1].to_list(), color=cores)
    ax.set_xlabel("F1-macro (%) — só imitação")
    ax.set_title("Vermelho = regra do projeto (não é o veredito de eficácia)")
    ax.invert_yaxis()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader(f"Confrontos com `{nome_regra}`")
    st.dataframe(baselines["confrontos"], width="stretch")
    if derrotas:
        st.caption(
            "Perder de baselines de cópia no teste de imitação é esperado e "
            "**não** invalida a regra prescritiva."
        )

    st.subheader("Matriz / amostra (cluster)")
    st.dataframe(baselines["matriz_cluster"], width="stretch")
    st.dataframe(baselines["amostra_cluster"], width="stretch")


with aba_ablacao:
    st.header("Quais partes da regra realmente importam?")
    st.markdown(
        """
        Removemos um componente por vez (meta de referência = `volume_moderado`
        alcançável). Se a métrica **não muda**, o componente é complexidade
        sem retorno. *Atenção:* a métrica aqui ainda é de **imitação**.
        """
    )

    with st.spinner("Rodando ablação…"):
        ablacao = ablacao_adr007()

    cols_pct = [c for c in ("f1_macro", "delta_f1_vs_completa") if c in ablacao.columns]
    ablacao_exibir = ablacao.with_columns(
        [(pl.col(c) * 100).round(2).alias(f"{c}_%") for c in cols_pct]
    )
    st.caption("Colunas `*_ %` em percentual (imitação).")
    st.dataframe(ablacao_exibir, width="stretch")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    variantes = ablacao["variante"].to_list()
    deltas = (ablacao["delta_f1_vs_completa"] * 100).to_list()
    cores = ["#4CAF50" if d > 0 else "#F44336" if d < 0 else "#90A4AE" for d in deltas]
    ax.barh(variantes, deltas, color=cores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ F1-macro vs completa (pp) — imitação")
    ax.set_title("Verde = remover o componente melhorou a imitação")
    ax.invert_yaxis()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


with aba_modelos:
    st.header("Modelos auxiliares — particionamento por atleta")
    st.markdown(
        """
        Pace/tempo sob `GroupKFold` por atleta. **Não** é a validação da
        recomendação — mostra otimismo do split antigo (tela Comparação).
        """
    )

    with st.spinner("Avaliando 8 modelos com CV agrupada (cacheado após a 1ª vez)…"):
        protocolo = avaliacao_protocolo()

    st.caption(f"Protocolo: {protocolo['protocolo']}")
    st.dataframe(protocolo["comparativa"], width="stretch")

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

    pareados = protocolo["pareados"]
    if pareados.height:
        st.subheader("Testes pareados (Wilcoxon + Holm)")
        st.dataframe(pareados, width="stretch")

    if protocolo["ressalvas"]:
        st.subheader("Ressalvas metodológicas")
        for nome, aviso in protocolo["ressalvas"].items():
            st.error(f"**{nome}** — {aviso}")
