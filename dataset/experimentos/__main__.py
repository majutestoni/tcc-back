"""
Executor dos experimentos.

    python -m dataset.experimentos protocolo
    python -m dataset.experimentos baselines
    python -m dataset.experimentos desfecho --max-atletas 8000
    python -m dataset.experimentos meta
    python -m dataset.experimentos tudo

Os blocos `desfecho` e `meta` exigem a fonte externa (ADR-018); rode
`bash scripts/baixar_long_distance.sh` antes.
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

BLOCOS = ("protocolo", "baselines", "desfecho", "meta", "tudo")


def _configurar_saida() -> None:
    pl.Config.set_tbl_rows(50)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(20)


def _titulo(texto: str) -> None:
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def _carregar_primaria() -> pl.DataFrame:
    from dataset.fontes import carregar_strava_kaggle
    from dataset.preprocessamento import preprocessar

    fonte = carregar_strava_kaggle()
    df = preprocessar(fonte.df)
    print(f"fonte primária: {fonte.resumo()} -> {df.height} corridas válidas")
    return df


def _carregar_externa(max_atletas: int | None) -> pl.DataFrame:
    from dataset.fontes import carregar_long_distance
    from dataset.preprocessamento import preprocessar

    fonte = carregar_long_distance(max_atletas=max_atletas)
    df = preprocessar(fonte.df)
    print(f"fonte externa: {fonte.resumo()} -> {df.height} corridas válidas")
    return df


def bloco_protocolo(args: argparse.Namespace) -> None:
    """ADR-017 item 1: protocolo experimental corrigido."""
    from dataset.avaliacao.comparacao import (
        avaliar_todos,
        comparar_por_categoria,
        tabela_comparacoes,
        tabela_comparativa,
    )

    df = _carregar_primaria()

    _titulo("ITEM 1 — MODELOS SOB PARTICIONAMENTO AGRUPADO POR ATLETA")
    resultados = avaliar_todos(df, verbose=not args.silencioso)
    print(tabela_comparativa(resultados))
    print(
        "\n`otimismo` > 0: o protocolo antigo (split aleatório por corrida) "
        "reportava desempenho melhor que o real."
    )

    _titulo("ITEM 1 — TESTES PAREADOS (Wilcoxon por atleta + Holm)")
    print(tabela_comparacoes(comparar_por_categoria(resultados)))

    avisos = {r.nome: r.observacao for r in resultados if r.observacao}
    if avisos:
        _titulo("RESSALVAS")
        for nome, aviso in avisos.items():
            print(f"\n{nome}:\n  {aviso}")


def bloco_baselines(args: argparse.Namespace) -> None:
    """ADR-017 item 2: baselines e ablação."""
    from dataset.avaliacao.estatistica import comparar_todos
    from dataset.experimentos.ablacao import rodar_ablacao
    from dataset.experimentos.baselines import (
        NOME_REGRA_META_CLUSTER,
        baselines_com_cluster,
    )
    from dataset.experimentos.protocolo_recomendacao import (
        avaliar_recomendador,
        montar_casos,
        tabela_recomendadores,
    )
    from dataset.constants import COL_ATHLETE
    from dataset.models.previsao_treino.recomendacao_cluster import montar_perfil_cluster

    df = _carregar_primaria()
    casos = montar_casos(df, n_teste=args.n_teste)
    print(f"casos de teste por origem rolante: {len(casos)}")

    perfil = montar_perfil_cluster(df)
    mapa_cluster = {
        int(row[COL_ATHLETE]): str(row["perfil_cluster"])
        for row in perfil.select(COL_ATHLETE, "perfil_cluster").iter_rows(named=True)
    }

    _titulo("ITEM 2 — BASELINES vs DÉFICIT VS META DO CLUSTER")
    resultados = [
        avaliar_recomendador(casos, nome, rec)
        for nome, rec in baselines_com_cluster(mapa_cluster).items()
    ]
    print(tabela_recomendadores(resultados, em_percentual=True))
    print(
        "\nProtocolo de imitação: mede acerto do próximo tipo de treino, não "
        "qualidade do conselho. F1-macro importa mais que acerto — um "
        "recomendador constante obtém acerto razoável e F1-macro baixo. "
        "Valores da tabela em %."
    )

    _titulo("ITEM 2 — TESTES PAREADOS CONTRA DÉFICIT VS META DO CLUSTER")
    alvo = NOME_REGRA_META_CLUSTER
    for c in comparar_todos({r.nome: r.erro_por_atleta for r in resultados}):
        if alvo in (c.modelo_a, c.modelo_b):
            outro = c.modelo_b if c.modelo_a == alvo else c.modelo_a
            print(
                f"  vs {outro:34s} p_holm={c.p_ajustado:9.3g}  "
                f"vencedor={c.vencedor or 'empate'}"
            )

    _titulo("ITEM 2 — ABLAÇÃO DOS COMPONENTES DA REGRA POR DÉFICIT")
    print(rodar_ablacao(df, n_teste=args.n_teste))
    print(
        "\n`delta_f1_vs_completa` > 0: remover o componente **melhorou** o "
        "resultado, isto é, o componente atrapalha."
    )


def bloco_desfecho(args: argparse.Namespace) -> None:
    """ADR-018 item 3: validação externa com desfecho de meia maratona."""
    from dataset.constants import DISTANCIA_MEIA_KM
    from dataset.experimentos.desfecho import montar_tabela_desfecho, resumo_desfecho
    from dataset.experimentos.validacao_externa import (
        comparar_encaixados,
        correlacoes_desfecho,
    )

    df = _carregar_externa(args.max_atletas)
    tabela = montar_tabela_desfecho(df, distancia_km=DISTANCIA_MEIA_KM)
    if tabela.height == 0:
        print("nenhum atleta com desfecho; aumente --max-atletas")
        return

    _titulo("ITEM 3 — DESFECHO: MELHOR MEIA MARATONA (21,0975 km)")
    for chave, valor in resumo_desfecho(tabela).items():
        print(f"  {chave:28s} {valor:.3f}" if isinstance(valor, float) else f"  {chave:28s} {valor}")

    _titulo("ITEM 3 — CORRELAÇÃO DAS FEATURES DE TREINO COM O TEMPO DE PROVA")
    print(correlacoes_desfecho(tabela))

    for usar_rf in (False, True):
        nome = "Random Forest" if usar_rf else "Regressão Linear"
        linhas, comparacoes = comparar_encaixados(tabela, usar_random_forest=usar_rf)
        _titulo(f"ITEM 3 — MODELOS ENCAIXADOS ({nome})")
        print(linhas)
        print(comparacoes)


def bloco_meta(args: argparse.Namespace) -> None:
    """ADR-019 item 4: meta de distribuição aprendida para 21 km."""
    from dataset.constants import DISTANCIA_MEIA_KM
    from dataset.experimentos.aprender_meta import (
        aprender_meta,
        tabela_regiao_factivel,
        validar_meta_fora_da_amostra,
    )
    from dataset.experimentos.desfecho import montar_tabela_desfecho

    df = _carregar_externa(args.max_atletas)
    tabela = montar_tabela_desfecho(df, distancia_km=DISTANCIA_MEIA_KM)
    if tabela.height == 0:
        print("nenhum atleta com desfecho; aumente --max-atletas")
        return
    print(f"atletas com desfecho de 21 km: {tabela.height}")

    _titulo("ITEM 4 — A META GLOBAL É ALCANÇÁVEL?")
    print(tabela_regiao_factivel(tabela))
    print(
        "\n`dentro_da_regiao=false`: nenhum atleta atinge aquele componente da "
        "meta, porque o classificador de ADR-004 é relativo ao atleta."
    )

    for restrita in (False, True):
        meta = aprender_meta(tabela, restrita=restrita, n_bootstrap=args.n_bootstrap)
        _titulo(f"ITEM 4 — META APRENDIDA PARA 21 km — {meta.nome}")
        print(f"tempo previsto na meta: {meta.tempo_previsto_min:.1f} min")
        print(meta.como_tabela())
        if restrita:
            instaveis = [t for t in meta.meta if not meta.bem_identificada(t)]
            if instaveis:
                print(
                    f"\nComponentes mal identificados: {', '.join(instaveis)}. "
                    "O IC bootstrap é mais largo que a própria diferença em "
                    "relação à meta global, então o valor aprendido não sustenta "
                    "substituir o afirmado."
                )

    _titulo("ITEM 4 — VALIDAÇÃO FORA DA AMOSTRA")
    print(validar_meta_fora_da_amostra(tabela))
    print(
        "\nCorrelação entre aderência à meta e tempo de prova, medida em folds "
        "em que a meta foi aprendida só no treino. Compare a diferença entre as "
        "metas com `corr_desvio`: se for menor, as metas empatam."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dataset.experimentos",
        description="Experimentos de ADR-017, ADR-018 e ADR-019.",
    )
    parser.add_argument("bloco", choices=BLOCOS)
    parser.add_argument(
        "--max-atletas",
        type=int,
        default=8000,
        help="limite de atletas da fonte externa (None = todos; padrão 8000)",
    )
    parser.add_argument("--n-teste", type=int, default=5, help="corridas por atleta no teste")
    parser.add_argument("--n-bootstrap", type=int, default=100, help="reamostras do IC da meta")
    parser.add_argument("--silencioso", action="store_true")

    args = parser.parse_args(argv)
    _configurar_saida()

    blocos = {
        "protocolo": bloco_protocolo,
        "baselines": bloco_baselines,
        "desfecho": bloco_desfecho,
        "meta": bloco_meta,
    }

    if args.bloco == "tudo":
        for funcao in blocos.values():
            funcao(args)
    else:
        blocos[args.bloco](args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
