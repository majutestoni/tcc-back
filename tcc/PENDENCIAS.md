# PENDÊNCIAS

Tudo o que a Maria precisa fornecer, gerar ou conferir. Agrupado por tipo e por prioridade.

Prioridade: **B** = bloqueia a banca · **M** = melhora o trabalho · **O** = opcional.

## Decisões que só a autora pode tomar

| Prio | Item | Onde |
|---|---|---|
| B | Confirmar o **formato**: monografia (padrão adotado, modelo TCC-2) ou artigo (modelo TCC-1). O capítulo 3 foi escrito em formato monografia. | global |
| B | Confirmar se o **backend orquestrador** entra no escopo do TCC. Hoje o texto o trata como arquitetura **proposta** (ADR-009, estado Proposta; SPEC em Draft). Se a autora quiser apresentá-lo como entrega, é preciso implementá-lo ou reescrever a seção 3.3. | 3.2 (abertura), 3.3 |
| B | Título do trabalho, nome completo, orientador, ano e semestre, curso. | pré-textuais |
| M | Confirmar junto ao orientador se a monografia exige declaração formal sobre **Comitê de Ética em Pesquisa**. O texto adota o entendimento de que dados públicos, anonimizados e preexistentes, sem intervenção, dispensam submissão. | 3.2.1, "Questões éticas" |
| O | Decidir se a volumetria das duas fontes vira tabela numerada ou permanece em texto corrido. | 3.2.1 |

## Dados pendentes (`[DADO PENDENTE]`)

| Prio | Item | Como obter |
|---|---|---|
| B | **Licença e data de acesso do conjunto `olegoaer/running-races-strava`** (fonte primária). Necessária para a referência ABNT e para a seção de ética. | abrir a página do conjunto no Kaggle, registrar a licença declarada e a data de acesso |
| M | Autoria real por trás do usuário `olegoaer` (nome completo), para a referência. | página do perfil no Kaggle |
| M | Autoria real por trás do usuário `mexwell`, idem. | página do perfil no Kaggle |

## Números a confirmar (`[CONFIRMAR NÚMERO]`)

| Prio | Item | Como obter |
|---|---|---|
| B | **6.821 atletas com desfecho de 21 km** — valor das ADR-018/019, não reexecutado. | `source .venv/bin/activate && python -m dataset.experimentos desfecho --max-atletas 12000` (a base externa já está em `~/.cache/tcc-back-2/dados`) |
| M | "~394 MB compactados / 2,3 GB descompactados" da fonte secundária — valor do script. Os 4 arquivos em disco somam ~2,2 GB (medido). | `du -h ~/.cache/tcc-back-2/dados` |

## Referências a verificar (`[VERIFICAR REFERÊNCIA]`)

Nenhuma obra externa foi citada nesta invocação — o capítulo 3 até aqui cita apenas
artefatos do próprio projeto. As referências aos dois conjuntos de dados do Kaggle precisam
ser compostas segundo a NBR 6023 assim que os itens acima forem resolvidos.

## Figuras, quadros e tabelas a produzir

Ver `tcc/PLANO-DE-FIGURAS.md`. Resumo desta invocação:

| Prio | Item |
|---|---|
| B | Figura 1 — fluxo de trabalho (desenhar) |
| B | Figura 2 — arquitetura e fronteira implementado × proposto (desenhar) |
| M | Figura 3 — distribuição anual dos registros (gerar com matplotlib) |
| B | Quadro 3 — dicionário de variáveis (compor) |
| B | Quadro 4 — adaptador da fonte primária (formatar o código já transcrito) |
| M | Quadro 5 — comparação entre as duas fontes (compor) |
| M | Tabela 1 — amostra de cinco linhas do CSV bruto (compor) |

## Divergências entre documentos do projeto

| Prio | Divergência | Situação |
|---|---|---|
| M | **ADR-018 diz "10,7 M registros"** na fonte externa; a contagem medida em `run_ww_2019_d.csv` é de **13.290.380 linhas** (4.677.389 com `distance > 0`). | **Aberta.** Não corrigi o ADR porque não sei a que recorte o número original se refere. Decidir e, se confirmada, corrigir o ADR e registrar no Histórico de atualizações. |
| O | **`FLUXO.md` cita "ADR-001…026"**; o `ADR.md` já vai até ADR-027. | Correção editorial de uma linha no `FLUXO.md`. Não alterada nesta invocação. |
| — | 114 × 111 atletas (ADR-017 × ADR-007) | **Resolvida por medição:** 116 no bruto → 114 após `preprocessar` → 111 com ≥ 20 corridas (`MIN_CORRIDAS_PERFIL`). Os três números estão corretos em etapas diferentes. Registrado em `01-FATOS.md` C01 e explicado no texto. |
| M | 76% × 78% de recomendações `leve` da regra global antiga | Aberta; escopo das seções 3.2.10/3.2.11. |
| M | Desvio entre *folds* do MAE de pace: 0,009→0,067 (ADR-017) × 0,010→0,043 (GUIA) | Aberta; escopo da seção 3.2.9/3.2.10. |
