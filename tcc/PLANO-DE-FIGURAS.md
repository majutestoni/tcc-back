# PLANO DE FIGURAS, QUADROS E TABELAS

Numeração contínua por tipo ao longo de todo o trabalho. **Reservas feitas para seções
ainda não escritas** estão marcadas como `RESERVADO` — as próximas invocações devem
respeitá-las e continuar a partir do último número usado.

Estado ∈ `existe` (arquivo já no repositório) / `gerar` (rodar comando ou script) /
`capturar` (captura de tela) / `desenhar` (diagrama) / `compor` (montar tabela no editor).

## Figuras

| ID | Seção | Legenda | Conteúdo | Origem | Estado |
|---|---|---|---|---|---|
| Figura 1 | 3.2 (abertura) | Fluxo de trabalho do modelo proposto | Seis caixas: coleta, limpeza e engenharia, categorização, modelos auxiliares, recomendação (A regras / B LLM), validação (3 frentes) | `FLUXO.md`, "Mapa mental (o fluxo inteiro)" e "Estrutura do código" | desenhar (draw.io ou Mermaid) |
| Figura 2 | 3.2 (abertura) | Arquitetura do ecossistema e fronteira do que foi implementado | Frontend ↔ Backend orquestrador ↔ Strava / ML / Banco, com linha tracejada no que é proposto | `FLUXO.md` ("No ecossistema completo"); `SPEC-BACKEND-ORQUESTRADOR.md` §1.3 e §6; ADR-009 | **pronto** (Mermaid no capítulo; exportar PNG para a versão final) |
| Figura 3 | 3.2.1 | Distribuição anual dos registros da fonte primária | Barras por ano, 2000–2020, com destaque em 2017–2019 (80,5% dos dados) | Polars sobre `dataset/raw-data-kaggle.csv`; valores em `01-FATOS.md` F16 | gerar (matplotlib) |

**Próximo número de figura livre: 4.**

Reservas previstas (não numeradas ainda, para evitar buracos): diagrama de atividades da
decisão de recomendação (3.2.7), gráficos de validação de *cluster* já existentes em
`notebooks/output/cluster_viz/` (3.2.5), capturas das 8 telas da demo (3.3).

## Quadros

| ID | Seção | Legenda | Conteúdo | Origem | Estado |
|---|---|---|---|---|---|
| Quadro 1 | 3.1 | Requisitos funcionais | RF01… derivados do que o repositório faz | a definir ao escrever 3.1 | RESERVADO |
| Quadro 2 | 3.1 | Requisitos não funcionais | RNF01… | a definir ao escrever 3.1 | RESERVADO |
| Quadro 3 | 3.2.1 | Dicionário de variáveis da fonte primária | 7 variáveis × tipo, unidade, domínio, papel | cabeçalho do CSV; `dataset/constants.py`; `01-FATOS.md` F04/F05/F17–F20 | **pronto** (montado no capítulo) |
| Quadro 4 | 3.2.1 | Adaptador da fonte primária (`dataset/fontes/strava_kaggle.py`) | 22 linhas numeradas de código | `dataset/fontes/strava_kaggle.py` | existe (texto já no capítulo; falta formatar como quadro) |
| Quadro 5 | 3.2.1 | Comparação entre a fonte primária e a fonte secundária | 14 aspectos × 2 fontes | `01-FATOS.md` F01–F45 | **pronto** (montado no capítulo) |

**Próximo número de quadro livre: 6.**

## Tabelas

| ID | Seção | Legenda | Conteúdo | Origem | Estado |
|---|---|---|---|---|---|
| Tabela 1 | 3.2.1 | Amostra das cinco primeiras linhas da fonte primária | 5 registros brutos do atleta 18042525 | `pl.read_csv(..., separator=";").head(5)` | compor (valores já apurados) |

**Próximo número de tabela livre: 2.**

Observação: a volumetria da fonte primária e a da fonte secundária foram redigidas **em
texto corrido**, não em tabela, para não fragmentar a argumentação. Caso a autora prefira
tabelar, os números estão em `01-FATOS.md` (F02–F24 e F30–F45) e as tabelas devem assumir
os números 2 e 3, deslocando as demais.

## Equações

Nenhuma equação numerada foi introduzida nesta invocação. A primeira equação prevista é a
do *pace* (seção 3.2.2), que deve receber o número (1).
