# 01-FATOS — fonte única de números e afirmações

Todo número que entra no texto do TCC sai desta tabela. Estado ∈ `confirmado` (reexecutado
nesta máquina) / `documento` (só consta em ADR/FLUXO/GUIA, não reexecutado) / `conflito` /
`pendente`.

Comandos de confirmação executados em 06/11/2026, com `source .venv/bin/activate`.

## Fonte primária — `dataset/raw-data-kaggle.csv`

| # | Afirmação / número | Valor | Procedência | Estado |
|---|---|---|---|---|
| F01 | Separador do CSV | `;` | `dataset/fontes/strava_kaggle.py` (`SEPARADOR = ";"`); FLUXO §1 | confirmado |
| F02 | Linhas de dados (sem cabeçalho) | 42.116 | `pl.read_csv(...).height` | confirmado |
| F03 | Colunas | 7 | idem | confirmado |
| F04 | Nomes das colunas | `athlete`, `gender`, `timestamp`, `distance (m)`, `elapsed time (s)`, `elevation gain (m)`, `average heart rate (bpm)` | cabeçalho do CSV; `dataset/constants.py` | confirmado |
| F05 | Tipos inferidos | Int64, String, String, Float64, Int64, Float64, Float64 | `df.schema` | confirmado |
| F06 | Atletas distintos no bruto | 116 | `df["athlete"].n_unique()` | confirmado |
| F07 | Corridas por atleta (bruto): mín / mediana / máx | 6 / 269,5 / 1.215 | `df.group_by("athlete").len()` | confirmado |
| F08 | Registros com `gender = M` | 32.289 (76,67%) | `value_counts` | confirmado |
| F09 | Registros com `gender = F` | 9.472 (22,49%) | `value_counts` | confirmado |
| F10 | Registros com `gender` nulo | 355 (0,84%) | `value_counts` | confirmado |
| F11 | Ausência em `average heart rate (bpm)` | 18.384 registros = 43,65% | `null_count` | confirmado |
| F12 | Ausência em `elevation gain (m)` | 0 (0,00%) — mas com valor 0,0 como mínimo | `null_count` / `describe` | confirmado |
| F13 | Ausência nas demais colunas | 0 | `null_count` | confirmado |
| F14 | Intervalo de `timestamp` | 04/01/2000 12:06 a 06/01/2020 05:24 | `str.strptime(...).min()/.max()` | confirmado |
| F15 | Timestamps não parseáveis no formato `%d/%m/%Y %H:%M` | 0 | idem | confirmado |
| F16 | Concentração temporal (registros por ano) | 2000:1 · 2010:4 · 2011:1 · 2012:61 · 2013:332 · 2014:975 · 2015:2.237 · 2016:4.463 · 2017:8.243 · 2018:12.230 · 2019:13.411 · 2020:158 | `dt.year().value_counts()` | confirmado |
| F17 | `distance (m)` — mín / p25 / mediana / p75 / máx | 0,0 / 7.189,7 / 10.137,0 / 13.236,3 / 218.950,0 | `describe()` | confirmado |
| F18 | `elapsed time (s)` — mín / mediana / máx | 0 / 3.442 / 2.971.531 | `describe()` | confirmado |
| F19 | `elevation gain (m)` — mín / mediana / máx | 0,0 / 86,0 / 12.078,5 | `describe()` | confirmado |
| F20 | `average heart rate (bpm)` — mín / mediana / máx (nos 23.732 presentes) | 0,0 / 148,4 / 237,0 | `describe()` | confirmado |
| F21 | Corridas após `preprocessar(df)` | 40.534 | `preprocessar(df).height` — bate com ADR-018 | confirmado |
| F22 | Atletas após `preprocessar(df)` | 114 | `preprocessar(df)["athlete"].n_unique()` — bate com ADR-017 | confirmado |
| F23 | Atletas com ≥ 20 corridas após o pipeline (`MIN_CORRIDAS_PERFIL`) | 111 | `group_by("athlete").len().filter(len>=20).height` | confirmado — **resolve a divergência 114 × 111** |
| F24 | Distribuição de `tipo_treino` (agregada sobre corridas, pós-pipeline) | leve 32,6% (13.213) · longão 25,2% (10.205) · ritmo 24,0% (9.718) · intervalado 18,3% (7.398) | `group_by("tipo_treino")` | confirmado — bate com o "~33/25/24/18" do ADR-007 |

## Fonte secundária — `mexwell/long-distance-running-dataset`

| # | Afirmação / número | Valor | Procedência | Estado |
|---|---|---|---|---|
| F30 | Identificador Kaggle | `mexwell/long-distance-running-dataset` | ADR-018; `scripts/baixar_long_distance.sh` | confirmado |
| F31 | Licença | CC BY 4.0 | ADR-018; cabeçalho de `scripts/baixar_long_distance.sh` | documento (não conferido na página do Kaggle) |
| F32 | Tamanho compactado / descompactado | ~394 MB / ~2,3 GB | `scripts/baixar_long_distance.sh` | documento |
| F33 | Arquivos usados | `run_ww_2019_d.csv`, `run_ww_2019_w.csv`, `run_ww_2020_d.csv`, `run_ww_2020_w.csv` | `scripts/baixar_long_distance.sh` (`ARQUIVOS`) | confirmado |
| F34 | Tamanho em disco de `run_ww_2019_d.csv` | 956.531.836 bytes (~956 MB) | `ls -la ~/.cache/tcc-back-2/dados` | confirmado |
| F35 | Colunas cruas | `""` (índice), `datetime`, `athlete`, `distance`, `duration`, `gender`, `age_group`, `country`, `major` | `pl.scan_csv(...).collect_schema().names()` | confirmado |
| F36 | Linhas do painel diário de 2019 | 13.290.380 | `scan_csv(...).select(pl.len())` | **conflito** com "10,7 M registros" do ADR-018 |
| F37 | Atletas distintos (2019, diário) | 36.412 | `n_unique()` | confirmado — bate com ADR-018 |
| F38 | Linhas com `distance > 0` (corridas efetivas, 2019) | 4.677.389 (35,2% do painel) | `filter(distance>0)` | confirmado — coerente com "~65% de dias de descanso" do ADR-018 |
| F39 | Unidades originais | `distance` em km; `duration` em min; `datetime` ISO `%Y-%m-%d` | `dataset/fontes/long_distance.py` | confirmado |
| F40 | Ausência de FC e de elevação | total (colunas criadas nulas pelo adaptador) | `long_distance.py`, `FonteDados(tem_fc=False, tem_elevacao=False)` | confirmado |
| F41 | Diretório padrão de download | `~/.cache/tcc-back-2/dados`, sobrescrevível por `DIRETORIO_DADOS_EXTERNOS` | `long_distance.py`, `scripts/baixar_long_distance.sh` | confirmado |
| F42 | Atletas com desfecho de 21 km utilizável | 6.821 | ADR-018, ADR-019, GUIA | documento (não reexecutado: exige `python -m dataset.experimentos desfecho`) |
| F43 | Distância-alvo da meia maratona | 21,0975 km, tolerância 0,6 km | `dataset/constants.py` (`DISTANCIA_MEIA_KM`, `TOLERANCIA_PROVA_KM`) | confirmado |
| F44 | Janela de preparo considerada | 12 semanas, mínimo de 20 treinos | `constants.py` (`SEMANAS_PREPARO`, `MIN_TREINOS_PREPARO`) | confirmado |
| F45 | Validação do desfecho de maratona (Boston 2019) | 4.294 de 4.365 inscritos com ~42 km em 15/04/2019 | ADR-018 | documento |

## Conflitos registrados

| # | Conflito | Situação |
|---|---|---|
| C01 | Atletas: 114 (ADR-017) × 111 (ADR-007, "111 de 111") | **Resolvido**: 116 no bruto → 114 após `preprocessar` → 111 com ≥ 20 corridas (`MIN_CORRIDAS_PERFIL`). Os três números são corretos em etapas diferentes e devem ser apresentados como tal (F06/F22/F23). |
| C02 | "10,7 M registros" (ADR-018) × 13.290.380 linhas medidas em `run_ww_2019_d.csv` | **Aberto.** Hipóteses: o ADR pode se referir a outro arquivo/recorte, ou a contagem após algum filtro. Não citar "10,7 M" sem resolver. Sugestão de redação: usar os números medidos (F36/F38). |
| C03 | Proporção de recomendações `leve` da regra global: 76% (ADR-007/017) × 78% (GUIA) | Aberto — fora do escopo desta invocação (3.2.10/3.2.11). |
| C04 | Desvio entre *folds* do MAE de pace: 0,009→0,067 (ADR-017) × 0,010→0,043 (GUIA, RF de pace) | Aberto — fora do escopo desta invocação. |
| C05 | `FLUXO.md` cita "ADR-001…026"; `ADR.md` vai até ADR-027 | Aberto — correção editorial no `FLUXO.md`, a confirmar com a autora. |
