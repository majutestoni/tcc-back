# ADR — tcc-back-2 (Serviço de ML / Análise de Corridas)

> **Documento vivo mantido pela IA.**
> Este é o registro de decisões de arquitetura (Architecture Decision Record) do projeto.
>
> **Instruções para a IA (assistente):**
> 1. **Consulte** este arquivo antes de responder dúvidas, implementar features ou refatorar código neste repositório. Ele é a fonte de verdade sobre o *porquê* das decisões.
> 2. **Atualize** este arquivo sempre que: uma nova decisão for tomada, uma decisão existente for revertida/alterada, uma convenção mudar, ou uma dúvida recorrente for resolvida.
> 3. Ao adicionar uma decisão, crie uma nova entrada `## ADR-NNN` (numeração sequencial) e registre a mudança no **Histórico de atualizações** no fim do arquivo.
> 4. Nunca apague decisões antigas: marque-as como `Status: Substituída por ADR-NNN` para preservar o rastro histórico.
> 5. Mantenha o conteúdo alinhado ao código real. Se encontrar divergência entre este doc e o código, sinalize e corrija.

---

## 1. Contexto do Projeto

**tcc-back-2** é o **serviço de Machine Learning / análise de dados** de um TCC sobre análise e recomendação de treinos para corredores amadores. É um projeto **Python de ciência de dados** (não é uma API ainda).

Ele faz parte de um ecossistema maior de 3 repositórios:

| Repositório | Papel |
|-------------|-------|
| `tcc-back-2` (**este**) | Serviço de ML: preprocessamento, categorização de corredores, recomendação de treino |
| `run-metrics-backend` | Backend orquestrador (a construir) — OAuth Strava, sync, persistência. Ver [SPEC-BACKEND-ORQUESTRADOR.md](SPEC-BACKEND-ORQUESTRADOR.md) |
| `run-metrics-mate` | Frontend ([github.com/majutestoni/run-metrics-mate](https://github.com/majutestoni/run-metrics-mate)) |

**Fluxo pretendido:** Frontend → Backend orquestrador → (Strava API + este serviço ML) → Banco.

**Dataset base:** [Kaggle — running-races-strava](https://www.kaggle.com/datasets/olegoaer/running-races-strava), armazenado em [dataset/raw-data-kaggle.csv](dataset/raw-data-kaggle.csv) (separador `;`).

---

## 2. Visão Geral da Arquitetura Atual

Projeto Python organizado como pacote `dataset/`:

```text
dataset/
├── constants.py            # Nomes de colunas + limiares/thresholds (fonte única)
├── preprocessamento.py     # Limpeza, cálculo de pace, classificação de tipo_treino
├── analise.py              # EDA + gráficos (histogramas, radar, coordenadas paralelas)
├── categorizar/            # Categorização de corredores
│   ├── perfil_atletas.py       # Agregação de métricas por atleta
│   ├── definir_perfil_atleta.py# Perfil por regras (iniciante/intermediario/fundista/avancado)
│   ├── kmeans.py               # Clusterização KMeans (global e por gênero)
│   ├── validacao_cluster.py    # Métricas internas (silhouette, davies-bouldin, etc.)
│   └── ...
├── models/
│   ├── model_main.py           # Ponto de entrada (python -m dataset.models.model_main)
│   ├── resultado.py            # dataclass ResultadoModelo (contrato de saída dos modelos)
│   ├── visualizacao.py         # Exibição de resultados
│   ├── previsao_pace/          # Modelos de previsão de pace (LinReg, RF, RF regressor)
│   ├── previsao_tempo/         # Modelos de previsão de tempo (RF, LinReg, XGBoost)
│   └── previsao_treino/        # Recomendação do próximo treino (regras + RF)
├── fontes/                 # Adaptadores de fonte de dados (ADR-018)
│   ├── base.py                 # FonteDados: schema canônico + tem_fc/tem_elevacao
│   ├── strava_kaggle.py        # Fonte primária
│   └── long_distance.py        # Fonte externa de validação (sem FC/elevação)
├── avaliacao/              # Protocolo experimental (ADR-017)
│   ├── particionamento.py      # GroupKFold por atleta, splits temporais
│   ├── estatistica.py          # IC bootstrap, Wilcoxon pareado, Holm
│   ├── avaliador.py            # Predições out-of-fold + erro por atleta
│   ├── registro_modelos.py     # EspecModelo dos 8 modelos
│   └── comparacao.py           # Tabela comparativa + testes pareados
├── experimentos/           # Experimentos (ADR-017/018/019)
│   ├── protocolo_recomendacao.py # Origem rolante para recomendadores
│   ├── baselines.py            # 7 baselines + regra de ADR-007
│   ├── ablacao.py              # Ablação dos componentes de ADR-007
│   ├── desfecho.py             # Desfecho de prova: 21 km e maratona
│   ├── validacao_externa.py    # Modelos encaixados vs desfecho
│   ├── aprender_meta.py        # Meta de distribuição via Scheffé
│   └── __main__.py             # CLI dos experimentos
└── raw-data-kaggle.csv
notebooks/                  # Notebooks Jupyter para visualização/comparação
tests/                      # Testes de unidade (pytest)
scripts/                    # baixar_long_distance.sh (fonte externa, ADR-018)
```

Execução principal: `python -m dataset.models.model_main` (recomendação por cluster, ADR-024).

Experimentos: `python -m dataset.experimentos protocolo|baselines|desfecho|meta|tudo`.

---

## 3. Decisões de Arquitetura (ADRs)

### ADR-001 — Polars como biblioteca de manipulação de dados
**Status:** Aceita
**Contexto:** Necessidade de manipular o dataset de corridas de forma performática e com API expressiva.
**Decisão:** Usar **Polars** (`pl`) em vez de Pandas em todo o pipeline de dados.
**Consequências:** Sintaxe de expressões (`pl.col`, `.over`, `.with_columns`). Todo código novo de dados deve seguir o padrão Polars já estabelecido.

---

### ADR-002 — Nomes de colunas e limiares centralizados em `constants.py`
**Status:** Aceita
**Contexto:** Os nomes de colunas do CSV têm espaços e unidades (ex.: `"distance (m)"`, `"average heart rate (bpm)"`) e são usados em vários módulos.
**Decisão:** Centralizar **todos** os nomes de coluna (`COL_*`) e todos os limiares/thresholds em [dataset/constants.py](dataset/constants.py). Nenhum módulo deve usar string literal de coluna.
**Consequências:** Qualquer novo threshold ou coluna entra em `constants.py`. Facilita calibração e evita erros de digitação.

---

### ADR-003 — Pipeline de preprocessamento fixo e determinístico
**Status:** Aceita
**Contexto:** Dados brutos do Strava/Kaggle contêm ruído (distâncias ínfimas, pace irreal, FC ausente/inválida, gênero inválido).
**Decisão:** [preprocessamento.py](dataset/preprocessamento.py) aplica, na ordem, `preprocessar(df)`:
1. `filtrar_genero_valido` — mantém apenas `M`/`F`.
2. `imputar_frequencia_cardiaca` — FC ausente preenchida com a **mediana**.
3. `filtrar_corridas_validas` — remove distância < 1000 m; calcula `pace_min_km = (tempo/60)/(dist/1000)`; mantém `2 < pace < 15` e `40 < FC < 220`.
4. `classificar_tipo_treino` — rotula cada corrida.
**Consequências:** Todo modelo consome o DataFrame **já preprocessado**. Mudanças nos filtros afetam todos os modelos a jusante.

---

### ADR-004 — Classificação de tipo de treino relativa ao atleta
**Status:** Aceita
**Contexto:** "Rápido" e "longo" são relativos ao histórico de cada corredor, não absolutos.
**Decisão:** `classificar_tipo_treino` usa estatísticas **por atleta** (`.over(COL_ATHLETE)`): pace médio, desvio-padrão do pace, percentil 75 de distância e FC mediana. Classes: **longao, intervalado, leve, ritmo** (nessa ordem de prioridade das regras). Limiares em `LIMIAR_PACE_*` e `LIMIAR_DIST_LONGAO` em `constants.py`.
**Consequências:** A classificação depende de haver histórico suficiente por atleta. Atletas com poucos dados têm `std` preenchido com `0.5` como fallback.

---

### ADR-005 — Perfil de atleta por regras interpretáveis (além do KMeans)
**Status:** Aceita
**Contexto:** Clusters de KMeans não são autoexplicativos. O TCC precisa de rótulos interpretáveis.
**Decisão:** Manter **duas abordagens complementares** de categorização de corredores:
- **Por regras** ([definir_perfil_atleta.py](dataset/categorizar/definir_perfil_atleta.py)): três eixos (pace em tercis, volume/frequência, distância) combinados em `perfil_atleta` ∈ {`iniciante`, `intermediario`, `fundista`, `avancado`}.
- **Por KMeans** ([kmeans.py](dataset/categorizar/kmeans.py)): clusterização não supervisionada.
**Consequências:** Ambas partem de `montar_perfil_atletas` (agregação mediana por atleta). Filtro `MIN_CORRIDAS_PERFIL = 20` aplicado no perfil por regras.

> ⚠️ **Débito técnico conhecido:** `montar_perfil_atletas` está **duplicado** em [perfil_atletas.py](dataset/categorizar/perfil_atletas.py) e [definir_perfil_atleta.py](dataset/categorizar/definir_perfil_atleta.py) (a versão de `perfil_atletas.py` inclui elevação). Consolidar em uma fonte única quando houver oportunidade.

---

### ADR-006 — KMeans com padronização e suporte a segmentação por gênero
**Status:** Aceita
**Contexto:** Features de corrida têm escalas muito diferentes (metros vs. min/km vs. bpm). Homens e mulheres têm distribuições distintas.
**Decisão:** [kmeans.py](dataset/categorizar/kmeans.py) usa `StandardScaler` antes do `KMeans` (`random_state=42`, `n_init=10`). Features padrão: `[distância, pace]`. Duas variantes:
- `treina_kmeans` — clusteriza toda a base.
- `treina_kmeans_por_genero` — clusteriza M e F separadamente (gera até 2×k categorias, rótulos `C{i}-{M|F}`).
Validação via `metricas_internas` (silhouette, davies-bouldin, calinski-harabasz).
**Consequências:** Centros são reportados na escala original (`inverse_transform`). `random_state=42` fixo para reprodutibilidade.

---

### ADR-007 — Recomendação de treino por regras baseada em déficit de distribuição
**Status:** Substituída por ADR-024 (produto). Defeito estrutural documentado em ADR-017/ADR-019; a `META` global permanece só como artefato histórico / experimentos de desfecho.
**Contexto:** Recomendar o próximo treino de forma explicável, corrigindo desequilíbrios no plano de treino do atleta.
**Decisão:** [previsao_treino/model.py](dataset/models/previsao_treino/model.py) — `treina_modelo`:
1. Calcula a distribuição de tipos de treino do atleta numa **janela de `SEMANAS_JANELA = 6` semanas**.
2. Compara com a meta `META` (`leve 60%, longao 20%, ritmo 12%, intervalado 8%`).
3. Recomenda o tipo com maior **déficit ponderado** (`PESOS_DEFICIT`).
4. Regra anti-fadiga: evita repetir longão/intervalado logo após intervalado ou longão.
5. Calibra distância/pace por tipo a partir das medianas do histórico (`calibrar_parametros`).
**Consequências:** Existe também `random_forest.py` (`treina_rf_recomendacao`) como alternativa de ML supervisionado, atualmente **desativada** no `model_main.py`. As regras são o baseline interpretável do TCC.

> ⚠️ **Defeito estrutural (medido em ADR-017).** A `META` é **inalcançável** dado o classificador de ADR-004. Como `longao` é definido pelo percentil 75 de distância *do próprio atleta*, ~25% das corridas de todo atleta caem nessa classe por construção, e as faixas de pace dividem o resto de forma aproximadamente equilibrada. A distribuição real é ~33% leve / 25% longão / 24% ritmo / 18% intervalado — nunca 60% leve. Consequências medidas:
> - o déficit de `leve` é positivo para **111 de 111** atletas;
> - `treina_modelo` emite `leve` em **76%** das recomendações;
> - no protocolo de ADR-017 a regra perde de `sempre_leve` (F1-macro 0,142 vs 0,128 mas acerto 0,265 vs 0,344, com `p_holm = 6,7e-4` a favor do baseline constante) e **empata** com `aleatorio` e `repetir_ultimo`.
>
> O mesmo padrão se reproduz na fonte externa de ADR-018 (30,4% / 25,7% / 24,2% / 19,6%), então não é artefato do dataset primário. Correção proposta em ADR-019.

---

### ADR-008 — Contrato único de saída de modelo: `ResultadoModelo`
**Status:** Aceita
**Contexto:** Vários tipos de modelo (pace, tempo, treino, corredor) precisam ser exibidos/comparados de forma uniforme.
**Decisão:** Todo modelo retorna um `ResultadoModelo` (dataclass em [resultado.py](dataset/models/resultado.py)) com `nome`, `categoria`, `metricas` e campos opcionais (`recomendacoes`, `df_exibir`, `y_pred`, `class_names`, etc.).
**Consequências:** Camada de visualização/comparação depende desse contrato. Novos modelos devem retornar `ResultadoModelo`.

---

### ADR-009 — Integração futura com o backend via HTTP `POST /analyze`
**Status:** Proposta (ainda não implementada)
**Contexto:** O backend orquestrador precisará chamar este serviço de ML por HTTP. Ver [SPEC-BACKEND-ORQUESTRADOR.md](SPEC-BACKEND-ORQUESTRADOR.md) seção 6.6.
**Decisão (planejada):** Expor uma API (provavelmente FastAPI) com `POST /analyze` recebendo `{ athlete_id, gender, activities[] }` e retornando `{ profile, cluster, recommendation, distribution }`. O schema de entrada mapeia para as colunas de `constants.py`.
**Consequências:** Ainda **não há** camada HTTP neste repo. Quando implementada, atualizar esta ADR para `Aceita` e documentar a estrutura de pastas da API.

---

### ADR-010 — Recomendação de treino via API do Gemini (LLM)
**Status:** Aceita
**Contexto:** Além do baseline por regras (ADR-007), queremos recomendações mais ricas e explicáveis (justificativa em linguagem natural) para um atleta específico, aproveitando um LLM.
**Decisão:** Novo pacote [gemini/](gemini/) (na raiz, fora de `dataset/`) que usa a API do Gemini (SDK oficial `google-genai`):
- `config.py` — `GeminiConfig` lê `GEMINI_API_KEY`, `GEMINI_MODEL` (padrão `gemini-3.6-flash`), `GEMINI_TEMPERATURE` (padrão `0.3`) via env/`.env`.
- `schemas.py` — `FeaturesAtleta` (entrada) e `RecomendacaoTreino` (saída Pydantic).
- `features.py` — `montar_features_atleta` **reaproveita** a engenharia determinística de ADR-007 (janela de 6 semanas, distribuição, déficits vs `META`, últimos treinos) + perfil por regras (ADR-005).
- `prompts.py` — `SYSTEM_INSTRUCTION` (persona de treinador) + prompt com as features em JSON.
- `client.py` — wrapper `GeminiClient.gerar_estruturado` usando `response_mime_type="application/json"` + `response_schema` (structured output, `response.parsed`).
- `recomendacao.py` — `recomendar_treino_gemini(df, athlete_id)`; executável via `python -m gemini.recomendacao <athlete_id>`.

**Otimização (o "porquê"):** o trabalho pesado e determinístico é feito em código (features confiáveis, poucos tokens); o LLM só faz o raciocínio final e a justificativa. Temperatura baixa e saída estruturada garantem previsibilidade. Modelo configurável para permitir versões mais novas.

**Compatibilidade:** `RecomendacaoTreino` mantém os campos do contrato de ADR-007 e do SPEC 6.6 (`tipo`, `descricao`, `distancia_km`, `pace_alvo_min_km`, `tempo_estimado_min`) e acrescenta `justificativa` e `alertas`. As regras (ADR-007) seguem como baseline; o Gemini é uma **alternativa complementar**, não um substituto.

**Consequências:** Nova dependência `google-genai` (traz Pydantic). Exige `GEMINI_API_KEY` (nunca commitar `.env`; ver `.env.example`). Requer rede em runtime. Quando a API HTTP de ADR-009 existir, `POST /analyze` poderá optar por regras ou Gemini.

---

### ADR-011 — Streamlit como camada de demonstração (complementar aos notebooks)
**Status:** Aceita (implementação inicial: só a página de recomendação)
**Contexto:** Os notebooks (`notebooks/`) são ótimos para exploração, mas ruins para demonstração/apresentação do TCC. As funções de plot já retornam objetos matplotlib e os modelos retornam `ResultadoModelo` (ADR-008), o que mapeia bem para uma UI.
**Decisão:** Adicionar um app **Streamlit** em [app/](app/) como vitrine do serviço de ML, **sem remover os notebooks** (decisão da usuária: manter ambos):
- `app/Home.py` — página inicial.
- `app/shared.py` — loaders cacheados (`@st.cache_data`) e helpers puros que **reaproveitam** `dataset/` e `gemini/`.
- `app/pages/1_Recomendacao_Treino.py` — campo para informar o ID do atleta; mostra perfil, distribuição vs meta, recomendação por regras (ADR-007) e recomendação via Gemini (ADR-010).
- `app/pages/2_Atletas.py` — tabela de atletas (id, gênero, perfil, corridas, frequência, distância, pace, FC) com filtros e busca por ID; botão de download e de exportação da planilha `output/atletas.csv` (gerada por `exportar_planilha_atletas`). A tabela é leve/cacheada; a planilha existe para consulta fora do app.
Execução: `streamlit run app/Home.py`.
**Escopo atual:** páginas de Recomendação de Treino e Atletas. As demais telas (dataset, clusterização, comparação) podem virar novas páginas depois.
**Consequências:** Nova dependência `streamlit`. O app é **demo interna do TCC**, não o produto final (o frontend oficial segue sendo `run-metrics-mate` sobre o backend, ver SPEC). Funções que chamam `plt.show()` internamente exigem wrapper que retorne a `fig` quando forem portadas; por isso `app/shared.py` monta seus próprios gráficos a partir dos dados retornados.

---

### ADR-012 — Imports lazy para não carregar dependências pesadas fora de uso
**Status:** Aceita
**Contexto:** A página `2_Atletas.py` do Streamlit (ADR-011) ficou **muito lenta para carregar** (~25 s no primeiro acesso, ~5,5 s nos seguintes). Medição isolou o custo: o pipeline de dados em si leva ~0,12 s; o gargalo era o **import**. A página só usa código Polars puro (`definir_perfil_atleta`, `montar_perfil_atletas`), mas:
- `dataset/categorizar/__init__.py` importava **avidamente** `kmeans` (arrasta `sklearn`, ~3 s + build de cache de fonte), `validacao_cluster` e `visualizacao_fichas`. Como o Python executa o `__init__.py` do pacote antes de qualquer submódulo, importar `dataset.categorizar.definir_perfil_atleta` já pagava o custo do sklearn.
- `dataset/models/previsao_treino/model.py` importava `matplotlib.pyplot`/`numpy` no topo, embora só sejam usados nas funções de plot; `app/shared.py` importa desse módulo (`META`, `calcular_deficit`, ...), arrastando matplotlib para a página de Atletas.

**Decisão:** Adotar **imports lazy** para dependências pesadas (sklearn, matplotlib, numpy) que não estão no caminho quente:
- `dataset/categorizar/__init__.py` usa `__getattr__` de módulo (PEP 562) com um mapa nome→submódulo. A API pública (`from dataset.categorizar import treina_kmeans`) continua idêntica, mas o submódulo só é importado no primeiro acesso ao símbolo. (De brinde, corrigiu um bug latente: `comparar_combinacoes_features`, `comparar_combinacoes_e_k` e `sugerir_melhor_config` estavam no `__all__` mas nunca eram importados.)
- Em `model.py`, `matplotlib.pyplot`/`numpy` foram movidos para dentro de `plotar_recomendacao_atleta` e `plotar_resumo_recomendacoes`.
- Em `app/shared.py`, `matplotlib` é importado dentro de `grafico_distribuicao` (com `matplotlib.use("Agg")` antes de `pyplot`).

**Consequências:** A página de Atletas passou a carregar em ~1 s (majoritariamente o import baseline do próprio `streamlit`), sem tocar em sklearn/matplotlib. **Convenção:** `__init__.py` de pacotes que agregam submódulos pesados deve usar `__getattr__` lazy; módulos de dados não devem importar libs de plot/ML no topo se elas só são usadas em funções específicas.

---

### ADR-013 — Persistência do histórico de recomendações da IA em CSV
**Status:** Aceita
**Contexto:** As recomendações via Gemini (ADR-010) eram efêmeras — geradas e exibidas, mas não guardadas. O TCC precisa de um **histórico** consultável das recomendações da IA por atleta.
**Decisão:** Persistir as recomendações em **CSV** (`output/recomendacoes.csv`), seguindo o padrão já usado para `output/atletas.csv` (ADR-011) e a stack Polars (ADR-001) — **sem introduzir banco de dados** (a persistência "de verdade" segue sendo responsabilidade do backend orquestrador, ver SPEC). Novo módulo [gemini/historico.py](gemini/historico.py):
- `salvar_recomendacao(athlete_id, recomendacao, origem="gemini", quando=None)` — anexa uma linha ao CSV (append: lê + `pl.concat` + reescreve). Timestamp no formato da convenção `"%d/%m/%Y %H:%M"`. Aceita **tanto** o `RecomendacaoTreino` do Gemini (com `justificativa`/`alertas`) **quanto** o dict da recomendação por regras (ADR-007) — nesse caso `justificativa` fica vazia e `alertas` `[]`, pois o modelo por regras não os produz.
- `carregar_historico()` — lê o CSV (schema fixo); retorna tabela vazia se ainda não existir.
- `alertas_para_lista(valor)` — os `alertas` (lista) são serializados como **JSON** na coluna para round-trip fiel.
- Colunas: `athlete_id, timestamp, origem, tipo, descricao, distancia_km, pace_alvo_min_km, tempo_estimado_min, justificativa, alertas`.
**Salvamento automático:** ao gerar via Gemini na página `1_Recomendacao_Treino.py`, são salvas **as duas** recomendações do atleta — a do Gemini (`origem="gemini"`) e a por regras (`origem="regras"`). O gatilho é o clique em "Recomendar via Gemini" (não o render da página), evitando duplicatas a cada rerun do Streamlit. O CSV é semeado com a primeira recomendação do atleta `5151970` (longão, 16 km, pace 5.35, 85.6 min).
**Consequências:** `origem` distingue gemini/regras no histórico. Sem controle de concorrência (append reescreve o arquivo) — aceitável para o demo mono-usuário. Clicar Gemini várias vezes gera múltiplos registros (cada um com seu timestamp), por design. Quando o backend (SPEC 6.6) assumir persistência, este CSV vira apenas conveniência de demo.

---

### ADR-014 — Página de Histórico de recomendações no Streamlit
**Status:** Aceita
**Contexto:** Complemento de ADR-011 e ADR-013: expor o histórico persistido na vitrine Streamlit.
**Decisão:** Nova página [app/pages/3_Historico_Recomendacoes.py](app/pages/3_Historico_Recomendacoes.py): tabela de recomendações com filtro por tipo e busca por ID do atleta, seção de detalhe (expander por recomendação, com justificativa e alertas) e botão de download do CSV. **Não** usa `@st.cache_data` para refletir imediatamente novas recomendações salvas (o CSV é pequeno; leitura por rerun é barata).
**Consequências:** Escopo atual do app passa a ser: Recomendação de Treino, Atletas e Histórico de Recomendações.

---

### ADR-015 — Portar os notebooks para telas do Streamlit
**Status:** Aceita
**Contexto:** ADR-011 introduziu o Streamlit como vitrine, mas apenas com Recomendação de Treino, Atletas e Histórico. Os notebooks (`notebooks/`) concentram os gráficos e análises importantes do TCC (EDA, clusterização, recomendação por cluster, comparação de modelos) e são ruins para apresentação. A usuária pediu para **trazer os notebooks para o Streamlit**, separando por telas na mesma divisão do Jupyter, sem remover os notebooks.
**Decisão:** Criar **uma página Streamlit por notebook**, espelhando as seções (headers) de cada um:
- `app/pages/2_Visualizacao_Dataset.py` ← `notebooks/visualizacao_dataset.ipynb` (resumo estatístico, impacto da normalização, painel EDA, relações dos modelos, gráficos individuais, posição multidimensional do atleta — radar + coordenadas paralelas).
- `app/pages/3_Comparacao_Modelos.py` ← `notebooks/comparacao_modelos.ipynb` (tabela comparativa, scatter de regressão pace/tempo, MAE/R², **detalhes por modelo** — coeficientes/importâncias/previsão-exemplo + **relatório de classificação** do RF de treino —, erro de distribuição, F1-macro e matriz de confusão do RF de treino, **teste de recomendação** — parâmetros calibrados + gráfico atual×meta por atleta (`plotar_recomendacao_atleta`) + amostra —, comparação de tipos recomendados e resumo do melhor por categoria).
- `app/pages/4_Categorizacao_Corredores.py` ← `notebooks/categorizacao_corredores.ipynb` (KMeans por gênero k=2, **testes de escolha de features × k** com sugestão automática e curvas de silhouette, tabela descritiva, scatter com centróides, **validação interna completa** — silhouette/Davies-Bouldin/Calinski-Harabasz + estabilidade ARI + grid de k=2..7 + silhouette por atleta —, **validação externa** — ARI/NMI/homogeneidade/completude/V-measure vs perfil por regras + tabelas de contingência —, e fichas de atletas — tabela + visuais + ficha individual).

O notebook `recomendacao_treino.ipynb` (recomendação **por cluster**) chegou a virar tela e foi **removida** a pedido da usuária; **reintroduzida em ADR-021** (`7_Recomendacao_por_Cluster.py` + loaders). A página `6_Recomendacao_Treino.py` segue com regras + Gemini (ADR-010/011) e, após ADR-021, também exibe a recomendação por cluster do atleta.

**Ordem do menu (prefixo numérico do arquivo):** ver ADR-021 (ordem atualizada).

Novos loaders cacheados em `app/shared.py` (`carregar_df_raw`, `perfil_multidim`, `clusters_por_genero`, `tabela_clusters`, `fichas_atletas`, `executar_modelos`, e — para a validação de clusters — `comparacao_features_k`, `grid_k_por_genero`, `validacao_clusters`, `silhouette_amostras`, reaproveitando `comparar_features.py` e `validacao_cluster.py`) **reaproveitam** as funções de `dataset/` e mantêm os imports pesados (sklearn/xgboost/matplotlib) **dentro das funções** (ADR-012), para não penalizar a página de Atletas.
**Reuso de gráficos:** funções de plot que **retornam** `Figure`/`Axes` (as de `dataset/analise.py`, `visualizacao_fichas.py`, e os helpers `plotar_scatter_regressao`/`plotar_matriz_confusao`) são chamadas direto e passadas a `st.pyplot`. Funções que fazem `plt.show()` internamente e retornam `None` são capturadas via `st.pyplot(plt.gcf())` + `plt.close("all")`. Blocos que só imprimiam (`resumo_melhores`, `exibir_detalhes_modelo`) foram reconstruídos em componentes nativos do Streamlit (`st.metric`, `st.dataframe`).
**Consequências:** Escopo do app passa a ter **7 telas** (Home + 6 páginas). Os notebooks seguem mantidos como fonte exploratória (decisão da usuária em ADR-011). A página de Comparação de Modelos treina os 8 modelos no primeiro acesso (cacheado por `@st.cache_data`).

---

### ADR-016 — Cache do Streamlit persistido em disco para telas pesadas
**Status:** Aceita
**Contexto:** O `@st.cache_data` padrão guarda o resultado **apenas em memória**: a cada restart do servidor (`streamlit run`, salvar um arquivo, etc.) o cache é perdido e tudo recomputa. Medições confirmaram o custo por restart: `executar_modelos` (8 modelos, inclui XGBoost) ≈ **26 s** e `clusters_por_genero` + fichas (import do sklearn) ≈ **2,8 s**. A usuária apontou que telas estáveis (ex.: Visualização do Dataset) não deveriam recarregar do zero.
**Decisão:** Adicionar `persist="disk"` a **todos** os loaders cacheados de `app/shared.py`. Sob `streamlit run`, o servidor usa `LocalDiskCacheStorageManager` (grava em `.streamlit/cache/`), então o resultado **sobrevive a restarts**: a 1ª execução paga o custo; as seguintes desserializam do disco. Cobertura por tela:
- **Comparação de Modelos:** `executar_modelos` (8 modelos, ~11 MB serializados).
- **Categorização — KMeans:** `clusters_por_genero`, `tabela_clusters`, `fichas_atletas` + loaders de análise/validação (`comparacao_features_k`, `grid_k_por_genero`, `validacao_clusters`, `silhouette_amostras`).
- **Atletas:** `tabela_atletas` (e dependências `carregar_df`, `tabela_perfil_regras`).
- **Visualização / Recomendação:** `perfil_multidim`, `carregar_df_raw`, `listar_atletas`.
Confirmado que todos os retornos são picláveis (requisito do `persist="disk"`).
**Consequências:**
- `.streamlit/cache/` foi adicionado ao `.gitignore` (não versionar cache).
- O Streamlit **invalida** o cache automaticamente quando o código da função (ou args) muda — durante desenvolvimento ativo isso força recomputar; em uso normal (só reiniciar o servidor) o cache é reaproveitado.
- Se o CSV base mudar sem mudar o código, o cache **não** invalida sozinho (a chave é o caminho + hash da função); limpar via menu "Clear cache" do Streamlit ou apagando `.streamlit/cache/`.
- O `AppTest`/execução fora do servidor usa `MemoryCacheStorageManager` (fallback), então a persistência só é observável sob `streamlit run` — comportamento confirmado no código-fonte do Streamlit (`create_default_cache_storage_manager` → `LocalDiskCacheStorageManager`).

---

### ADR-017 — Protocolo experimental: particionamento por atleta, IC e testes pareados
**Status:** Aceita
**Contexto:** A avaliação dos 8 modelos tinha três problemas que impediam qualquer afirmação defensável:
1. **Particionamento inadequado.** Todos usavam `train_test_split` aleatório **por corrida**, colocando o mesmo atleta em treino e teste. Como as features são relativas ao atleta (pace relativo, FC relativa) e as corridas de um atleta são correlacionadas, a métrica media interpolação dentro de atletas conhecidos — não generalização para um atleta novo, que é o caso de uso do SPEC 6.6.
2. **Ausência de incerteza.** Métrica pontual por modelo, sem IC nem teste; "o modelo X é melhor" não era verificável.
3. **Ausência de baseline e de ablação** para a regra de ADR-007.

**Decisão:** Novo pacote [dataset/avaliacao/](dataset/avaliacao/) (protocolo) e [dataset/experimentos/](dataset/experimentos/) (experimentos), com imports lazy (ADR-012):
- `particionamento.py` — `folds_por_atleta` (`GroupKFold`, nenhum atleta nos dois lados), `split_temporal` e `split_temporal_por_atleta`.
- `estatistica.py` — IC por **bootstrap de atletas** (as corridas não são independentes, então reamostra-se atleta), **Wilcoxon pareado** (cancela a variação entre atletas, que domina a diferença entre modelos) e **correção de Holm** (comparar 8 modelos gera 28 testes).
- `avaliador.py` — predições **out-of-fold** para todas as corridas, dando métrica global honesta e um erro por atleta, que é a unidade dos testes pareados. `avaliar_protocolo_antigo` reproduz o split antigo só para medir o otimismo.
- `registro_modelos.py` — `EspecModelo` para os 8 modelos. Não reaproveita as funções `treina_*` porque cada uma faz o próprio split internamente; elas seguem servindo à demonstração e ao Streamlit (ADR-008/ADR-011).
- `experimentos/protocolo_recomendacao.py`, `baselines.py`, `ablacao.py` — avaliação de recomendadores por **origem rolante**, 7 baselines e ablação dos 4 componentes de ADR-007.

`recomendar_proximo_treino` ganhou `pesos` e `usar_antifadiga` (padrões preservam o comportamento de ADR-007).

**Resultados medidos (fonte primária, 114 atletas, 40.534 corridas):**
- **Otimismo do protocolo antigo:** +0,058 a +0,089 min/km de MAE nos modelos de pace. Mais relevante que o viés: o desvio entre folds sobe de 0,009 para 0,067, ou seja o modelo generaliza de forma desigual para atletas novos e o protocolo antigo escondia isso.
- **Uma conclusão antiga cai:** Random Forest (pace) vs Regressão Linear (pace) dá `p_holm = 0,399` — **não são distinguíveis**. O vencedor real em pace é o `RF Regressor` (`p_holm = 2,1e-7`); em tempo, o XGBoost (`p_holm = 2,7e-8`).
- **Alvo circular:** o RF de `tipo_treino` prevê um rótulo que é função determinística das próprias features (ADR-004). F1-macro 0,908 não é evidência preditiva, e o `registro_modelos` carrega essa ressalva no campo `observacao`.
- **Baselines (555 casos):** a regra de ADR-007 fica em F1-macro 0,142, **abaixo** de `distribuicao_atleta` (0,341), `repetir_ultimo` (0,316) e `aleatorio` (0,233).
- **Ablação:** remover `PESOS_DEFICIT` muda o F1-macro em +0,0008 (**componente inerte**); remover a regra anti-fadiga **melhora** em +0,005; a janela de 2 semanas supera a de 6 em +0,0275, e a de 12 é a pior.

**Consequências:** Nova dependência explícita de `scipy` (já vinha via sklearn) e `pytest`. Executável por `python -m dataset.experimentos protocolo|baselines` e também pela tela `9_Protocolo_Experimental.py` (ADR-020/021). `SEED`, `N_FOLDS`, `N_BOOTSTRAP` e `ALFA` entraram em `constants.py` (ADR-002).

**Otimização necessária.** A avaliação dos dois classificadores levava ~200 s cada, tornando a tela inviável. A causa: `f1_score` sobre arrays de **string** refaz `np.unique` em strings a cada chamada, e o bootstrap chama a métrica 2000 vezes. `_codificar` mapeia os rótulos para inteiros uma única vez — 210 s → 20 s e 193 s → 16 s, com F1 idêntico (0,9084 e 0,8596). `avaliar` e `avaliar_todos` ganharam `n_reamostras` para permitir ICs mais baratos em contexto interativo.

> ⚠️ **O que o protocolo de recomendação mede:** origem rolante mede **imitação** (acerta o próximo tipo que o atleta de fato fez), não qualidade do conselho — um atleta que treina mal é imitado com sucesso por um sistema que recomenda mal. Medir qualidade exige desfecho, que é ADR-018.

---

### ADR-018 — Fonte externa com desfecho de prova (long-distance-running-dataset)
**Status:** Aceita
**Contexto:** O dataset primário não tem **variável de desfecho**: a recomendação só podia ser avaliada contra a própria premissa (erro em relação à `META`), o que é circular. Sem desfecho não há como afirmar que o projeto é útil.
**Decisão:** Adotar como **fonte de validação externa** o Kaggle [mexwell/long-distance-running-dataset](https://www.kaggle.com/datasets/mexwell/long-distance-running-dataset) (CC BY 4.0): 10,7 M registros de 36.412 atletas em 2019-2020, com `age_group`, `country` e `major` (maratonas corridas). Ela **não substitui** o dataset primário (ADR-003), que segue sendo a base do projeto.

Novo pacote [dataset/fontes/](dataset/fontes/) com `FonteDados` (schema canônico + metadados `tem_fc`/`tem_elevacao`) e um adaptador por fonte. Diferenças tratadas no adaptador: km→m, min→s, data ISO→`%d/%m/%Y %H:%M`, e descarte dos dias de descanso (painel denso, ~65% das linhas com `distance = 0`).

**Pipeline tornado tolerante a FC ausente.** A fonte externa não tem FC nem elevação, e o pipeline de ADR-003 **zerava o dataset**: sem FC a mediana é nula, e `FC > 40` em nulo resulta em nulo, descartando tudo. `preprocessamento.py` ganhou `tem_frequencia_cardiaca` (detecta coluna ausente, toda nula **ou constante**) e propaga `usar_fc` — com FC ausente, `intervalado` cai para o critério de pace apenas. Verificado sem regressão na fonte primária (40.534 corridas, distribuição idêntica).

**Desfecho — meia maratona como principal.** [experimentos/desfecho.py](dataset/experimentos/desfecho.py):
- **21,0975 km (principal)** — melhor tempo do atleta na distância. Escolhida porque o público do projeto é o corredor amador (seção 1) e 21 km é a distância de prova dominante nesse público.
- **42,195 km (robustez)** — cruzando `major` com a data oficial da prova, o esforço é uma **corrida verificada**. Validado: 4.294 dos 4.365 inscritos em Boston 2019 têm ~42 km em 15/04/2019, e o ordenamento entre provas confere com a realidade (Boston, que exige índice, é a mais rápida com 3h36; Chicago e NY, por sorteio, as mais lentas).

**Resultados medidos (6.821 atletas com desfecho de 21 km):** tempo mediano 101 min, p10 85, p90 129 — distribuição amadora plausível. Treino prediz desempenho com **R² = 0,617** (linear) e 0,619 (RF). A **composição** de treino acrescenta informação além de volume e pace: ΔR² = +0,016, MAE −0,17 min, `p_holm = 1,7e-8`; confirmado com RF (`p_holm` ainda menor), então não é artefato de linearidade. `prop_leve` correlaciona −0,097 com o tempo e `prop_intervalado` +0,146 — mais corrida leve associada a prova mais rápida, na direção que a `META` supõe.

**Consequências:** Os arquivos (2,3 GB) **não são versionados**; `scripts/baixar_long_distance.sh` baixa do endpoint público do Kaggle (dispensa credenciais) para `$DIRETORIO_DADOS_EXTERNOS` (padrão `~/.cache/tcc-back-2/dados`, fora do OneDrive para não sincronizar 1 GB). Modelos que exigem FC/elevação são filtrados por `EspecModelo.aplicavel`. **Limitação registrada:** o melhor esforço de 21 km pode ter sido um treino intenso, não uma prova — é proxy de capacidade, não resultado oficial.

---

### ADR-019 — Meta de distribuição aprendida do desfecho: método aceito, números não
**Status:** Aceita quanto ao **método e ao diagnóstico**; **rejeitada** quanto a substituir os valores de `META`
**Contexto:** A `META` de ADR-007 foi afirmada, não medida. Com o desfecho de ADR-018 é possível estimá-la: qual distribuição de tipos se associa ao melhor tempo de 21 km, mantendo volume e pace constantes (senão a resposta seria "treine mais e mais rápido", verdadeira e inútil).
**Decisão (método):** [experimentos/aprender_meta.py](dataset/experimentos/aprender_meta.py) trata o problema como **experimento de mistura** (as 4 proporções somam 1) e usa o **polinômio quadrático de Scheffé** — termos lineares mais os 6 produtos cruzados. Um modelo linear nas proporções tem mínimo necessariamente num **vértice** do simplexo, isto é, só consegue dizer "faça 100% de um tipo"; os cruzados permitem ótimo interior. Otimização por SLSQP no simplexo, com múltiplos inícios, `Ridge` para conter a colinearidade estrutural, e IC por bootstrap que reajusta **e reotimiza** em cada reamostra. Duas variantes: **irrestrita** (diagnóstico) e **factível**, restrita ao intervalo p5-p95 observado — a única utilizável, porque um atleta precisa poder alcançar a meta.

**Resultados e por que os números foram rejeitados:**
1. **Confirmado e robusto — a `META` é inalcançável.** `leve = 60%` cai **fora** da região observada [0,115; 0,475]; os outros três componentes estão dentro. É a causa raiz do defeito de ADR-007, agora medida em dataset independente.
2. **Confirmado — aderência a *uma* meta sensata prediz desempenho.** Fora da amostra, a correlação entre aderência e tempo de prova é 0,120 (ADR-007) e 0,099 (aprendida), contra 0,029 da meta uniforme. Ou seja, a estrutura da `META` captura algo real; a uniforme não.
3. **Rejeitado — os valores específicos não são identificáveis.** Com n = 1.686 a meta aprendida vencia fora da amostra (0,182 vs 0,145); com n = 6.821 ela **perde** (0,099 vs 0,120), e os desvios entre folds (0,033 e 0,040) cobrem a diferença. O IC bootstrap do componente `longao` vai de 0,130 a 0,417 — amplitude maior que a própria diferença em relação a ADR-007. Apenas `leve` é bem identificado, e está colado na fronteira.
4. **Diagnóstico da superfície:** o ótimo permanece na fronteira mesmo com os termos de Scheffé, então a monotonicidade é propriedade dos dados, não artefato de linearidade. Dentro do observável, "o máximo de corrida leve que o classificador permite" é o melhor — consistente com a literatura de treino polarizado (80/20) e com o sinal da `META`.

**Conclusão registrada:** a direção de ADR-007 está certa (mais leve, menos intervalado); o erro é o **nível** de `leve`, fixado num valor que o classificador de ADR-004 não pode produzir. A correção defensável é tornar a meta alcançável — derivá-la da região factível em vez de afirmá-la —, **não** trocar 60/20/12/8 por outros quatro números de precisão falsa. `MetaAprendida.bem_identificada` e `na_fronteira` expõem isso na saída, para que o número não seja lido como mais preciso do que é.

**Consequências:** `META` em `model.py` **não foi alterada** — trocá-la sem identificação estatística seria substituir um chute por outro. As opções de correção (recalibrar `LIMIAR_DIST_LONGAO`, ou definir a meta sobre a região factível) ficam para uma decisão futura. **Natureza observacional:** associação, não causalidade — atletas mais rápidos podem treinar diferente *porque* são mais rápidos. O que os dados estabelecem é o mais fraco e ainda útil: se nem a associação existisse, a recomendação não teria base.

---

### ADR-020 — Experimentos como telas do Streamlit, preservando os números antigos
**Status:** Aceita
**Contexto:** Os experimentos de ADR-017/018/019 só existiam como CLI. Pior: a tela de Comparação de Modelos (ADR-015) continuava exibindo as métricas do protocolo antigo — as que ADR-017 mostrou serem otimistas — sem nenhuma indicação disso, o que é a pior configuração possível para uma apresentação de TCC.
**Decisão:** Duas telas novas, seguindo a divisão natural dos experimentos, e um aviso cruzado:
- **`9_Protocolo_Experimental.py`** (itens 1 e 2; arquivo era `8_…` antes de ADR-021) — três abas: modelos sob particionamento por atleta com gráfico do otimismo por modelo; baselines vs ADR-007; ablação dos componentes. Destaca automaticamente os pares **sem** diferença estatística, as derrotas significativas de ADR-007 e os componentes inertes — os achados aparecem como conclusão, não como tabela para o leitor interpretar.
- **`10_Desfecho_e_Meta.py`** (itens 3 e 4; arquivo era `9_…` antes de ADR-021) — três abas: o desfecho de meia maratona (histograma + dispersão treino×desempenho); a comparação de modelos encaixados; e a meta aprendida, com um gráfico que sobrepõe a `META` de ADR-007 à região factível observada, tornando visual o defeito estrutural.
- **`3_Comparacao_Modelos.py`** ganhou um `st.warning` explicando que suas métricas usam o protocolo original e apontando para a tela nova. Os números **não foram substituídos**: a tela é porte de notebook (ADR-015) e a paridade é intencional.

**Preservar o antigo em vez de corrigir no lugar** é deliberado: mostrar as duas medições lado a lado, com a diferença rotulada, é mais defensável numa banca do que trocar silenciosamente os números.

**Decisões de implementação:**
- Loaders em `app/shared.py` com `@st.cache_data(persist="disk")` (ADR-016) e imports pesados dentro das funções (ADR-012). Devolvem apenas DataFrames e tipos primitivos, não os objetos `ResultadoAvaliacao`/`MetaAprendida`, para manter a serialização em disco trivial.
- `fonte_externa_disponivel` **não** é cacheada de propósito: é uma checagem de arquivo barata, e cachear faria a tela seguir dizendo "indisponível" depois do download.
- A tela de Desfecho degrada com instrução de download quando a base externa (2,3 GB, não versionada) está ausente, em vez de quebrar. Ambos os caminhos foram verificados com `streamlit.testing.v1.AppTest`.
- Seletor lateral de número de atletas na tela de Desfecho (3.000 a 36.412), porque poder estatístico e tempo de primeira carga são um tradeoff que depende do uso.

**Consequências:** Na época, o app passou a ter **9 telas**. Ordem atual do menu: ver ADR-021. `tests/test_paginas_streamlit.py` é teste de fumaça: compila as páginas e confere que os símbolos importados de `app.shared` existem — não executa os loaders, que são caros demais para teste de unidade.

---

### ADR-021 — Reintroduzir recomendação de treino por cluster no Streamlit
**Status:** Aceita
**Contexto:** ADR-015 removeu a tela de recomendação por cluster a pedido da usuária. O módulo [recomendacao_cluster.py](dataset/models/previsao_treino/recomendacao_cluster.py) e o notebook `recomendacao_treino.ipynb` permaneceram, mas a demo não mostrava a personalização por perfil KMeans. A usuária pediu para **re-adicionar** essa recomendação.

**Decisão:**
- Nova página [app/pages/7_Recomendacao_por_Cluster.py](app/pages/7_Recomendacao_por_Cluster.py) — **mesmo fluxo da tela 6**: campo de ID → perfil → distribuição vs meta do cluster → próximo treino (tipo, distância, pace, tempo). Comparação com a recomendação global fica num expander opcional.
- Loaders em `app/shared.py` (cache disco, ADR-016): `recomendacoes_por_cluster`, `comparacao_global_vs_cluster`; helpers `recomendacao_cluster_atleta`, `meta_do_cluster`, `distribuicao_vs_meta_cluster`.
- Página [6_Recomendacao_Treino.py](app/pages/6_Recomendacao_Treino.py) também exibe a recomendação por cluster ao lado de regras e Gemini (mesmo atleta).
- Renumeração para abrir o slot 7: Histórico `7→8`, Protocolo `8→9`, Desfecho `9→10`.

`gerar_recomendacoes_por_cluster` usa `META_POR_PERFIL` e calibração no histórico do cluster. *(ADR-024: a meta global deixou de ser regra de produto; cluster é o único caminho de recomendação.)*

**Consequências:** App passa a ter **10 telas**. Ordem: 1) Home, 2) Visualização, 3) Comparação de Modelos, 4) Categorização, 5) Atletas, 6) Recomendação de Treino, 7) Recomendação por Cluster, 8) Histórico, 9) Protocolo Experimental, 10) Desfecho e Meta.

---

### ADR-022 — Protocolo Experimental centrado na validação da recomendação
**Status:** Aceita
**Contexto:** A tela `9_Protocolo_Experimental.py` (ADR-020) abria com tabelas densas dos 8 modelos de pace/tempo; a validação da **recomendação** (produto do TCC) ficava na segunda aba, sem matriz de confusão, sem exemplos de casos e sem a variante por cluster. A usuária pediu maior clareza e dados de validação da recomendação.

**Decisão:**
- Reordenar a tela: (1) **Validação da recomendação**, (2) Ablação, (3) Modelos auxiliares (pace/tempo).
- Conclusão em 30 segundos no topo (métricas-chave da ADR-007 + melhor baseline + aviso de imitação × desfecho).
- Extender `comparacao_baselines` em `app/shared.py` para devolver `matriz_*`, `amostra_*` e `veredito`.
- Incluir `cluster (ADR-021)` no ranking via `baselines_com_cluster` / `recomendador_cluster` em [baselines.py](dataset/experimentos/baselines.py) (mesma regra de déficit com `META_POR_PERFIL`).
- Helpers `matriz_confusao` e `tabela_amostra_casos` em [protocolo_recomendacao.py](dataset/experimentos/protocolo_recomendacao.py).

**Consequências:** A primeira carga da aba de recomendação passa a depender também do KMeans (mapa atleta→perfil_cluster), reaproveitando o cache de `clusters_por_genero` (ADR-016). A CLI `python -m dataset.experimentos baselines` continua usando `baselines_padrao` (sem cluster), a menos que seja estendida depois; a UI é a superfície completa.

---

### ADR-023 — Demo Streamlit: recomendação só por cluster; remover telas dedicadas
**Status:** Substituída por ADR-024 (quanto à coexistência da meta global no Protocolo)
**Contexto:** A usuária pediu para simplificar a vitrine: uma única tela de recomendação (por cluster), sem a página dedicada de cluster e sem a tela de Desfecho/Meta (os experimentos de desfecho seguem disponíveis via CLI).

**Decisão:**
- Removidas as páginas `7_Recomendacao_por_Cluster.py` e `10_Desfecho_e_Meta.py`.
- [6_Recomendacao_Treino.py](app/pages/6_Recomendacao_Treino.py) passa a ser **só por cluster** (remove a seção de déficit vs meta global). Ao informar o ID, mostra também **posição relativa** (`tabela_percentis_atleta` + `plotar_painel_posicao_atleta`) e a **ficha/card do cluster** (`plotar_ficha_atleta`).
- Em **Próximo treino recomendado**, a demo exibe justificativa descritiva (`explicar_recomendacao`) — tipo (déficit vs meta do cluster + anti-fadiga), distância (faixa calibrada no cluster × histórico do atleta na janela), pace alvo (média da janela + offset) e tempo (distância × pace).
- Gemini permanece; ao salvar histórico, grava `origem="cluster"` (não mais regras globais).
- Renumeração: Histórico `8→7`, Protocolo `9→8`. App fica com **8 telas** (Home + 7 páginas).

**Consequências:** Loaders de desfecho (`desfecho_meia`, `meta_aprendida`, …) permanecem em `app/shared.py` e nos experimentos CLI; só saem do menu Streamlit. *(ADR-024 remove a meta global também do Protocolo e dos baselines de produto.)*

### ADR-024 — Recomendação de treino exclusivamente por cluster
**Status:** Aceita
**Contexto:** A demo ainda citava “meta global fixa” e o Protocolo/baselines tratavam `déficit vs meta global` (ADR-007 / `META` 60/20/12/8) como regra do projeto. A usuária pediu para remover totalmente a regra global: **toda** recomendação de treino usa só o cluster.

**Decisão:**
- Regra de produto = `gerar_recomendacoes_por_cluster` / `recomendador_cluster` com `META_POR_PERFIL` (`volume_alto` / `volume_moderado`).
- Fallback sem `perfil_cluster` = `META_CLUSTER_PADRAO` (`volume_moderado`) — **nunca** a `META` global.
- Removido `déficit vs meta global` de `baselines_padrao` / ranking do Protocolo; confrontos e veredito centram em `déficit vs meta do cluster`.
- Gemini (`montar_features_atleta`) e `model_main` usam meta/params do cluster.
- Ablação usa meta de referência `volume_moderado` (não a `META` 60/20/12/8).
- A constante `META` em `model.py` permanece só para experimentos históricos de desfecho/meta aprendida (ADR-018/019) e para `treina_modelo` na tela Comparação de Modelos (paridade com notebooks) — **não** é caminho de recomendação de produto.

**Consequências:** Protocolo e CLI `baselines` validam a regra por cluster. Textos da UI não contrastam mais com “meta global”.

### ADR-025 — Renomear tela “Protocolo Experimental” → “Validação da recomendação”
**Status:** Aceita
**Contexto:** O nome “Protocolo Experimental” era jargão de ADR-017 e não explicava a tela para quem abre o app.

**Decisão:**
- Arquivo `8_Protocolo_Experimental.py` → `8_Validacao_da_Recomendacao.py`.
- Título: **Validação da recomendação**; subtítulo **A regra funciona?**; lead **Experimentos — recomendação vs baselines**.
- Aba 1 renomeada para **A regra vs baselines** (conteúdo igual).
- Home, Comparação de Modelos, FAQ e `GUIA-EXPERIMENTOS.md` atualizados.

**Consequências:** O menu lateral Streamlit passa a mostrar “Validacao da Recomendacao”. A lógica/loaders não mudam.

---

## 4. Convenções

- **Idioma do código:** nomes de funções, variáveis e domínio em **português** (padrão atual do repo: `treina_modelo`, `montar_perfil_atletas`, `calibrar_parametros`). Docstrings em português.
  > Nota: o SPEC do backend orquestrador (seção 5, Princípio 10) prescreve *código em inglês* para aquele projeto. **Este repo (`tcc-back-2`) usa português** — não misturar as convenções entre repositórios.
- **Reprodutibilidade:** `random_state=42` / `seed=42` em tudo que envolve aleatoriedade (constante `SEED` em `constants.py`).
- **Avaliação (ADR-017):** particionamento sempre **por atleta** (`GroupKFold`), nunca aleatório por corrida. Toda métrica reportada com IC; toda comparação entre modelos com teste pareado e correção para múltiplas comparações.
- **Fontes de dados (ADR-018):** código novo consome `FonteDados` de `dataset/fontes/`, não caminhos de arquivo. Features que dependem de FC ou elevação declaram `exige_fc`/`exige_elevacao`.
- **Testes:** módulos novos de avaliação e experimentos têm teste de unidade em `tests/`.
- **Formatação de pace:** usar `formatar_pace` (decimal → `min:seg/km`), nunca formatar manualmente.
- **Gráficos:** matplotlib; paletas de cor por tipo de treino já definidas (`CORES_TIPO`).
- **Timestamps:** formato `"%d/%m/%Y %H:%M"` no dataset.

---

## 5. Dúvidas Recorrentes Resolvidas (FAQ para a IA)

| Pergunta | Resposta |
|----------|----------|
| Onde ficam os nomes de coluna e thresholds? | Sempre em [constants.py](dataset/constants.py). |
| Como rodar o projeto? | `python -m dataset.models.model_main` (ver [readme.md](readme.md)). Notebooks em `notebooks/`. |
| Quantas corridas mínimas para gerar perfil? | `MIN_CORRIDAS_PERFIL = 20`. |
| Qual a meta de distribuição de treinos? | Por perfil_cluster em `META_POR_PERFIL` (ADR-024): `volume_alto` / `volume_moderado`. A antiga `META` global (60/20/12/8) não é mais usada na recomendação. |
| Regras ou ML para recomendação? | Produto = **cluster** (`gerar_recomendacoes_por_cluster`, ADR-024). Alternativa via LLM = **Gemini** (ADR-010, features com meta do cluster). RF existe mas está desativado. |
| Este projeto tem API HTTP? | Ainda não (ver ADR-009). |
| Como recomendar treino via Gemini? | `python -m gemini.recomendacao <athlete_id>` (requer `GEMINI_API_KEY`). Ver ADR-010 e [gemini/](gemini/). |
| Onde configurar a chave do Gemini? | `.env` (copiar de `.env.example`); lido por `GeminiConfig.from_env()`. |
| Como clonar e rodar em outra máquina? | `bash run.sh` (macOS/Linux) ou `.\run.ps1` (Windows). Cria venv, instala `requirements.txt`, gera `.env` e sobe o Streamlit. |
| Como avaliar um modelo corretamente? | `dataset.avaliacao.avaliar` com `EspecModelo` — **nunca** `train_test_split` por corrida (ADR-017). Particionamento é por **atleta**. |
| Como rodar os experimentos? | CLI: `python -m dataset.experimentos protocolo\|baselines\|desfecho\|meta\|tudo` (os dois últimos exigem a fonte externa). No app: tela **Validação da recomendação** (itens 1–2); desfecho/meta só via CLI (ADR-023). |
| Onde está a validação da recomendação? | Tela **Validação da recomendação** (`8_Validacao_da_Recomendacao.py`, ADR-022/025): ranking vs baselines, matriz, amostra. Desfecho de prova → CLI `desfecho`/`meta`. |
| Por que a tela de Comparação de Modelos mostra números diferentes da tela de Protocolo? | De propósito (ADR-020). A primeira é porte do notebook e usa o split antigo; a segunda usa o protocolo corrigido de ADR-017. As duas coexistem com a diferença rotulada. |
| Quantas telas o app tem? | 8 (ADR-023/025): Home, Visualização, Comparação de Modelos, Categorização, Atletas, Recomendação de Treino, Histórico, Validação da recomendação. |
| A recomendação usa os clusters? | Sim — **só** cluster (ADR-024): demo, Gemini, Protocolo e `model_main`. |
| Onde fica a fonte externa de validação? | `$DIRETORIO_DADOS_EXTERNOS` (padrão `~/.cache/tcc-back-2/dados`); baixe com `bash scripts/baixar_long_distance.sh` (ADR-018). Não versionada, 2,3 GB. |
| A regra de ADR-007 funciona? | Não como está: emite `leve` em 76% dos casos e perde de `sempre_leve` no protocolo de ADR-017. A causa é a `META` inalcançável (ADR-019). |
| Posso trocar os valores de `META`? | Não sem identificação estatística. ADR-019 mostrou que a meta ótima não é identificável (IC de `longao` de 0,13 a 0,42). O problema é o **nível** de `leve`, não os quatro números. |
| O projeto tem variável de desfecho? | Sim, na fonte externa: melhor tempo de **21 km** (principal, público amador) e maratona verificada via `major` (robustez). Ver ADR-018. |
| Posso rodar o pipeline sem FC? | Sim (ADR-018). `preprocessar` detecta FC ausente/constante e adapta; `intervalado` passa a usar só pace. |
| Como rodar os testes? | `pytest tests/` (ou `python -m pytest tests/`). |
| Onde entender os experimentos sem jargão? | [GUIA-EXPERIMENTOS.md](GUIA-EXPERIMENTOS.md) — conclusões em português simples, glossário estatístico, o que cada tela responde e perguntas de banca. Este ADR é o registro do *porquê*; o guia é a explicação. |

---

## 6. Histórico de Atualizações

| Data | Autor | Mudança |
|------|-------|---------|
| 2026-09-21 | IA (Claude) | Criação inicial do ADR a partir da análise do código e do SPEC do backend. ADR-001 a ADR-009. |
| 2026-09-21 | IA (Claude) | ADR-010: recomendação de treino via API do Gemini (pacote `gemini/` na raiz). Atualização de FAQ e convenções. |
| 2026-09-21 | IA (Claude) | ADR-011: app Streamlit em `app/` como camada de demonstração (só a página de recomendação nesta primeira versão). Notebooks mantidos. |
| 2026-09-21 | IA (Claude) | Modelo padrão do Gemini atualizado para `gemini-3.6-flash` (o `gemini-2.5-flash` foi descontinuado para novos usuários — 404). Página de recomendação passa a carregar vazia com campo de ID do atleta. |
| 2026-09-21 | IA (Claude) | Nova página `app/pages/2_Atletas.py` (tabela + filtros + busca) e planilha `output/atletas.csv` via `exportar_planilha_atletas`. |
| 2026-09-21 | IA (Claude) | `GeminiClient` ganha retry com backoff em erros transitórios (429/5xx, ex.: 503 UNAVAILABLE), configurável por `GEMINI_MAX_TENTATIVAS`/`GEMINI_ESPERA_BASE_S`. Logger do SDK silenciado (aviso AFC benigno). Streamlit: `use_container_width` → `width="stretch"`. |
| 2026-09-21 | IA (Claude) | Adicionados `requirements.txt` e scripts `run.sh`/`run.ps1` para clonar e rodar em um comando. Readme com seção de setup rápido. |
| 2026-09-21 | IA (Claude) | ADR-012: imports lazy (sklearn/matplotlib) para acelerar a página de Atletas do Streamlit (de ~25 s/~5,5 s para ~1 s). `dataset/categorizar/__init__.py` via `__getattr__` (corrige bug do `__all__`); matplotlib/numpy movidos para dentro das funções de plot em `model.py` e `app/shared.py`. |
| 2026-09-21 | IA (Claude) | ADR-013 e ADR-014: persistência do histórico de recomendações da IA em `output/recomendacoes.csv` (módulo `gemini/historico.py`, salvo automaticamente ao gerar via Gemini) e nova página `app/pages/3_Historico_Recomendacoes.py`. CSV semeado com a 1ª recomendação do atleta 5151970. |
| 2026-09-21 | IA (Claude) | ADR-013: salvamento automático passa a gravar **as duas** recomendações (Gemini + por regras, com justificativa/alertas) a cada geração via Gemini. `salvar_recomendacao` aceita `RecomendacaoTreino` ou dict de regras. |
| 2026-09-23 | IA (Claude) | ADR-015: portados os 4 notebooks para telas do Streamlit (`4_Visualizacao_Dataset`, `5_Categorizacao_Corredores`, `6_Recomendacao_por_Cluster`, `7_Comparacao_Modelos`), com novos loaders cacheados em `app/shared.py`. App passa a ter 7 telas; notebooks mantidos. |
| 2026-09-24 | IA (Claude) | ADR-015 atualizada: removida a página de recomendação **por cluster** (a pedido da usuária) e o loader `recomendacoes_por_cluster`. Páginas renumeradas para a ordem: Visualização (2), Comparação de Modelos (3), Categorização (4), Atletas (5), Recomendação de Treino (6), Histórico (7). |
| 2026-09-25 | IA (Claude) | ADR-016: cache do Streamlit persistido em disco (`@st.cache_data(persist="disk")`) nos loaders pesados/estáveis de `app/shared.py`, para não recomputar KMeans e o treino dos 8 modelos a cada restart. `.streamlit/cache/` adicionado ao `.gitignore`. |
| 2026-09-25 | IA (Claude) | ADR-015: página `4_Categorizacao_Corredores.py` trazida à paridade com o notebook — faltavam os **testes de escolha de features × k**, a **validação interna completa** (silhouette/Davies-Bouldin/Calinski-Harabasz + estabilidade ARI + grid k=2..7 + silhouette por atleta) e toda a **validação externa** (ARI/NMI/homogeneidade/completude/V-measure vs perfil por regras + contingências). Novos loaders em `app/shared.py`: `comparacao_features_k`, `grid_k_por_genero`, `validacao_clusters`, `silhouette_amostras`. |
| 2026-09-25 | IA (Claude) | ADR-016: `persist="disk"` estendido a **todos** os loaders de `app/shared.py` (incluindo `carregar_df_raw` e `listar_atletas`) — cache de Comparação de Modelos, Categorização e Atletas totalmente persistido entre restarts. Retornos verificados como picláveis (`executar_modelos` ~11 MB). |
| 2026-09-25 | IA (Claude) | ADR-015: paridade das demais telas. `4_Categorizacao` ganhou as 2 visualizações restantes do notebook (original vs espaço padronizado com elipses de covariância + boxplots por categoria). `3_Comparacao_Modelos` ganhou os **detalhes por modelo** (`exibir_detalhes_modelo` → coeficientes/importâncias/relatório de classificação) e o **teste de recomendação** (parâmetros calibrados + `plotar_recomendacao_atleta` capturado via `st.pyplot(plt.gcf())`); removida a antiga "Amostra de recomendações" (redundante). `5_Atletas` **não** deriva de notebook (ADR-011) — já reaproveita `tabela_atletas`, sem lacuna. |
| 2026-09-28 | IA (Claude) | **ADR-017**: protocolo experimental corrigido — `dataset/avaliacao/` com `GroupKFold` por atleta, IC por bootstrap de atletas, Wilcoxon pareado e Holm; `dataset/experimentos/` com 7 baselines e ablação de ADR-007. Medido o otimismo do protocolo antigo (+0,06 a +0,09 de MAE em pace) e derrubada a conclusão "RF > LinReg" em pace (`p_holm = 0,399`). `recomendar_proximo_treino` ganhou `pesos`/`usar_antifadiga` para ablação. |
| 2026-09-28 | IA (Claude) | **ADR-007 marcada com defeito estrutural**: a `META` é inalcançável dado o classificador relativo de ADR-004 — déficit de `leve` positivo em 111/111 atletas, 76% das recomendações são `leve`, e a regra perde de `sempre_leve`. Replicado na fonte externa. |
| 2026-09-28 | IA (Claude) | **ADR-018**: fonte externa `mexwell/long-distance-running-dataset` com desfecho de prova; novo pacote `dataset/fontes/` e pipeline tolerante a FC ausente (`tem_frequencia_cardiaca`). Desfecho de 21 km em 6.821 atletas; composição de treino acrescenta informação além de volume/pace (`p_holm = 1,7e-8`). Sem regressão na fonte primária. |
| 2026-09-28 | IA (Claude) | **ADR-019**: meta de distribuição aprendida via polinômio de Scheffé. Método e diagnóstico aceitos (a `META` de `leve = 60%` cai fora da região factível [0,115; 0,475]); **valores rejeitados** — a meta ótima não é identificável (IC de `longao` de 0,130 a 0,417; o vencedor fora da amostra inverte entre n=1.686 e n=6.821). `META` mantida inalterada em `model.py`. |
| 2026-09-28 | IA (Claude) | Adicionados `tests/` (52 testes), `scripts/baixar_long_distance.sh`, CLI `python -m dataset.experimentos`, e `scipy`/`pytest` em `requirements.txt`. Convenções de avaliação, fontes e testes registradas na seção 4. |
| 2026-09-29 | IA (Claude) | **ADR-020**: experimentos refletidos no Streamlit — telas `8_Protocolo_Experimental.py` (itens 1 e 2) e `9_Desfecho_e_Meta.py` (itens 3 e 4), com novos loaders cacheados em `app/shared.py`. `3_Comparacao_Modelos.py` e a Home passam a avisar que as métricas de lá usam o protocolo antigo. App vai a **9 telas**. |
| 2026-09-29 | IA (Claude) | ADR-017: otimização de `_codificar` na avaliação de classificadores (rótulos string → inteiros antes do bootstrap): 210 s → 20 s e 193 s → 16 s, F1 inalterado. Viabilizou a tela 8. `avaliar`/`avaliar_todos` ganharam `n_reamostras`. |
| 2026-09-29 | IA (Claude) | `tests/test_paginas_streamlit.py`: teste de fumaça das 9 telas (compilação + símbolos de `app.shared` + degradação sem a base externa). Total de **59 testes**. |
| 2026-09-29 | IA (Claude) | Criado [GUIA-EXPERIMENTOS.md](GUIA-EXPERIMENTOS.md) a pedido da usuária: os experimentos ficaram difíceis de acompanhar (conceitos estatísticos, telas e conclusões). Conclusões em português simples, glossário com o *porquê* de cada técnica, mapa do que cada tela responde, perguntas de banca com resposta pronta e a lista do que **não** foi provado. As telas 8 e 9 ganharam um expander de glossário no topo. Código e resultados inalterados. |
| 2026-10-08 | IA (Claude) | **ADR-021**: reintroduzida a recomendação por cluster no Streamlit — página `7_Recomendacao_por_Cluster.py`, loaders `recomendacoes_por_cluster` / `comparacao_global_vs_cluster`, seção na tela 6. Histórico/Protocolo/Desfecho renumerados para 8/9/10. App vai a **10 telas**. |
| 2026-10-08 | IA (Claude) | ADR-021: tela 7 simplificada para o fluxo ID → recomendação (igual à tela 6); o dashboard/comparação do notebook deixou de ser a UI principal. |
| 2026-10-09 | IA (Claude) | **ADR-022**: Protocolo Experimental redesenhado — validação da recomendação como aba principal (conclusão, matriz, amostra de casos, cluster no ranking); modelos pace/tempo na última aba. |
| 2026-10-09 | IA (Claude) | UI: nomes descritivos nas telas (`déficit vs meta global` / `déficit vs meta do cluster`) em vez de códigos ADR; métricas de recomendação exibidas em %. |
| 2026-10-12 | IA (Claude) | **ADR-023**: removidas telas Recomendação por Cluster e Desfecho/Meta; Recomendação de Treino fica só por cluster + posição relativa + ficha; Histórico/Protocolo renumerados para 7/8. |
| 2026-10-12 | IA (Composer) | ADR-023: justificativa descritiva do próximo treino (`explicar_recomendacao` em `recomendacao_cluster.py`) na tela 6 — tipo/distância/pace/tempo em linguagem natural. |
| 2026-10-12 | IA (Composer) | **ADR-024**: recomendação exclusivamente por cluster — remove meta global dos baselines/Protocolo/Gemini/`model_main`; fallback `volume_moderado`. |
| 2026-10-12 | IA (Composer) | **ADR-025**: tela `8_Protocolo_Experimental` renomeada para `8_Validacao_da_Recomendacao` — título/subtítulo/lead descritivos. |
| 2026-10-12 | IA (Composer) | ADR-025: glossário visível de **Déficit** e **Baseline** no topo da tela de Validação. |
