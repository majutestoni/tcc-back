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
└── raw-data-kaggle.csv
notebooks/                  # Notebooks Jupyter para visualização/comparação
```

Execução principal: `python -m dataset.models.model_main` (atualmente chama `treina_modelo` — recomendação de treino por regras).

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
**Status:** Aceita (modelo principal atual)
**Contexto:** Recomendar o próximo treino de forma explicável, corrigindo desequilíbrios no plano de treino do atleta.
**Decisão:** [previsao_treino/model.py](dataset/models/previsao_treino/model.py) — `treina_modelo`:
1. Calcula a distribuição de tipos de treino do atleta numa **janela de `SEMANAS_JANELA = 6` semanas**.
2. Compara com a meta `META` (`leve 60%, longao 20%, ritmo 12%, intervalado 8%`).
3. Recomenda o tipo com maior **déficit ponderado** (`PESOS_DEFICIT`).
4. Regra anti-fadiga: evita repetir longão/intervalado logo após intervalado ou longão.
5. Calibra distância/pace por tipo a partir das medianas do histórico (`calibrar_parametros`).
**Consequências:** Existe também `random_forest.py` (`treina_rf_recomendacao`) como alternativa de ML supervisionado, atualmente **desativada** no `model_main.py`. As regras são o baseline interpretável do TCC.

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

O notebook `recomendacao_treino.ipynb` (recomendação **por cluster**) **não** virou tela: a página `6_Recomendacao_por_Cluster.py` chegou a ser criada mas foi **removida a pedido da usuária** (o loader `recomendacoes_por_cluster` em `app/shared.py` foi retirado junto). A recomendação de treino no app segue apenas pela página `6_Recomendacao_Treino.py` (regras + Gemini, ADR-010/011).

**Ordem do menu (prefixo numérico do arquivo):** 1) Home, 2) Visualização do Dataset, 3) Comparação de Modelos, 4) Categorização de Corredores — KMeans, 5) Atletas, 6) Recomendação de Treino, 7) Histórico de Recomendações.

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

## 4. Convenções

- **Idioma do código:** nomes de funções, variáveis e domínio em **português** (padrão atual do repo: `treina_modelo`, `montar_perfil_atletas`, `calibrar_parametros`). Docstrings em português.
  > Nota: o SPEC do backend orquestrador (seção 5, Princípio 10) prescreve *código em inglês* para aquele projeto. **Este repo (`tcc-back-2`) usa português** — não misturar as convenções entre repositórios.
- **Reprodutibilidade:** `random_state=42` / `seed=42` em tudo que envolve aleatoriedade.
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
| Qual a meta de distribuição de treinos? | `META` em [model.py](dataset/models/previsao_treino/model.py): leve 60%, longão 20%, ritmo 12%, intervalado 8%. |
| Regras ou ML para recomendação? | Baseline = **regras** (`treina_modelo`, ADR-007). Alternativa via LLM = **Gemini** (ADR-010). RF existe mas está desativado. |
| Este projeto tem API HTTP? | Ainda não (ver ADR-009). |
| Como recomendar treino via Gemini? | `python -m gemini.recomendacao <athlete_id>` (requer `GEMINI_API_KEY`). Ver ADR-010 e [gemini/](gemini/). |
| Onde configurar a chave do Gemini? | `.env` (copiar de `.env.example`); lido por `GeminiConfig.from_env()`. |
| Como clonar e rodar em outra máquina? | `bash run.sh` (macOS/Linux) ou `.\run.ps1` (Windows). Cria venv, instala `requirements.txt`, gera `.env` e sobe o Streamlit. |

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
