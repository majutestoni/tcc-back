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
