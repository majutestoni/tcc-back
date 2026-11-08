# TCC — Serviço de ML (`tcc-back-2`)

Pipeline de análise e recomendação de treinos para corredores (Python + Streamlit).

Para o **fluxo científico** (coleta → limpeza → cluster → recomendação → validação), veja [FLUXO.md](FLUXO.md).  
Para decisões de arquitetura: [ADR.md](ADR.md). Para a banca: [GUIA-EXPERIMENTOS.md](GUIA-EXPERIMENTOS.md).

---

## Pré-requisitos

- Python **3.11+** (testado com 3.13)
- macOS, Linux ou Windows
- (Opcional) chave [Gemini API](https://aistudio.google.com/apikey) para recomendação por LLM
- (Opcional) ~2,3 GB livres para a base externa de validação

---

## Rodar em 1 comando

**macOS / Linux**

```bash
bash run.sh
```

**Windows (PowerShell)**

```powershell
.\run.ps1
```

O script:

1. Cria `.venv` se não existir  
2. Instala `requirements.txt`  
3. Copia `.env.example` → `.env` se ainda não houver `.env`  
4. Sobe o Streamlit em `http://localhost:8501`

Depois, se for usar Gemini, edite `.env` e preencha `GEMINI_API_KEY`.

---

## Setup manual

```bash
# 1. Ambiente
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .\.venv\Scripts\Activate.ps1

# 2. Dependências
pip install -r requirements.txt

# 3. Variáveis de ambiente (Gemini opcional)
cp .env.example .env
# edite GEMINI_API_KEY=

# 4. App
streamlit run app/Home.py
```

### Comandos úteis

| Objetivo | Comando |
|----------|---------|
| Demo Streamlit | `streamlit run app/Home.py` |
| Pipeline CLI (recomendação por cluster) | `python -m dataset.models.model_main` |
| Recomendação Gemini | `python -m gemini.recomendacao <athlete_id>` |
| Testes | `pytest tests/` |
| Experimentos | `python -m dataset.experimentos protocolo` (ou `baselines`, `desfecho`, `meta`, `tudo`) |
| Baixar base externa (desfecho 21 km) | `bash scripts/baixar_long_distance.sh` |
| Notebooks | `jupyter notebook notebooks/` |

### Variáveis no `.env`

| Variável | Obrigatória? | Default | Uso |
|----------|--------------|---------|-----|
| `GEMINI_API_KEY` | Só para Gemini | — | API Google AI |
| `GEMINI_MODEL` | Não | `gemini-3.6-flash` | Modelo |
| `GEMINI_TEMPERATURE` | Não | `0.3` | Criatividade (menor = mais estável) |

Nunca commite o `.env` (já está no `.gitignore`).

### Cache do Streamlit

Loaders pesados ficam em `.streamlit/cache/`. Se mudar código de recomendação/metas e a UI parecer “velha”:

```bash
rm -rf .streamlit/cache
```

---

## Estrutura de pastas

```text
tcc-back-2/
├── app/                      # Demo Streamlit
│   ├── Home.py               # Página inicial (entrypoint do app)
│   ├── shared.py             # Loaders cacheados + helpers da UI
│   └── pages/                # Telas do menu lateral (prefixo numérico = ordem)
│
├── dataset/                  # Núcleo de dados e ML
│   ├── raw-data-kaggle.csv   # Base primária (Strava/Kaggle)
│   ├── constants.py          # Colunas + limiares (fonte única)
│   ├── preprocessamento.py   # Limpeza, pace, tipo_treino
│   ├── analise.py            # EDA / gráficos
│   ├── categorizar/          # Perfil por regras + KMeans + nomeação
│   ├── models/               # Modelos e recomendação
│   │   ├── model_main.py     # CLI: python -m dataset.models.model_main
│   │   ├── previsao_pace/    # Modelos de pace
│   │   ├── previsao_tempo/   # Modelos de tempo
│   │   └── previsao_treino/  # Recomendação (cluster = produto)
│   ├── fontes/               # Adaptadores Kaggle / long-distance
│   ├── avaliacao/            # Protocolo por atleta (IC, testes)
│   └── experimentos/         # Baselines, desfecho, eficácia, meta
│
├── gemini/                   # Recomendação via LLM (features + prompt)
├── notebooks/                # Exploração Jupyter (paridade com algumas telas)
├── tests/                    # pytest
├── scripts/                  # Scripts auxiliares (ex.: baixar base externa)
├── output/                   # Artefatos gerados (atletas.csv, recomendacoes.csv)
│
├── run.sh / run.ps1          # Setup + sobe o Streamlit
├── requirements.txt
├── .env.example
├── conftest.py               # Config pytest
│
├── readme.md                 # Este arquivo (como rodar + pastas)
├── FLUXO.md                  # Narrativa ponta a ponta do TCC
├── ADR.md                    # Decisões de arquitetura
├── GUIA-EXPERIMENTOS.md      # Experimentos em português simples
└── SPEC-BACKEND-ORQUESTRADOR.md  # Spec do backend (outro repo)
```

---

## O que cada pasta faz

### `app/` — interface de demonstração

| Arquivo / pasta | Função |
|-----------------|--------|
| `Home.py` | Entrada do Streamlit |
| `shared.py` | Carrega dados/modelos com `@st.cache_data` (persist em disco) |
| `pages/2_…` | Visualização do dataset |
| `pages/3_…` | Comparação de modelos |
| `pages/4_…` | Categorização (KMeans) |
| `pages/5_…` | Tabela de atletas |
| `pages/6_…` | Recomendação (cluster + Gemini) |
| `pages/7_…` | Histórico de recomendações |
| `pages/8_…` | Validação da recomendação |

O menu lateral do Streamlit ordena as páginas pelo **número no nome do arquivo**.

### `dataset/` — ciência de dados

| Pasta / arquivo | Função |
|-----------------|--------|
| `constants.py` | Nomes de coluna (`COL_*`) e thresholds — **não** hardcodar strings no resto do código |
| `preprocessamento.py` | Pipeline: filtrar → imputar FC → pace → classificar tipo |
| `categorizar/` | Agregar atleta, perfil interpretável, KMeans, validação de cluster |
| `models/previsao_treino/` | Regra de recomendação; `recomendacao_cluster.py` é o caminho de produto |
| `fontes/` | Abstração `FonteDados` (primária vs externa) |
| `avaliacao/` | GroupKFold por atleta, bootstrap, Wilcoxon/Holm |
| `experimentos/` | CLI dos experimentos do TCC |

### `gemini/` — LLM

Monta features determinísticas (`features.py`), chama a API (`client.py`) e devolve JSON tipado (`schemas.py`). Histórico em CSV via `historico.py`.

### `notebooks/` — exploração

Espelhados em parte no Streamlit. Úteis para mexer graficamente sem subir o app.

### `tests/` — regressão

Rode `pytest tests/` após mudanças em preprocessamento, recomendação ou telas.

### `output/` — gerado em runtime

Não é código-fonte. Planilhas exportadas pela demo (ex.: recomendações da IA).

### `scripts/` — utilitários

`baixar_long_distance.sh` baixa a base externa usada na validação por desfecho (não versionada; vai para cache local).

---

## Telas do app (o que abrir)

1. **Home** — visão geral  
2. **Visualização do Dataset** — EDA  
3. **Comparação de Modelos** — métricas no protocolo *antigo* (paridade notebooks)  
4. **Categorização** — clusters e fichas  
5. **Atletas** — lista / filtros  
6. **Recomendação de Treino** — informe um `athlete_id`  
7. **Histórico** — recomendações salvas  
8. **Validação da recomendação** — números defensáveis (*a regra funciona?*)

> Para a aba de eficácia por desfecho (tela 8), baixe antes a base externa com `bash scripts/baixar_long_distance.sh`.

---

## Dependências principais

Definidas em `requirements.txt`:

`polars` · `numpy` · `scipy` · `scikit-learn` · `matplotlib` · `xgboost` · `streamlit` · `google-genai` · `jupyter` · `pytest`

---

## Problemas comuns

| Sintoma | O que fazer |
|---------|-------------|
| Gemini falha / 401 | Preencher `GEMINI_API_KEY` no `.env` e reinstalar `google-genai` |
| Tela 8 pede base externa | `bash scripts/baixar_long_distance.sh` |
| UI com números antigos | Apagar `.streamlit/cache/` e recarregar |
| Import lento / sklearn | Normal nas telas de modelo; Atletas deve ser rápido (imports lazy) |
| `ModuleNotFoundError` | Ativar `.venv` e `pip install -r requirements.txt` na raiz do repo |

---

## Onde ler mais

| Documento | Quando abrir |
|-----------|--------------|
| [FLUXO.md](FLUXO.md) | Entender o pipeline completo e o *porquê* de cada etapa |
| [ADR.md](ADR.md) | Decisão de arquitetura específica |
| [GUIA-EXPERIMENTOS.md](GUIA-EXPERIMENTOS.md) | Preparar apresentação / banca |
| [SPEC-BACKEND-ORQUESTRADOR.md](SPEC-BACKEND-ORQUESTRADOR.md) | Integração futura com Strava + frontend |
