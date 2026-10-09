# TCC - Ánalise de modelos

### bases de dados usadas
- **primária** — https://www.kaggle.com/datasets/olegoaer/running-races-strava
  (em `dataset/raw-data-kaggle.csv`, separador `;`)
- **validação externa** (ADR-018) — https://www.kaggle.com/datasets/mexwell/long-distance-running-dataset
  (CC BY 4.0; 2,3 GB, **não versionada**, baixe com `bash scripts/baixar_long_distance.sh`)

A base externa é o que permite avaliar a recomendação contra um **desfecho de
prova** (melhor tempo de meia maratona) em vez de contra a própria premissa.

---

## 📖 Comece por aqui

**[GUIA-EXPERIMENTOS.md](GUIA-EXPERIMENTOS.md)** — os 4 experimentos e suas
conclusões em português simples, com glossário dos termos estatísticos, o que
cada tela do app responde, e perguntas de banca com resposta pronta. Leia antes
de mexer em `dataset/avaliacao/` ou `dataset/experimentos/`.

O [ADR.md](ADR.md) tem o registro completo das decisões e o *porquê* de cada uma.

---

## 🚀 Setup rápido (clonar e rodar)

Depois de clonar o repositório, um único comando cria o ambiente, instala tudo e sobe o app:

**macOS / Linux**
```bash
bash run.sh
```

**Windows (PowerShell)**
```powershell
.\run.ps1
```

O script cria o `.venv`, instala as dependências do `requirements.txt`, gera o `.env`
(se não existir) e abre o Streamlit. Depois é só preencher a `GEMINI_API_KEY` no `.env`
para usar a recomendação via Gemini.

---

## Setup manual (passo a passo)

###  criar um novo venv caso preciese 
```bash
python -m venv .venv
```
### ative o ambiente:
```bash
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

###  Instale as dependências
```bash
   pip install -r requirements.txt
```

### recomendação de treino via Gemini
Configure a chave da API (copie `.env.example` para `.env` e preencha `GEMINI_API_KEY`):
```bash
cp .env.example .env
```
Rode a recomendação para um atleta específico:
```bash
python -m gemini.recomendacao <athlete_id>
```

### app Streamlit (demo)
Interface interativa (mantém os notebooks para exploração):
```bash
streamlit run app/Home.py
```

### rodar (sem jupyter):
```bash
    python -m dataset.models.model_main 
```

### experimentos (ADR-017, ADR-018, ADR-019)
```bash
# item 1 — os 8 modelos sob particionamento por atleta, com IC e testes pareados
python -m dataset.experimentos protocolo

# item 2 — baselines vs a regra de ADR-007 + ablação dos seus componentes
python -m dataset.experimentos baselines

# item 3 — validação externa com desfecho de meia maratona (precisa da base externa)
bash scripts/baixar_long_distance.sh
python -m dataset.experimentos desfecho --max-atletas 12000

# item 4 — meta de distribuição aprendida do desfecho de 21 km
python -m dataset.experimentos meta --max-atletas 12000

# tudo de uma vez
python -m dataset.experimentos tudo
```

### testes
```bash
pytest tests/
```

### notebooks
```bash
pip install jupyter
jupyter notebook notebooks/visualizacao_dataset.ipynb
jupyter notebook notebooks/comparacao_modelos.ipynb
jupyter notebook notebooks/categorizacao_corredores.ipynb
jupyter notebook notebooks/recomendacao_treino.ipynb
```


## Comunicao
flowchart LR
    U[Usuário] --> FE[Frontend\nrun-metrics-mate]
    FE <-->|REST/JSON| BE[Backend\norquestrador]
    BE <-->|OAuth + sync| ST[Strava API]
    BE <-->|persistência| DB[(Banco de dados)]
    BE <-->|análise/predição| ML[Serviço ML\ntcc-back-2]


flowchart TD
    A[Usuário desconectado] --> B{Escolhe provider}
    B -->|Strava| C[OAuth Strava]
    B -->|Intervals| D[OAuth Intervals]
    C --> E[Sessão provider=strava]
    D --> F[Sessão provider=intervals]
    E --> G[Sync usa StravaClient]
    F --> H[Sync usa IntervalsClient]
    E -->|Conectar Intervals| I[Disconnect Strava + limpa dados]
    I --> F
    F -->|Conectar Strava| J[Disconnect Intervals + limpa dados]
    J --> E
