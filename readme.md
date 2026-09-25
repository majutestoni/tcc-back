# TCC - Ánalise de modelos

### bases de dados usadas
strava1 = https://www.kaggle.com/datasets/olegoaer/running-races-strava

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
