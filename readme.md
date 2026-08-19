# TCC - Ánalise de modelos

### bases de dados usadas
strava1 = https://www.kaggle.com/datasets/olegoaer/running-races-strava


###  criar um novo venv caso preciese 
```bash
python -m venv .venv
```
### ative o ambiente:
```bash
    source .venv/bin/activate
```

```bash
    .\.venv\Scripts\Activate.ps1
```

###  Instale as dependências
```bash
   pip install polars numpy scikit-learn matplotlib xgboost jupyter
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
