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
