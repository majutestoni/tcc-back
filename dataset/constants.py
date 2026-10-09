COL_GENDER = "gender"
COL_ATHLETE = "athlete"
COL_HR = "average heart rate (bpm)"
COL_DISTANCE = "distance (m)"
COL_ELAPSED = "elapsed time (s)"
COL_PACE = "pace_min_km"
COL_TIMESTAMP = "timestamp"
COL_ELEVATION = "elevation gain (m)"
COL_TIPO_TREINO = "tipo_treino"
COL_CLUSTER_CORREDOR = "cluster_corredor"
COL_FREQUENCIA = "frequencia_semana"
COL_N_CORRIDAS = "n_corridas"
COL_NIVEL_PACE = "nivel_pace"
COL_NIVEL_VOLUME = "nivel_volume"
COL_NIVEL_DISTANCIA = "nivel_distancia"
COL_PERFIL_ATLETA = "perfil_atleta"

# Limiares para classificação de tipo de treino (preprocessamento.py)
LIMIAR_DIST_LONGAO = 0.75
LIMIAR_PACE_INTERVALADO = 0.5
LIMIAR_PACE_LEVE = 0.4
LIMIAR_PACE_RITMO = 0.3

# Limiares de volume (corridas/semana) e distância (km) para perfil do atleta
LIMIAR_VOLUME_BAIXO = 1.5
LIMIAR_VOLUME_ALTO = 3.0
LIMIAR_DIST_CURTA_KM = 8.0
LIMIAR_DIST_LONGA_KM = 12.0
MIN_CORRIDAS_PERFIL = 20

# Janela temporal para cálculo da distribuição de treinos
SEMANAS_JANELA = 6

# --- Avaliação experimental (ADR-017) ---
SEED = 42
N_FOLDS = 5
N_BOOTSTRAP = 2000
ALFA = 0.05

# --- Desfecho de prova (ADR-018) ---
# Meia maratona é o foco: público amador. Maratona fica como checagem de robustez.
DISTANCIA_MEIA_KM = 21.0975
DISTANCIA_MARATONA_KM = 42.195
TOLERANCIA_PROVA_KM = 0.6
SEMANAS_PREPARO = 12
MIN_TREINOS_PREPARO = 20

# Colunas extra da fonte long-distance (ADR-018); ausentes na fonte primária
COL_FAIXA_IDADE = "age_group"
COL_PAIS = "country"
COL_MAJORS = "major"

COL_TEMPO_PROVA = "tempo_prova_min"
COL_PACE_PROVA = "pace_prova_min_km"
COL_DATA_PROVA = "data_prova"
COL_DISTANCIA_PROVA = "distancia_prova_km"
