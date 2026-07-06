COL_GENDER = "gender"
COL_ATHLETE = "athlete"
COL_HR = "average heart rate (bpm)"
COL_DISTANCE = "distance (m)"
COL_ELAPSED = "elapsed time (s)"
COL_PACE = "pace_min_km"
COL_TIMESTAMP = "timestamp"
COL_ELEVATION = "elevation gain (m)"
COL_TIPO_TREINO = "tipo_treino"

# Limiares para classificação de tipo de treino (normalizacao.py)
LIMIAR_DIST_LONGAO = 0.75
LIMIAR_PACE_INTERVALADO = 0.5
LIMIAR_PACE_LEVE = 0.4
LIMIAR_PACE_RITMO = 0.3

# Janela temporal para cálculo da distribuição de treinos
SEMANAS_JANELA = 6