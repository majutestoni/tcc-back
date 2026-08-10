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

# Limiares para classificação de tipo de treino (normalizacao.py)
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
