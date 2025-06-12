"""
    EELT 7019 - Inteligência Artificial Aplicada
    Constantes - Geolocalização de eventos atmosféricos
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np

# raio médio da Terra (em metros)
AVG_EARTH_RADIUS: np.float64 = 1000 * 6371.0088

# velocidade mpedia da luz no vácuo (em metros por segundo)
AVG_LIGHT_SPEED: np.float64 = 299_792_458.0

# desvio padrão temporal - caracteristica do sensor (em segundos)
SIGMA_T: np.float64 = 1.26e-6

# desvio padrão espacial - relacionada ao std temporal
SIGMA_D: np.float64 = AVG_LIGHT_SPEED * SIGMA_T

# limites máximos de erro
# temporal
EPSILON_T: np.float64 = 800 * SIGMA_T
# espacial
EPSILON_D: np.float64 = 800 * SIGMA_D

# numero mínimo de elementos por cluster
CLUSTER_MIN_PTS: np.int32 = 3

# metros por grau de latitude
R_LAT: np.float64 = 111320.0

# Distância de alcance de cada sensor

MAX_DISTANCE: np.float64 = 160 * 1000.0
