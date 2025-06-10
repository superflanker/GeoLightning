"""
    EELT 7019 - Inteligência Artificial Aplicada
    Constantes - Geolocalização de eventos atmosféricos
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
# raio médio da Terra (em metros)
AVG_EARTH_RADIUS = 1000 * 6371.0088

# velocidade mpedia da luz no vácuo (em metros por segundo)
AVG_LIGHT_SPEED = 299_792_458.0

# desvio padrão temporal - caracteristica do sensor (em segundos)
SIGMA_T = 1.26e-6

# desvio padrão espacial - relacionada ao std temporal
SIGMA_D = AVG_LIGHT_SPEED * SIGMA_T

# limites máximos de erro 
# temporal
EPSILON_T = 3 * SIGMA_T
# espacial
EPSILON_D = 3 * SIGMA_D
# numero mínimo de elementos por cluster
CLUSTER_MIN_PTS = 3
