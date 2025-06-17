"""
    EELT 7019 - Inteligência Artificial Aplicada
    Utilitários - Geolocalização de eventos atmosféricos
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

import numpy as np
from numba import jit
from .Constants import AVG_EARTH_RADIUS, AVG_LIGHT_SPEED

#################################################################
# Utilitários comuns
#################################################################


@jit(nopython=True, cache=True, fastmath=True)
def coordenadas_em_radianos(coordenadas: np.ndarray) -> np.ndarray:
    """
        Converte um vetor de coordenadas esféricas com ângulos em graus
        para o mesmo vetor com angulos em radianos
        Args:
            coordenadas (np.ndarray): vetor de coordenadas em graus
        Returns:
            novas_coordenadas (np.ndarray): vetor de coordenadas em radianos
    """
    novas_coordenadas = np.array([np.deg2rad(coordenadas[0]),
                                  np.deg2rad(coordenadas[1]),
                                  coordenadas[2]])
    return novas_coordenadas


@jit(nopython=True, cache=True, fastmath=True)
def coordenadas_em_radianos_batelada(s_coordinates: np.ndarray) -> np.ndarray:
    """
        Converte um vetor de coordenadas esféricas com ângulos em graus
        para o mesmo vetor com angulos em radianos - batch_processing
        Args:
            coordenadas (np.ndarray): vetor de coordenadas em graus
        Returns:
            novas_coordenadas (np.ndarray): vetor de coordenadas em radianos
    """
    novas_coordenadas = np.copy(s_coordinates)
    for i in range(0, len(s_coordinates)):
        novas_coordenadas[i] = coordenadas_em_radianos(s_coordinates[i])
    return novas_coordenadas


@jit(nopython=True, cache=True, fastmath=True)
def determinacao_angular_minima(angulo: np.float64) -> np.float64:
    """
        Redução à determinação mínima
        Args:
            angulo (np.float64): ângulo
        Returns:
            angulo (np.float64): ângulo reduzido à sua minima determinação
    """
    angulo = angulo % (2 * np.pi)
    if angulo > np.pi:
        angulo -= 2 * np.pi
    elif angulo < -np.pi:
        angulo += 2 * np.pi
    return angulo

#################################################################
# Distâncias 
#################################################################


@jit(nopython=True, cache=True, fastmath=True)
def distancia_esferica_entre_pontos(a: np.ndarray,
                                    s: np.ndarray) -> np.float64:
    """
        Computa distâncias no sistema de coordenadas esféricas
        ([lat, long, alt])
        Args:
            a (np.ndarray): vetor de coordenadas do evento
            s (np.ndarray): vetor de coordenadas da estação ou detector
        Returns:
            distancia (np.float64): a distância estimada entre os 
            dois pontos a e s
    """
    r_a = coordenadas_em_radianos(a)
    r_s = coordenadas_em_radianos(s)
    alt_a = AVG_EARTH_RADIUS + a[2]
    alt_s = AVG_EARTH_RADIUS + s[2]
    delta_alt = alt_a - alt_s
    half_delta_lat = (r_a[0] - r_s[0]) / 2.0
    half_delta_long = (r_a[1] - r_s[1]) / 2.0
    lat_a = r_a[0]
    lat_s = r_s[0]
    arg = 4 * alt_a * alt_s * (np.power(np.sin(half_delta_lat), 2.0)
                               + np.cos(lat_a) * np.cos(lat_s) *
                               np.power(np.sin(half_delta_long), 2.0))
    + np.power(delta_alt, 2.0)
    return np.sqrt(arg)


@jit(nopython=True, cache=True, fastmath=True)
def distancia_cartesiana_entre_pontos(a: np.ndarray,
                                      s: np.ndarray) -> np.float64:
    """
        Computa distâncias no sistema de coordenadas cartesianas
        ([x, y, z])
        Args:
            a (np.ndarray): vetor de coordenadas do evento
            s (np.ndarray): vetor de coordenadas da estação ou detector
        Returns:
            distancia (np.float64): a distância estimada entre os 
            dois pontos a e s
    """
    temp = np.subtract(a, s)
    return np.sqrt(temp.dot(temp.T))

@jit(nopython=True, cache=True, fastmath=True)
def computa_distancia(a: np.ndarray,
                      s: np.ndarray,
                      sistema_cartesiano: bool = False) -> np.float64:
    """
        Computa distâncias dependendo do sistema de coordenadas
        Args:
            a (np.ndarray): vetor de coordenadas do evento
            s (np.ndarray): vetor de coordenadas da estação ou detector
            sistema_cartesiano (bool): usa distância euclidiana (True) 
                ou esférica (False) 
        Returns:
            distancia (np.float64): a distância estimada entre os 
            dois pontos a e s
    """
    if sistema_cartesiano:
        return distancia_cartesiana_entre_pontos(
                a, s)
    return distancia_esferica_entre_pontos(a, s)


@jit(nopython=True, cache=True, fastmath=True)
def computa_distancias(origem: np.ndarray,
                       destinos: np.ndarray,
                       sistema_cartesiano: bool = False) -> np.ndarray:
    """
        Computa as distâncias entre um ponto de origem e os pontos de destino
        Uso: no cálculo da verissimilhança espaço-temporal
        Args:
            origem (np.ndarray): o ponto de origem
            destinos (np.ndarray): os pontos de destino
            sistema_cartesiano (bool): usa distância euclidiana (True) 
                ou esférica (False)
        Returns:
            distancias (np.ndarray): as distâncias entre origem e destinos
    """
    distancias = np.zeros(len(destinos), dtype=np.float64)

    for i in range(len(destinos)):
        if sistema_cartesiano:
            distancias[i] = distancia_cartesiana_entre_pontos(
                origem, destinos[i])
        else:
            distancias[i] = distancia_esferica_entre_pontos(
                origem, destinos[i])

    return distancias

@jit(nopython=True, cache=True, fastmath=True)
def computa_tempos_de_origem(solucoes: np.ndarray,
                             clusters_espaciais: np.ndarray,
                             tempos_de_chegada: np.ndarray,
                             pontos_de_deteccao: np.ndarray,
                             sistema_cartesiano: bool = False) -> np.ndarray:
    """
        Computa os tempos de origem para cada par (solução, sensor, tempo_de_chegada).

        Args:
            solucoes (np.ndarray): array (N, 3) de posições candidatas
            clusters_espaciais (np.ndarray): array (M,) de mapeamento das soluções para os tempos de chegada
                e pontos de detecção (M <= N)
            tempos_de_chegada (np.ndarray): array (N,) de tempos de chegada
            sensores (np.ndarray): array (N, 3) de coordenadas dos sensores (pode haver repetição)
            sistema_cartesiano (bool): usa distância euclidiana (True) ou esférica (False)

        Returns:
            np.ndarray: vetor (N,) com os tempos de origem estimados
    """
    N = len(clusters_espaciais)
    distancias = np.zeros(N, dtype=np.float64)
    clusters_espaciais = clusters_espaciais.astype(np.int64)
    
    for i in range(N):
        if sistema_cartesiano:
            distancias[i] = distancia_cartesiana_entre_pontos(
                solucoes[clusters_espaciais[i]], pontos_de_deteccao[i])
        else:
            distancias[i] = distancia_esferica_entre_pontos(
                solucoes[clusters_espaciais[i]], pontos_de_deteccao[i])

    tempos_de_origem = tempos_de_chegada - distancias / AVG_LIGHT_SPEED

    return tempos_de_origem
