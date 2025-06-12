"""
    EELT 7019 - Inteligência Artificial Aplicada
    "Bearing" => pontos de destino dado um azimute e uma distância
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from numba import jit
from GeoLightning.Utils.Utils import coordenadas_em_radianos
from GeoLightning.Utils.Constants import AVG_EARTH_RADIUS


@jit(nopython=True, cache=True, fastmath=True)
def destino_esferico(posicao: np.ndarray,
                     distancia: np.float64,
                     azimute_deg: np.float64,
                     raio: np.float64 = AVG_EARTH_RADIUS) -> np.ndarray:
    """
    Calcula o ponto de destino sobre uma esfera, a partir de uma posição inicial,
    uma distância e um azimute (bearing), considerando altitude constante.

    Args:
        ponto_inicial (np.ndarray): posição com [latitude, longitude, altitude] em graus e metros.
        distancia (np.float64): Distância (m).
        azimute (np.float64): Azimute (graus).
        raio (float): Raio da esfera. Padrão é o raio médio da Terra (6371000 m).

    Returns:
        np.ndarray: Array numpy de shape (N, 3) com [latitude, longitude, altitude] do ponto de destino.
    """

    # Converte para radianos
    r_posicao = coordenadas_em_radianos(posicao)
    lat1 = r_posicao[0]
    long1 = r_posicao[1]
    theta = np.radians(azimute_deg)
    delta = distancia / raio

    # Cálculos
    lat2 = np.arcsin(np.sin(lat1) * np.cos(delta) +
                     np.cos(long1) * np.sin(delta) * np.cos(theta))
    long2 = long1 + np.arctan2(np.sin(theta) * np.sin(delta) * np.cos(lat1),
                               np.cos(delta) - np.sin(lat1) * np.sin(lat2))

    # Converte de volta para graus
    lat2_deg = np.degrees(lat2)
    lon2_deg = np.degrees(long2)
    destino = np.empty(len(posicao))
    destino[0] = lat2_deg
    destino[1] = lon2_deg
    destino[2] = posicao[2]
    return destino


if __name__ == "__main__":

    # Exemplo de teste com vetor de posições
    pontos_iniciais = np.array(
        [[-45.0, 45.0, 100.0], [45.0, -45.0, 100.0]], dtype=np.float64)
    distancias = np.array([200000, 200000], dtype=np.float64)  # em metros
    azimutes = np.array([-45.0, 45.0], dtype=np.float64)        # em graus

    for i in range(len(pontos_iniciais)):
        destino = destino_esferico(pontos_iniciais[i],
                                   distancias[i],
                                   azimutes[i])
        print(destino)
