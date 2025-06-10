import numpy as np
from numba import jit


@jit(nopython=True, cache=True, fastmath=True)
def destino_esferico(posicao: np.ndarray, 
                     distancia: np.float64, 
                     azimute_deg: np.float64, 
                     raio: np.float64=6371000) -> np.ndarray:
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
    lat1 = np.radians(posicao[0])
    long1 = np.radians(posicao[1])
    theta = np.radians(azimute_deg)
    delta = distancia / raio

    # Cálculos
    lat2 = np.arcsin(np.sin(lat1) * np.cos(delta) + np.cos(long1) * np.sin(delta) * np.cos(theta))
    long2 = long1 + np.arctan2(np.sin(theta) * np.sin(delta) * np.cos(lat1),
                       np.cos(delta) - np.sin(lat1) * np.sin(lat2))

    # Converte de volta para graus
    lat2_deg = np.degrees(lat2)
    lon2_deg = np.degrees(long2)
    destino = np.empty(len(posicao))
    destino[0] = lat2_deg
    destino[1] = lon2_deg
    destino[2] = posicao[2]
    return lat2_deg, lon2_deg


@jit(nopython=True, cache=True, fastmath=True)
def destino_esferico_vetorizado(ponto_inicial: np.ndarray,
                                distancia: np.ndarray,
                                azimute: np.ndarray,
                                raio: np.float64 = 6371000.0) -> np.ndarray:
    """
    Calcula o ponto de destino sobre uma esfera, a partir de uma posição inicial,
    uma distância e um azimute (bearing), considerando altitude constante.

    Args:
        ponto_inicial (np.ndarray): Array numpy de shape (N, 3) com [latitude, longitude, altitude] em graus e metros.
        distancia (np.ndarray): Distância (m) escalar ou array com N elementos.
        azimute (np.ndarray): Azimute (graus) escalar ou array com N elementos.
        raio (float): Raio da esfera. Padrão é o raio médio da Terra (6371000 m).

    Returns:
        np.ndarray: Array numpy de shape (N, 3) com [latitude, longitude, altitude] do ponto de destino.
    """
    phi = np.radians(ponto_inicial[:, 0])
    lmbda = np.radians(ponto_inicial[:, 1])
    h = ponto_inicial[:, 2]
    theta = np.radians(azimute)
    delta = distancia / raio

    # Cálculo do novo ponto
    phi2 = np.arcsin(np.sin(phi) * np.cos(delta) + np.cos(phi)
                     * np.sin(delta) * np.cos(theta))
    lmbda2 = lmbda + np.arctan2(
        np.sin(theta) * np.sin(delta) * np.cos(phi),
        np.cos(delta) - np.sin(phi) * np.sin(phi2)
    )

    # Conversão de volta para graus
    lat2 = np.degrees(phi2)
    lon2 = np.degrees(lmbda2)
    lon2 = (lon2 + 180.0) % 360.0 - 180.0

    # Alocação e preenchimento do resultado
    n = ponto_inicial.shape[0]
    resultado = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        resultado[i, 0] = lat2[i]
        resultado[i, 1] = lon2[i]
        resultado[i, 2] = h[i]

    return resultado


if __name__ == "__main__":

    # Exemplo de teste com vetor de posições
    pontos_iniciais = np.array(
        [[-45.0, 45.0, 100.0], [45.0, -45.0, 100.0]], dtype=np.float64)
    distancias = np.array([200000, 200000], dtype=np.float64)  # em metros
    azimutes = np.array([-45.0, 45.0], dtype=np.float64)        # em graus

    destinos = destino_esferico_vetorizado(
        pontos_iniciais, distancias, azimutes)
    print(destinos)

    print(destino_esferico(np.array([-45.0, 45.0, 100.0]),
                           200000,
                           -45.0))
