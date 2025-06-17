"""
    EELT 7019 - Inteligência Artificial Aplicada
    Gerador de Detecções - Geolocalização de eventos atmosféricos
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from numba import jit


@jit(nopython=True, cache=True, fastmath=True)
def get_detection_probability(distance: np.float64) -> np.float64:
    """
    Determina a probabilidade de detecção em função da
        Distância
        Args: 
            distance (np.float64): a distância estimada
        Returns:
            p (np.float64): a probabilidade de detecção
    """
    if distance < 120_000.0:
        return 0.9
    else:
        p = (0.9 - 0.005 *
                   (distance - 120_000.0)
                   / 1_000.0)
        if p > 0.0:
            return p
        return 0.0
    
@jit(nopython=True, cache=True, fastmath=True)
def sensor_detection(distance: np.float64) -> bool:
    """
        Simula uma detecção bem sucedida (ou não) do evento
        Args:
            distance (np.float64): a distância estimada
        Returns:
            detected (bool): indicativo de detecção
    """
    probability = get_detection_probability(distance)

    rand_int = np.random.random_sample()
    if rand_int <= probability:
        return True
    return False

if __name__ == "__main__":
    for i in range(100):
        distance = np.random.uniform(20_000.0, 200_000.0)
        print(distance, sensor_detection(distance))