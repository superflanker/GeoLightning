"""
    EELT 7019 - Inteligência Artificial Aplicada
    Função Fitness - Geolocalização de eventos atmosféricos
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
from typing import List, Dict, Tuple
import numpy as np
from GeoLightning.Utils.utils import haversine
from GeoLightning.Utils.constants import AVG_LIGHT_SPEED


def fitness(solution: np.ndarray,
            detections: List[Dict[str, float]],
            sensors: List[Tuple[float, float]],
            max_events: int,
            speed: float = AVG_LIGHT_SPEED) -> float:
    """
        Calcula o erro total da solução proposta, considerando:
        - Atribuição de detecções a eventos.
        - Localização e tempo dos eventos.
        - Penalização por número de eventos ativos.

        Args:
            solution (ndarray): Vetor com parâmetros dos eventos e agrupamento das detecções.
            detections (List[Dict[str, float]]): Lista de dicionários com timestamps e ids dos sensores.
            sensors (List[Tuple[float, float]]): Lista de tuplas com latitudes e longitudes dos sensores.
            max_events (int): Número máximo possível de eventos (superestimado).
            speed (float): Velocidade de propagação do sinal (em km/s).

        Returns:
            total_error (float): Erro total da solução proposta.
    """

    event_params = solution[:max_events*4]
    assignments = solution[max_events*4:].astype(int)
    total_error = 0.0
    active_events = 0

    for idx in range(max_events):
        flag = event_params[idx*4 + 3]
        if flag >= 0.5:
            active_events += 1

    for i, detection in enumerate(detections):
        event_idx = assignments[i]
        flag = event_params[event_idx*4 + 3]
        if flag < 0.5:
            total_error += 10.0  # Penalização por atribuição inválida
            continue

        lat_ev = event_params[event_idx*4]
        lon_ev = event_params[event_idx*4 + 1]
        time_ev = event_params[event_idx*4 + 2]

        lat_s, lon_s = sensors[detection["sensor_id"]]

        distance = haversine(lat_ev, lon_ev, lat_s, lon_s) * 1000  # em metros
        expected_time = time_ev + (distance / speed)

        error = abs(detection["timestamp"] - expected_time)
        total_error += error

    # Penalização leve por quantidade de eventos ativos
    total_error += active_events * 0.5
    return total_error
