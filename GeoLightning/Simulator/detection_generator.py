"""
    EELT 7019 - Inteligência Artificial Aplicada
    Gerador de Detecções - Geolocalização de eventos atmosféricos
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
from typing import List, Dict, Tuple, Union
from GeoLightning.Utils.utils import haversine
from GeoLightning.Utils.constants import AVG_LIGHT_SPEED, SIGMA_T
import numpy as np


def generate_detections(events: List[Tuple[float, float, float]],
                        sensors: List[Tuple[float, float]],
                        speed: float = AVG_LIGHT_SPEED,
                        jitter_std: float = SIGMA_T,
                        ) -> List[Dict[str, Union[int, float]]]:
    """
    Gera detecções de eventos por sensores, considerando o tempo de propagação 
    e ruído gaussiano.

    Args:
        events (List[Tuple[float, float, float]]): Lista de eventos [(lat, lon, tempo)]
        sensors (List[Tuple[float, float]]): Lista de sensores [(lat, lon)]
        speed (float): Velocidade de propagação (default = 299792458 m/s)
        jitter_std (float): Desvio padrão do ruído no timestamp (default = 0.0001)

    Returns:
        detections (List[Dict[str, Union[int, float]]]): as detecções dos eventos 
        para cada sensor
    """
    detections = []
    for idx_event, (lat_ev, lon_ev, t_ev) in enumerate(events):
        for idx_sensor, (lat_s, lon_s) in enumerate(sensors):
            distance = haversine(lat_ev, lon_ev, lat_s, lon_s)
            propagation_time = distance / speed
            noise = np.random.normal(0, jitter_std)
            timestamp = t_ev + propagation_time + noise
            detections.append({
                "event_id": idx_event,
                "sensor_id": idx_sensor,
                "timestamp": timestamp
            })
    return detections
