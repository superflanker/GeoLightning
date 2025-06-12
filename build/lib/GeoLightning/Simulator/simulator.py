"""
    EELT 7019 - Inteligência Artificial Aplicada
    Simulador de sensores e Eventos - Geolocalização de eventos atmosféricos
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

import numpy as np
from typing import List, Tuple


def generate_sensors(num_sensors: int = 10,
                     lat_range: Tuple[float, float] = (-10, -5),
                     lon_range: Tuple[float, float] = (-50, -45)
                     ) -> List[Tuple[float, float]]:
    """
    Gera a localização geográfica dos sensores.

    Args:
        num_sensors (int): Número de sensores a serem gerados.
        lat_range (Tuple[float, float]): Intervalo de latitude.
        lon_range (Tuple[float, float]): Intervalo de longitude.

    Returns:
        List[Tuple[float, float]]: Lista de sensores [(lat, lon)].
    """
    lats = np.random.uniform(lat_range[0], lat_range[1], num_sensors)
    lons = np.random.uniform(lon_range[0], lon_range[1], num_sensors)
    return list(zip(lats, lons))


def generate_events(num_events: int = 5,
                    lat_range: Tuple[float, float] = (-10, -5),
                    lon_range: Tuple[float, float] = (-50, -45),
                    time_range: Tuple[float, float] = (0, 10)
                    ) -> List[Tuple[float, float, float]]:
    """
    Gera eventos com localização geográfica e tempo.

    Args:
        num_events (int): Número de eventos a serem gerados.
        lat_range (Tuple[float, float]): Intervalo de latitude.
        lon_range (Tuple[float, float]): Intervalo de longitude.
        time_range (Tuple[float, float]): Intervalo de tempo (em segundos ou unidade arbitrária).

    Returns:
        List[Tuple[float, float, float]]: Lista de eventos [(lat, lon, tempo)].
    """
    lats = np.random.uniform(lat_range[0], lat_range[1], num_events)
    lons = np.random.uniform(lon_range[0], lon_range[1], num_events)
    times = np.random.uniform(time_range[0], time_range[1], num_events)
    return list(zip(lats, lons, times))
