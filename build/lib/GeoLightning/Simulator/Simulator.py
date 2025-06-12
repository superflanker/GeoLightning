"""
    EELT 7019 - Inteligência Artificial Aplicada
    Simulador de sensores e Eventos - Geolocalização de eventos atmosféricos
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

import numpy as np
from numba import jit
from numba.typed import List
from GeoLightning.Utils.Constants import R_LAT, AVG_LIGHT_SPEED, SIGMA_T
from GeoLightning.Utils.Utils import computa_distancias, computa_tempos_de_origem
from GeoLightning.Simulator.SensorModel import sensor_detection


@jit(nopython=True, cache=True, fastmath=True)
def get_random_sensors(num_sensors: np.int32,
                       min_lat: np.float64,
                       max_lat: np.float64,
                       min_lon: np.float64,
                       max_lon: np.float64,
                       min_alt: np.float64,
                       max_alt: np.float64) -> np.ndarray:
    """
    Gera a localização geográfica dos sensores.

    Args:
        num_sensors (np.int32): número de sensores
        min_lat (np.float64): latitude mínima
        max_lat (np.float64): latitude máxima
        min_lon (np.float64): longitude mínima
        max_lon (np.float64): longitude máxima
        min_alt (np.float64): altitude mínima
        max_alt (np.float64): altitude máxima
    Returns:
        np.ndarray: Lista de sensores [(lat, lon, alt)].
    """
    lats = np.random.uniform(min_lat, max_lat, num_sensors)
    lons = np.random.uniform(min_lon, max_lon, num_sensors)
    alts = np.random.uniform(min_alt, max_alt, num_sensors)
    sensors = np.empty((num_sensors, 3))
    for i in range(num_sensors):
        sensors[i, 0] = lats[i]
        sensors[i, 1] = lons[i]
        sensors[i, 2] = alts[i]

    return sensors


@jit(nopython=True, cache=True, fastmath=True)
def get_sensors() -> np.ndarray:
    """
    Gera a localização geográfica dos sensores - disposição estática.

    Args:
        None
    Returns:
        np.ndarray: Lista de sensores [(lat, lon, alt)].
    """
    sensors = np.array([[-24.5, -51.0,  935.0],
                        [-23.77771914, -51.0,  935.0],
                        [-24.13733229, -50.3183839, 935.0],
                        [-24.85955584, -50.31447896, 935.0],
                        [-25.22221178, -51.0, 935.0],
                        [-24.85955584, -51.68552104, 935.0],
                        [-24.13733229, -51.6816161, 935.0]])
    return sensors


@jit(nopython=True, cache=True, fastmath=True)
def get_lightning_limits(sensores_latlon: np.ndarray,
                         margem_metros: float = 5000.0) -> tuple:
    """
    Calcula os limites geográficos (min_lat, min_lon, max_lat, max_lon) que cercam a constelação de sensores,
    com uma margem adicional especificada em metros.

    Args:
        sensores_latlon (np.ndarray): matriz (N,2) com colunas [latitude, longitude] em graus
        margem_metros (float): margem adicional em metros ao redor do retângulo mínimo

    Returns:
        tuple: (min_lat, min_lon, max_lat, max_lon) com a margem aplicada
    """
    latitudes = sensores_latlon[:, 0]
    longitudes = sensores_latlon[:, 1]

    min_lat = np.min(latitudes)
    max_lat = np.max(latitudes)
    min_lon = np.min(longitudes)
    max_lon = np.max(longitudes)

    lat_media = np.mean(latitudes)

    delta_lat = margem_metros / R_LAT
    delta_lon = margem_metros / (R_LAT * np.cos(np.radians(lat_media)))

    min_lat -= delta_lat
    max_lat += delta_lat
    min_lon -= delta_lon
    max_lon += delta_lon

    return min_lat, max_lat, min_lon, max_lon


@jit(nopython=True, cache=True, fastmath=True)
def generate_events(num_events:  np.int32,
                    min_lat: np.float64,
                    max_lat: np.float64,
                    min_lon: np.float64,
                    max_lon: np.float64,
                    min_alt: np.float64,
                    max_alt: np.float64,
                    min_time: np.float64,
                    max_time: np.float64,
                    ) -> tuple:
    """
    Gera eventos com localização geográfica e tempo.

    Args:
        num_events (np.int32): Número de eventos a serem gerados.
        min_lat (np.float64): latitude mínima
        max_lat (np.float64): latitude máxima
        min_lon (np.float64): longitude mínima
        max_lon (np.float64): longitude máxima
        min_alt (np.float64): altitude mínima
        max_alt (np.float64): altitude máxima
        min_time (np.float64): tempo mínimo
        max_time (np.float64): tempo_máximo

    Returns:
        tuple =>
            event_points (np.ndarray): os pontos de descarga
            event_times (np.ndarray): os tempos de descarga
    """

    lats = np.random.uniform(min_lat, max_lat, num_events)
    lons = np.random.uniform(min_lon, max_lon, num_events)
    alts = np.random.uniform(min_alt, max_alt, num_events)
    times = np.linspace(min_time, max_time, num_events)
    event_positions = np.empty((num_events, 3))
    for i in range(num_events):
        event_positions[i, 0] = lats[i]
        event_positions[i, 1] = lons[i]
        event_positions[i, 2] = alts[i]

    return event_positions, times


def generate_detections(event_positions: np.ndarray,
                        event_times: np.ndarray,
                        sensor_positions: np.ndarray,
                        jitter_std: np.float64 = SIGMA_T) -> tuple:
    """
    Gera as detecções dado um grupo de sensores
    Args:
        event_positions (np.ndarray): as posições dos eventos
        event_times (np.ndarray): os tempos dos eventos
        sensor_positions (np.ndarray): as posições dos sensores
    Returns: 
        tuple:
            detections (np.ndarray): o sensor associado à 
                detecção do evento
            detection_times (np.ndarray): os tempos associados 
                à detecção do evento
            n_event_positions (np.ndarray): versão expandida 
                de event_positions (verificação  e teste)
            n_event_times (np.ndarray): versão expandida 
                de event_times (verificação e teste)
            distances (np.ndarray): as distâncias calculadas
                (verificação e testes)
    """
    detections = []
    detection_times = []
    n_event_positions = []
    n_event_times = []
    distances = []
    spatial_clusters = []

    cluster_id = 0

    for i in range(event_positions.shape[0]):
        event_position = event_positions[i]
        event_time = event_times[i]
        event_distances = computa_distancias(event_position, sensor_positions)
        t_detections = []
        t_detection_times = []
        t_n_event_positions = []
        t_n_event_times = []
        t_distances = []
        t_spatial_clusters = []
        for j in range(sensor_positions.shape[0]):
            noise = np.clip(np.random.normal(0.0, jitter_std),
                            min=-6 * jitter_std,
                            max=6 * jitter_std)
            t_detect = event_time + \
                event_distances[j] / AVG_LIGHT_SPEED + noise
            if sensor_detection(event_distances[j]):
                t_detections.append(sensor_positions[j])
                t_detection_times.append(t_detect)
                t_n_event_positions.append(event_position)
                t_n_event_times.append(event_time)
                t_distances.append(event_distances[j])
                t_spatial_clusters.append(cluster_id)
        if len(t_detections) >= 3:
            cluster_id += 1
            detections += t_detections
            detection_times += t_detection_times
            n_event_positions += t_n_event_positions
            n_event_times += t_n_event_times
            distances += t_distances
            spatial_clusters += t_spatial_clusters

    return (np.array(detections),
            np.array(detection_times),
            np.array(n_event_positions),
            np.array(n_event_times),
            np.array(distances),
            np.array(spatial_clusters))


if __name__ == "__main__":

    # recuperando o grupo de sensores
    sensors = get_sensors()
    min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors)

    # gerando os eventos
    min_alt = 0
    max_alt = 10000
    min_time = 10000
    max_time = 10100
    num_events = 500

    print(min_lat, max_lat, min_lon, max_lon, min_alt, max_alt)

    event_positions, event_times = generate_events(num_events,
                                                   min_lat,
                                                   max_lat,
                                                   min_lon,
                                                   max_lon,
                                                   min_alt,
                                                   max_alt,
                                                   min_time,
                                                   max_time)

    # gerando as detecções
    (detections,
     detection_times,
     n_event_positions,
     n_event_times,
     distances,
     spatial_clusters) = generate_detections(event_positions,
                                             event_times,
                                             sensors)

    print(detections)
    print(n_event_positions)
    print(detection_times)
    print(distances)
