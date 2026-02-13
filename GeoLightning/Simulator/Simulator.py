"""
Network and Event Simulator
===========================

Sensor and Event Simulator – Atmospheric Event Geolocation

Summary
-------
This module implements the simulation of sensor detections and event generation for atmospheric discharge localization experiments.
It is used to evaluate the performance of geolocation algorithms under varying spatial and temporal configurations.

The simulator includes:
- Emission of impulsive events at known positions and times;
- Probabilistic detection by sensors based on propagation distance;
- Addition of Gaussian noise in detection times;
- Derivation of Time-of-Arrival (TOA) information for multilateration.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- Sensor-event interaction model
- Distance and TOA calculations
- Time jitter simulation (σₜ)
- TOA-based emission time estimation

Notes
-----
This module is part of the activities of the discipline
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numba
- GeoLightning.Utils.Constants
- GeoLightning.Utils.Utils
- GeoLightning.Simulator.SensorModel
"""


import numpy as np
from numba import jit
from numba.typed import List
from GeoLightning.Utils.Constants import R_LAT, AVG_LIGHT_SPEED, SIGMA_T
from GeoLightning.Utils.Utils import computa_distancias, computa_distancia
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
    Generates random geographic coordinates for a given number of sensors.

    Parameters
    ----------
    num_sensors : np.int32
        Number of sensors.
    min_lat : np.float64
        Minimum latitude.
    max_lat : np.float64
        Maximum latitude.
    min_lon : np.float64
        Minimum longitude.
    max_lon : np.float64
        Maximum longitude.
    min_alt : np.float64
        Minimum altitude.
    max_alt : np.float64
        Maximum altitude.

    Returns
    -------
    np.ndarray
        Array of shape (num_sensors, 3) with sensor positions as (lat, lon, alt).
    """
    np.random.seed(42)
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
    Returns a static array of sensor positions (lat, lon, alt).

    Returns
    -------
    np.ndarray
        Array of shape (7, 3) with fixed sensor coordinates.
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
def get_parana_state_sensors() -> np.ndarray:
    """
    Returns a network over Paraná's state - StormEye Projections

    Returns
    -------
    np.ndarray
        Array of shape (30, 3) with fixed sensor coordinates.
    """
    sensors = np.array([[-24.84724728, -50.0468059, 935.0],
                        [-25.13080449, -53.95538321, 935.0],
                        [-25.6502094, -50.54685779, 935.0],
                        [-23.08138184, -51.86892399, 935.0],
                        [-25.67956232, -49.23511376, 935.0],
                        [-25.21635234, -52.03071597, 935.0],
                        [-24.00697991, -53.56584922, 935.0],
                        [-25.05870181, -52.67555993, 935.0],
                        [-23.70833581, -51.03976658, 935.0],
                        [-24.0994583, -52.19448522, 935.0],
                        [-24.36170654, -52.86064998, 935.0],
                        [-24.17208351, -50.01961616, 935.0],
                        [-23.63157396, -49.76620254, 935.0],
                        [-23.02581538, -52.8103863, 935.0],
                        [-24.53030951, -53.85851116, 935.0],
                        [-25.72041993, -51.26124428, 935.0],
                        [-24.84820667, -51.42181062, 935.0],
                        [-23.32281152, -50.33646275, 935.0],
                        [-24.20921727, -51.38527814, 935.0],
                        [-23.48428711, -53.24842583, 935.0],
                        [-25.2547867, -48.71641353, 935.0],
                        [-24.66839792, -50.65967657, 935.0],
                        [-26.03148476, -53.2408701, 935.0],
                        [-26.03427782, -51.868153, 935.0],
                        [-25.62400935, -49.89004677, 935.0],
                        [-25.44005846, -53.37693271, 935.0],
                        [-25.95367629, -52.58149423, 935.0],
                        [-24.89614112, -49.31104506, 935.0],
                        [-23.13667294, -51.1806208, 935.0],
                        [-23.44671885, -52.37700765, 935.0]])
    return sensors


@jit(nopython=True, cache=True, fastmath=True)
def get_sensor_matrix(sensors: np.ndarray,
                      wave_speed: np.float64 = AVG_LIGHT_SPEED,
                      sistema_cartesiano: bool = False) -> np.ndarray:
    """
    Computes the Sensor Time-To-Travel Assiciation Matrix

    Parameters
    ----------
    sensors: np.ndarray
        list of georeferenced positions of sensors in the network
    wave_speed: np.float64
        wave propagation speed (default is the speed of light)
    sistema_cartesiano : bool, optional
        If True, uses Euclidean (Cartesian) distance. If False, uses spherical distance (default: False).

    Returns
    -------
    sensor_tt: np.ndarray
        The Time-to-Travel matrix
    """
    sensor_tt = np.empty((len(sensors), len(sensors)), dtype=np.float64)
    for i in range(len(sensors)):
        for j in range(len(sensors)):
            sensor_tt[i, j] = computa_distancia(sensors[i],
                                                sensors[j],
                                                sistema_cartesiano)/wave_speed

    return sensor_tt


@jit(nopython=True, cache=True, fastmath=True)
def get_lightning_limits(sensores_latlon: np.ndarray,
                         margem_metros: np.float64 = 200.0) -> tuple:
    """
    Computes geographic bounding box around sensor constellation with an additional margin.

    Parameters
    ----------
    sensores_latlon : np.ndarray
        Array of shape (N, 2) with latitude and longitude in degrees.
    margem_metros : np.float64, optional
        Margin in meters around the bounding box (default is 200000.0).

    Returns
    -------
    tuple
        (min_lat, max_lat, min_lon, max_lon) with margin applied.
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
def get_lightning_area_grid(min_lat: np.float64,
                            max_lat: np.float64,
                            min_lon: np.float64,
                            max_lon: np.float64,
                            step: np.float64 = 1000.0) -> np.ndarray:
    """

    Generates a grid inside a ligntning area

    Parameters
    ----------
    min_lat : np.float64
        Minimum latitude.
    max_lat : np.float64
        Maximum latitude.
    min_lon : np.float64
        Minimum longitude.
    max_lon : np.float64
        Maximum longitude.
    step: np.float64
        the grid step in meters

    Returns
    -------
    grid : np.ndarray
        the lightning area grid points
    """
    avg_lat = (min_lat + max_lat) / 2.0
    delta_lat = step / R_LAT
    delta_lon = step / (R_LAT * np.cos(np.radians(avg_lat)))

    n_lat = int(np.floor((max_lat - min_lat) / delta_lat)) + 1
    n_lon = int(np.floor((max_lon - min_lon) / delta_lon)) + 1

    total = n_lat * n_lon
    grid = np.empty((total, 3), dtype=np.float64)

    idx = 0
    for i in range(n_lat):
        for j in range(n_lon):
            lat = min_lat + i * delta_lat + delta_lat / 2
            lon = min_lon + j * delta_lon + delta_lon / 2
            grid[idx, 0] = lat
            grid[idx, 1] = lon
            grid[idx, 2] = 935.0
            idx += 1

    return grid


@jit(nopython=True, cache=True, fastmath=True)
def compute_network_detection_efficiency(sensors: np.ndarray,
                                         margem_metros: np.float64,
                                         step: np.float64,
                                         num_events: np.int32 = 1000,
                                         sistema_cartesiano: bool = False) -> tuple:
    """
    Computes Network Detection Efficienty (NDE) by simulation

    Parameters
    ----------

    sensores : np.ndarray
        Array of shape (N, 2) with latitude and longitude in degrees.
    margem_metros : np.float64, optional
        Margin in meters around the bounding box.
    step: np.float64
        the grid step in meters
    num_events : np.int32
        Number of events to generate.
    sistema_cartesiano : bool, optional
        If True, uses Euclidean (Cartesian) distance. If False, uses spherical distance (default: False).

    Returns
    -------
    tuple
        (grid: np.ndarray, nde: np.ndarray)

    """
    (min_lat,
     max_lat,
     min_lon,
     max_lon) = get_lightning_limits(sensors,
                                     margem_metros)

    lightning_grid = get_lightning_area_grid(min_lat,
                                             max_lat,
                                             min_lon,
                                             max_lon,
                                             step)

    n = len(lightning_grid)

    nde = np.zeros(n, dtype=np.float64)

    for i in range(n):

        distancias = computa_distancias(lightning_grid[i],
                                        sensors,
                                        sistema_cartesiano)
        current_nde = 0.0
        # eficiência de detecção
        for _ in range(num_events):

            detections = 0

            for j in range(sensors.shape[0]):

                if sensor_detection(distancias[j]):
                    detections += 1

            if detections >= 3:
                current_nde += 1
        nde[i] = current_nde / num_events

    return lightning_grid, nde


def generate_events(num_events: np.int32,
                    min_lat: np.float64,
                    max_lat: np.float64,
                    min_lon: np.float64,
                    max_lon: np.float64,
                    min_alt: np.float64,
                    max_alt: np.float64,
                    min_time: np.float64,
                    max_time: np.float64,
                    sigma_t: np.float64 = SIGMA_T,
                    max_attempts: np.int32 = 10000) -> tuple:
    """
    Generates atmospheric events with spatial and temporal attributes,
    ensuring a minimum temporal spacing of 6 * sigma_t.

    Parameters
    ----------
    num_events : np.int32
        Number of events to generate.
    min_lat : np.float64
        Minimum latitude.
    max_lat : np.float64
        Maximum latitude.
    min_lon : np.float64
        Minimum longitude.
    max_lon : np.float64
        Maximum longitude.
    min_alt : np.float64
        Minimum altitude.
    max_alt : np.float64
        Maximum altitude.
    min_time : np.float64
        Minimum timestamp.
    max_time : np.float64
        Maximum timestamp.
    sigma_t : np.float64, optional
        Temporal standard deviation (default is SIGMA_T).
    max_attempts : np.int32, optional
        Maximum number of attempts to generate valid timestamps (default is 10000).

    Returns
    -------
    tuple of np.ndarray
        event_positions : (N, 3) array of (lat, lon, alt)
        event_times : (N,) array of timestamps
    """

    np.random.seed(42)
    lats = np.random.uniform(min_lat, max_lat, num_events)
    lons = np.random.uniform(min_lon, max_lon, num_events)
    alts = np.random.uniform(min_alt, max_alt, num_events)

    event_times = np.linspace(min_time, max_time, num_events)

    event_positions = np.stack((lats, lons, alts), axis=1)

    return event_positions, event_times


def generate_detections(event_positions: np.ndarray,
                        event_times: np.ndarray,
                        sensor_positions: np.ndarray,
                        jitter_std: np.float64 = SIGMA_T) -> tuple:
    """
    Simulates detections from a sensor network based on lightning events.

    Parameters
    ----------
    event_positions : np.ndarray
        Array of event coordinates with shape (N, 3).
    event_times : np.ndarray
        Array of event timestamps with shape (N,).
    sensor_positions : np.ndarray
        Array of sensor coordinates with shape (M, 3).
    jitter_std : np.float64, optional
        Standard deviation of temporal noise added to detection times (default is SIGMA_T).

    Returns
    -------
    tuple
        detections : np.ndarray
            Array of sensor positions that detected events.
        detection_times : np.ndarray
            Corresponding detection timestamps.
        n_event_positions : np.ndarray
            Event positions repeated for each detection (for traceability).
        n_event_times : np.ndarray
            Event times repeated for each detection (for traceability).
        distances : np.ndarray
            Propagation distances between each detection pair.
        sensor_indexes: np.ndarray
            sensor indexes association with time of detections (phase one)
        spatial_clusters : np.ndarray
            Cluster IDs identifying to which event each detection belongs.
    """

    # np.random.seed(42)
    detections = []
    detection_times = []
    n_event_positions = []
    n_event_times = []
    distances = []
    spatial_clusters = []
    sensor_indexes = []

    cluster_id = 0

    for i in range(len(event_positions)):
        while True:
            event_position = event_positions[i]
            event_time = event_times[i]
            event_distances = computa_distancias(
                event_position, sensor_positions)
            t_detections = []
            t_detection_times = []
            t_n_event_positions = []
            t_n_event_times = []
            t_distances = []
            t_spatial_clusters = []
            t_sensor_indexes = []
            for j in range(sensor_positions.shape[0]):
                noise = np.clip(np.random.normal(0.0, jitter_std),
                                min=-12 * jitter_std,
                                max=12 * jitter_std)
                t_detect = event_time + \
                    event_distances[j] / AVG_LIGHT_SPEED + noise
                if sensor_detection(event_distances[j]):
                    t_detections.append(sensor_positions[j])
                    t_detection_times.append(t_detect)
                    t_n_event_positions.append(event_position)
                    t_n_event_times.append(event_time)
                    t_distances.append(event_distances[j])
                    t_spatial_clusters.append(cluster_id)
                    t_sensor_indexes.append(j)
            if len(t_detections) >= 3:
                cluster_id += 1
                detections += t_detections
                detection_times += t_detection_times
                n_event_positions += t_n_event_positions
                n_event_times += t_n_event_times
                distances += t_distances
                spatial_clusters += t_spatial_clusters
                sensor_indexes += t_sensor_indexes
                break

    return (np.array(detections),
            np.array(detection_times),
            np.array(n_event_positions),
            np.array(n_event_times),
            np.array(distances),
            np.array(sensor_indexes),
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
