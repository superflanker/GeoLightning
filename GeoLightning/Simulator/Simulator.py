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
def get_lightning_limits(sensores_latlon: np.ndarray,
                         margem_metros: float = -40000.0) -> tuple:
    """
    Computes geographic bounding box around sensor constellation with an additional margin.

    Parameters
    ----------
    sensores_latlon : np.ndarray
        Array of shape (N, 2) with latitude and longitude in degrees.
    margem_metros : float, optional
        Margin in meters around the bounding box (default is 5000.0).

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

def generate_events(num_events: int,
                    min_lat: float,
                    max_lat: float,
                    min_lon: float,
                    max_lon: float,
                    min_alt: float,
                    max_alt: float,
                    min_time: float,
                    max_time: float,
                    sigma_t: float = SIGMA_T,
                    max_attempts: int = 10000) -> tuple:
    """
    Generates atmospheric events with spatial and temporal attributes,
    ensuring a minimum temporal spacing of 6 * sigma_t.

    Parameters
    ----------
    num_events : int
        Number of events to generate.
    min_lat : float
        Minimum latitude.
    max_lat : float
        Maximum latitude.
    min_lon : float
        Minimum longitude.
    max_lon : float
        Maximum longitude.
    min_alt : float
        Minimum altitude.
    max_alt : float
        Maximum altitude.
    min_time : float
        Minimum timestamp.
    max_time : float
        Maximum timestamp.
    sigma_t : float, optional
        Temporal standard deviation (default is SIGMA_T).
    max_attempts : int, optional
        Maximum number of attempts to generate valid timestamps (default is 10000).

    Returns
    -------
    tuple of np.ndarray
        event_positions : (N, 3) array of (lat, lon, alt)
        event_times : (N,) array of timestamps
    """

    min_dt = 6.0 * sigma_t
    """event_times = []

    attempts = 0
    while len(event_times) < num_events and attempts < max_attempts:
        candidate = np.random.uniform(min_time, max_time)
        if all(abs(candidate - t) >= min_dt for t in event_times):
            event_times.append(candidate)
        attempts += 1

    if len(event_times) < num_events:
        raise RuntimeError("Unable to generate events with minimum time spacing within the number of attempts.")
    """

    # event_times = np.array(sorted(event_times))
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
    jitter_std : float, optional
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
        spatial_clusters : np.ndarray
            Cluster IDs identifying to which event each detection belongs.
    """
    np.random.seed(42)
    detections = []
    detection_times = []
    n_event_positions = []
    n_event_times = []
    distances = []
    spatial_clusters = []

    cluster_id = 0

    for i in range(event_positions.shape[0]):
        while True:
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
                    event_distances[j] / AVG_LIGHT_SPEED  + noise
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
                break

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
