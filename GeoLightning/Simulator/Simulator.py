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
from typing import Tuple
from numba.typed import List as nList
from GeoLightning.Utils.Constants import R_LAT, AVG_LIGHT_SPEED, SIGMA_T, CLUSTER_MIN_PTS
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


@jit(nopython=True)
def is_inside_hull(point: np.ndarray,
                   hull_vertices: np.ndarray) -> bool:
    n = len(hull_vertices)
    for i in range(n):
        p1 = hull_vertices[i]
        p2 = hull_vertices[(i + 1) % n]

        val = (p2[1] - p1[1]) * (point[0] - p1[0]) - \
              (p2[0] - p1[0]) * (point[1] - p1[1])

        if val > 0:
            return False
    return True


@jit(nopython=True)
def generate_latlon_inside_hull(num_events: np.int32,
                                hull_vertices: np.ndarray,
                                min_lat: np.float64,
                                max_lat: np.float64,
                                min_lon: np.float64,
                                max_lon: np.float64,
                                min_alt: np.float64,
                                max_alt: np.float64) -> Tuple[np.ndarray,
                                                              np.ndarray,
                                                              np.ndarray]:
    """
    Generates atmospheric events with spatial and temporal attributes inside a convex hull.

    Parameters
    ----------
    num_events : np.int32
        Number of events to generate.
    vertices_hull : np.ndarray
        vertices of the convex hull region for some network
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

    Returns
    -------
    lats: np.ndarray
        randomized latitudes inside convex hull
    lons: np.ndarray
        randomized longitudes inside convex hull
    alts:
        randomized altitudes 
    """

    lats = np.empty(num_events, dtype=np.float64)
    lons = np.empty(num_events, dtype=np.float64)
    alts = np.empty(num_events, dtype=np.float64)
    count = 0
    while count < num_events:
        lat_c = np.random.uniform(min_lat, max_lat)
        lon_c = np.random.uniform(min_lon, max_lon)
        alt_c = np.random.uniform(min_alt, max_alt)
        p_c = np.array([lat_c, lon_c])

        if is_inside_hull(p_c, hull_vertices):
            lats[count] = lat_c
            lons[count] = lon_c
            alts[count] = alt_c
            count += 1
    return lats, lons, alts


@jit(nopython=True, cache=True, fastmath=True)
def generate_events(num_events: np.int32,
                    vertices_hull: np.ndarray,
                    min_lat: np.float64,
                    max_lat: np.float64,
                    min_lon: np.float64,
                    max_lon: np.float64,
                    min_alt: np.float64,
                    max_alt: np.float64,
                    min_time: np.float64,
                    max_time: np.float64,
                    fixed_seed: bool = True) -> Tuple[np.ndarray,
                                                      np.ndarray]:
    """
    Generates atmospheric events with spatial and temporal attributes.

    Parameters
    ----------
    num_events : np.int32
        Number of events to generate.
    vertices_hull : np.ndarray
        vertices of the convex hull region for some network
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
    fixed_seed: bool
        Flag indicating whether the generation should fix the seed for traceability or not
    Returns
    -------
    tuple of np.ndarray
        event_positions : (N, 3) array of (lat, lon, alt)
        event_times : (N,) array of timestamps
    """
    if fixed_seed:
        np.random.seed(42)

    (lats,
     lons,
     alts) = generate_latlon_inside_hull(num_events=num_events,
                                         hull_vertices=vertices_hull,
                                         min_lat=min_lat,
                                         max_lat=max_lat,
                                         min_lon=min_lon,
                                         max_lon=max_lon,
                                         min_alt=min_alt,
                                         max_alt=max_alt)

    event_times = np.linspace(min_time, max_time, num_events)

    event_positions = np.stack((lats, lons, alts), axis=1)

    return event_positions, event_times


@jit(nopython=True, cache=True, fastmath=True)
def generate_detections(event_positions: np.ndarray,
                        event_times: np.ndarray,
                        sensor_positions: np.ndarray,
                        simulate_complete_detections: bool = True,
                        fixed_seed: bool = True,
                        min_pts: np.int32 = CLUSTER_MIN_PTS,
                        jitter_std: np.float64 = SIGMA_T) -> Tuple[np.ndarray,
                                                                   np.ndarray,
                                                                   np.ndarray,
                                                                   np.ndarray,
                                                                   np.ndarray,
                                                                   np.ndarray,
                                                                   np.ndarray]:
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
    simulate_complete_detections: bool
        Flag indicating that the generation sohould respect the min_pts parameter
    fixed_seed: bool
        Flag indicating whether the generation should fix the seed fr traceability or not
    min_pts: np.int32
        minimum points to consider a complete detection (Imposed by TOA/TDOA standards)
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

    num_events = len(event_positions)
    num_sensors = len(sensor_positions)

    max_size = num_events * num_sensors

    out_detections = np.empty((max_size, 3), dtype=np.float64)
    out_detection_times = np.empty(max_size, dtype=np.float64)
    out_event_positions = np.empty((max_size, 3), dtype=np.float64)
    out_event_times = np.empty(max_size, dtype=np.float64)
    out_distances = np.empty(max_size, dtype=np.float64)
    out_spatial_clusters = np.empty(max_size, dtype=np.int64)
    out_sensor_indexes = np.empty(max_size, dtype=np.int64)
    write_idx = 0
    cluster_id = 0

    if fixed_seed:
        np.random.seed(42)

    for i in range(num_events):
        event_pos = event_positions[i]
        event_t = event_times[i]

        while True:
            event_distances = computa_distancias(event_pos, sensor_positions)

            start_idx = write_idx

            for j in range(num_sensors):
                noise = np.random.normal(0.0, jitter_std)
                t_detect = event_t + \
                    (event_distances[j] / AVG_LIGHT_SPEED) + noise

                if sensor_detection(event_distances[j]):
                    out_detections[write_idx] = sensor_positions[j]
                    out_detection_times[write_idx] = t_detect
                    out_event_positions[write_idx] = event_pos
                    out_event_times[write_idx] = event_t
                    out_distances[write_idx] = event_distances[j]
                    out_spatial_clusters[write_idx] = cluster_id
                    out_sensor_indexes[write_idx] = j
                    write_idx += 1

            if simulate_complete_detections:

                if (write_idx - start_idx) >= min_pts:
                    cluster_id += 1
                    break
                else:
                    write_idx = start_idx
            else:
                cluster_id += 1
                break

    shuffled_indices = np.arange(write_idx)
    np.random.shuffle(shuffled_indices)

    return (out_detections[shuffled_indices],
            out_detection_times[shuffled_indices],
            out_event_positions[shuffled_indices],
            out_event_times[shuffled_indices],
            out_distances[shuffled_indices],
            out_sensor_indexes[shuffled_indices],
            out_spatial_clusters[shuffled_indices])


if __name__ == "__main__":

    # recuperando o grupo de sensores
    sensors = get_sensors()

    min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensores_latlon=sensors,
                                                              margem_metros=3000)

    from scipy.spatial import ConvexHull

    # sensores: array [[lat, lon], ...]
    hull = ConvexHull(sensors[:, :2])
    vertices_hull = sensors[hull.vertices, :2]  # Apenas os sensores da borda,

    # gerando os eventos
    min_alt = 0
    max_alt = 10000
    min_time = 10000
    max_time = 10100
    num_events = 500

    print(min_lat, max_lat, min_lon, max_lon, min_alt, max_alt)

    event_positions, event_times = generate_events(num_events=num_events,
                                                   vertices_hull=vertices_hull,
                                                   min_lat=min_lat,
                                                   max_lat=max_lat,
                                                   min_lon=min_lon,
                                                   max_lon=max_lon,
                                                   min_alt=min_alt,
                                                   max_alt=max_alt,
                                                   min_time=min_time,
                                                   max_time=max_time)

    # gerando as detecções
    (detections,
     detection_times,
     n_event_positions,
     n_event_times,
     distances,
     spatial_clusters) = generate_detections(event_positions=event_positions,
                                             event_times=event_times,
                                             sensors=sensors,
                                             jitter_std=SIGMA_T,
                                             simulate_complete_detections=True,
                                             min_pts=CLUSTER_MIN_PTS)

    print(detections)
    print(n_event_positions)
    print(detection_times)
    print(distances)
