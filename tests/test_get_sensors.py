"""
EELT 7019 - Applied Artificial Intelligence
Sensor Simulation Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Simulator.Simulator import (get_sensors,
                                              get_sensor_matrix,
                                              get_random_sensors,
                                              get_lightning_limits,
                                              generate_detections,
                                              generate_events)
from GeoLightning.Utils.Constants import *

from GeoLightning.Stela.Stela import stela_phase_one, stela_phase_two


def test_simulations():

    # recuperando o grupo de sensores
    sensors = get_sensors()
    sensor_tt = get_sensor_matrix(sensors, AVG_LIGHT_SPEED, False)
    min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors)

    # gerando os eventos
    min_alt = 0
    max_alt = 10000
    min_time = 10000
    max_time = min_time + 72 * 3600
    num_events = 100

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
     sensor_indexes,
     spatial_clusters) = generate_detections(event_positions,
                                             event_times,
                                             sensors)

    # clusterizando

    clusters_espaciais = stela_phase_one(detection_times,
                                         sensor_indexes,
                                         sensor_tt,
                                         EPSILON_T/1.5,
                                         CLUSTER_MIN_PTS)

    assert len(np.unique(clusters_espaciais)) == len(event_positions)

    sensors = get_random_sensors(7,
                                 min_lat,
                                 min_lat,
                                 min_lon,
                                 max_lon,
                                 min_alt,
                                 max_alt)

    sensor_tt = get_sensor_matrix(sensors, AVG_LIGHT_SPEED, False)

    # gerando as detecções
    (detections,
     detection_times,
     n_event_positions,
     n_event_times,
     distances,
     sensor_indexes,
     spatial_clusters) = generate_detections(event_positions,
                                             event_times,
                                             sensors)

    # clusterizando
    clusters_espaciais = stela_phase_one(detection_times,
                                         sensor_indexes,
                                         sensor_tt,
                                         EPSILON_T/1.5,
                                         CLUSTER_MIN_PTS)

    assert len(np.unique(clusters_espaciais)) == len(event_positions)
