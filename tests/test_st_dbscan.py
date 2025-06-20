"""
    EELT 7019 - Inteligência Artificial Aplicada
    Testes DBSCAN3D
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Simulator.Simulator import (get_sensors,
                                              get_random_sensors,
                                              get_lightning_limits,
                                              generate_detections,
                                              generate_events)

from GeoLightning.Stela.STDBSCAN import st_dbscan
from GeoLightning.Utils.Constants import *


def test_st_dbscan():

    # recuperando o grupo de sensores
    sensors = get_sensors()
    min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors)

    # gerando os eventos
    min_alt = 0
    max_alt = 10000
    min_time = 10000
    max_time = 10100
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
     spatial_clusters) = generate_detections(event_positions,
                                             event_times,
                                             sensors)

    (labels,
     distancias,
     centroides,
     detectores,
     novas_solucoes,
     verossimilhanca) = st_dbscan(n_event_positions,
                                  detection_times,
                                  spatial_clusters,
                                  EPSILON_D,
                                  EPSILON_T,
                                  SIGMA_D,
                                  CLUSTER_MIN_PTS,
                                  False)

    assert len(np.unique(labels)) == len(event_positions)

    sensors = get_random_sensors(7,
                                 min_lat,
                                 min_lat,
                                 min_lon,
                                 max_lon,
                                 min_alt,
                                 max_alt)

    # gerando as detecções
    (detections,
     detection_times,
     n_event_positions,
     n_event_times,
     distances,
     spatial_clusters) = generate_detections(event_positions,
                                             event_times,
                                             sensors)

    (labels,
     distancias,
     centroides,
     detectores,
     novas_solucoes,
     verossimilhanca) = st_dbscan(n_event_positions,
                                  detection_times,
                                  spatial_clusters,
                                  EPSILON_D,
                                  EPSILON_T,
                                  SIGMA_D,
                                  CLUSTER_MIN_PTS,
                                  False)

    assert len(np.unique(labels)) == len(event_positions)
