"""
EELT 7019 - Applied Artificial Intelligence
STELA Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Stela.Stela import stela_phase_one, stela_phase_two_phase_one, stela_phase_two

from GeoLightning.Simulator.Simulator import (get_sensors,
                                              get_random_sensors,
                                              get_lightning_limits,
                                              generate_detections,
                                              generate_events)

from GeoLightning.Stela.LogLikelihood import maxima_log_verossimilhanca

from GeoLightning.Utils.Constants import SIGMA_D


def test_stela():

    num_events = [2, 5, 10, 15, 20, 25,
                  30, 100, 500, 800, 1000]

    for i in range(len(num_events)):
        # recuperando o grupo de sensores
        sensors = get_sensors()
        min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors)

        # gerando os eventos
        min_alt = 935.0
        max_alt = 935.0
        min_time = 10000
        max_time = min_time + 72 * 3600

        event_positions, event_times = generate_events(num_events[i],
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
        (clusters_espaciais,
         verossimilhanca) = stela_phase_one(n_event_positions,
                                            detection_times,
                                            detections,
                                            sistema_cartesiano=False)

        assert len(np.unique(clusters_espaciais)) == len(event_positions)

    sensors = get_sensors()
    min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors)

    # gerando os eventos
    min_alt = 935.0
    max_alt = 935.0
    min_time = 10000
    max_time = min_time + 72 * 3600

    event_positions, event_times = generate_events(1,
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

    verossimilhanca = stela_phase_two(event_positions[0],
                                      detection_times,
                                      detections)

    maxima_verossimilhanca = maxima_log_verossimilhanca(
        len(detection_times), SIGMA_D)
    assert np.isclose(verossimilhanca, maxima_verossimilhanca, 10)


if __name__ == "__main__":
    test_stela()
