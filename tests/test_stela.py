"""
    EELT 7019 - Inteligência Artificial Aplicada
    Teste do Algoritmo Principal
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Stela.Stela import stela

from GeoLightning.Simulator.Simulator import (get_sensors,
                                              get_random_sensors,
                                              get_lightning_limits,
                                              generate_detections,
                                              generate_events)

def test_stela():
    from GeoLightning.Stela.TemporalClustering import clusterizacao_temporal_stela
    from GeoLightning.Utils.Utils import computa_tempos_de_origem

    num_events = [2, 5, 10, 15, 20, 25,
                  30, 100, 500, 800, 1000]

    for i in range(len(num_events)):
        # recuperando o grupo de sensores
        sensors = get_sensors()
        min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors)

        # gerando os eventos
        min_alt = 0
        max_alt = 10000
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
        (lb,
         ub,
         centroides,
         detectores,
         clusters_espaciais,
         novas_solucoes,
         verossimilhanca) = stela(n_event_positions,
                                  detection_times,
                                  detections,
                                  spatial_clusters,
                                  sistema_cartesiano=False)
        
        assert len(np.unique(clusters_espaciais)) == len(event_positions)
