"""
    EELT 7019 - Inteligência Artificial Aplicada
    Testes de Clusteruzação Espacial
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Stela.SpatialClustering import clusterizacao_espacial_stela
from GeoLightning.Simulator.Simulator import (get_sensors,
                                              get_random_sensors,
                                              get_lightning_limits,
                                              generate_detections,
                                              generate_events)


import os
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def test_clusterizacao_espacial_stela_sintetico():

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
        max_time = 10100

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

        tempos_de_origem = computa_tempos_de_origem(n_event_positions,
                                                    spatial_clusters,
                                                    detection_times,
                                                    detections)

        labels = clusterizacao_temporal_stela(
            tempos_de_origem)

        (centroides,
         detectores,
         solucoes_unicas,
         final_clusters,
         novas_solucoes,
         loglikelihood) = clusterizacao_espacial_stela(n_event_positions,
                                                       labels.astype(
                                                           dtype=np.int32),
                                                       tempos_de_origem)

        assert len(np.unique(final_clusters)) == len(event_positions)
