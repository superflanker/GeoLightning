"""
    EELT 7019 - Inteligência Artificial Aplicada
    Teste do Algoritmo Principal
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Stela.Stela import stela

import os
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def test_stela():

    num_events = [2, 5, 10, 15, 20, 25,
                  30, 100, 500, 800, 1000]

    for i in range(len(num_events)):

        file_detections = basedir + "/../data/static_constellation_detections_{:06d}.npy".format(
            num_events[i])

        file_detections_times = basedir + "/../data/static_constelation_detection_times_{:06d}.npy".format(
            num_events[i])

        file_event_positions = basedir + "/../data/static_constelation_event_positions_{:06d}.npy".format(
            num_events[i])

        file_event_times = basedir + "/../data/static_constelation_event_times_{:06d}.npy".format(
            num_events[i])

        file_n_event_positions = basedir + "/../data/static_constelation_n_event_positions_{:06d}.npy".format(
            num_events[i])

        file_n_event_times = basedir + "/../data/static_constelation_n_event_times_{:06d}.npy".format(
            num_events[i])

        file_distances = basedir + "/../data/static_constelation_distances_{:06d}.npy".format(
            num_events[i])

        file_spatial_clusters = basedir + "/../data/static_constelation_spatial_clusters_{:06d}.npy".format(
            num_events[i])

        event_positions = np.load(file_event_positions)
        event_times = np.load(file_event_times)
        pontos_de_deteccao = np.load(file_detections)
        tempos_de_chegada = np.load(file_detections_times)
        solucoes = np.load(file_n_event_positions)
        # spatial_clusters = np.load(file_spatial_clusters)
        spatial_clusters = np.cumsum(
            np.ones(len(solucoes), dtype=np.int32)) - 1
        
        (lb,
         ub,
         centroides,
         detectores,
         clusters_espaciais,
         novas_solucoes,
         verossimilhanca) = stela(solucoes,
                                  tempos_de_chegada,
                                  pontos_de_deteccao,
                                  spatial_clusters,
                                  sistema_cartesiano=False)
        
        assert len(np.unique(clusters_espaciais)) == num_events[i]
