"""
    EELT 7019 - Inteligência Artificial Aplicada
    Teste de Clusterização Temporal
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Stela.TemporalClustering import clusterizacao_temporal_stela

import os
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def test_temporal_clustering():

    from GeoLightning.Utils.Utils import computa_tempos_de_origem
    from time import perf_counter

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

        n_event_times = np.load(file_n_event_times)
        n_event_positions = np.load(file_n_event_positions)
        detection_times = np.load(file_detections_times)
        detection_positions = np.load(file_detections)
        spatial_clustering = np.load(file_spatial_clusters)


        tempos_de_origem = computa_tempos_de_origem(n_event_positions, 
                                           spatial_clustering, 
                                           detection_times, 
                                           detection_positions)
        labels = clusterizacao_temporal_stela(tempos_de_origem)
        
        assert len(np.unique(labels)) == num_events[i]
