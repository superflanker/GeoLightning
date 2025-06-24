"""
    EELT 7019 - Inteligência Artificial Aplicada
    Testes de Metaheurísticas
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
import os
from GeoLightning.Simulator.Simulator import (get_sensors,
                                              get_random_sensors,
                                              get_lightning_limits,
                                              generate_detections,
                                              generate_events)

from GeoLightning.Utils.Constants import *
from GeoLightning.Runners.RunnerAOA import runner_AOA


def test_runner_AOA():

    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # recuperando o grupo de sensores
    sensors = get_sensors()
    min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors)

    # gerando os eventos
    min_alt = 935
    max_alt = 935
    min_time = 10000
    max_time = min_time + 72 * 3600
    num_events = [2, 5, 10, 50, 100, 1000, 2000]

    desired_sigma_t = [SIGMA_T,
                       2 * SIGMA_T,
                       3 * SIGMA_T,
                       4 * SIGMA_T,
                       5 * SIGMA_T,
                       6 * SIGMA_T,
                       7 * SIGMA_T,
                       8 * SIGMA_T,
                       9 * SIGMA_T,
                       10 * SIGMA_T]

    for sigma_t in desired_sigma_t:

        for i in range(len(num_events)):

            event_positions, event_times = generate_events(num_events[i],
                                                           min_lat,
                                                           max_lat,
                                                           min_lon,
                                                           max_lon,
                                                           min_alt,
                                                           max_alt,
                                                           min_time,
                                                           max_time,
                                                           sigma_t=sigma_t)

            # gerando as detecções
            (detections,
             detection_times,
             n_event_positions,
             n_event_times,
             distances,
             spatial_clusters) = generate_detections(event_positions,
                                                     event_times,
                                                     sensors)

            # tudo pronto, rodando o runner

            (sol_centroides_espaciais,
             sol_centroides_temporais,
             sol_detectores,
             sol_best_fitness,
             sol_reference,
             delta_d,
             delta_t,
             execution_time,
             associacoes_corretas) = runner_AOA(event_positions,
                                                event_times,
                                                spatial_clusters,
                                                detections,
                                                detection_times,
                                                sensors,
                                                min_alt,
                                                max_alt,
                                                CLUSTER_MIN_PTS,
                                                sigma_t,
                                                AVG_LIGHT_SPEED * sigma_t,
                                                EPSILON_T,
                                                AVG_LIGHT_SPEED,
                                                False)

            assert len(sol_centroides_temporais) == len(event_positions)



            # calculando rmse, amse, mae, 

if __name__ == "__main__":
    test_runner_AOA()
