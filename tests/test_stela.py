"""
EELT 7019 - Applied Artificial Intelligence
STELA Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Stela.Stela import stela_phase_one, stela_phase_two

from GeoLightning.Simulator.Simulator import (get_sensors,
                                              get_sensor_matrix,
                                              get_lightning_limits,
                                              generate_detections,
                                              generate_events)

from GeoLightning.Stela.LogLikelihood import maxima_log_verossimilhanca, funcao_log_verossimilhanca

from GeoLightning.Utils.Constants import (SIGMA_T, 
                                          SIGMA_D, 
                                          AVG_LIGHT_SPEED, 
                                          CLUSTER_MIN_PTS, 
                                          EPSILON_T)

from time import perf_counter


def test_stela():

    num_events = [2, 5, 10, 15, 20, 25,
                  30, 100, 500, 800, 1000]
    for i in range(len(num_events)):
        # recuperando o grupo de sensores
        sensors = get_sensors()
        sensor_tt = get_sensor_matrix(sensors, AVG_LIGHT_SPEED, False)
        min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors)

        # gerando os eventos
        min_alt = 935.0
        max_alt = 935.0
        min_time = 10000
        max_time = min_time + 72 * 3600

        # protagonista da história - eventos
        event_positions, event_times = generate_events(num_events=num_events,
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
            sensor_indexes,
            spatial_clusters) = generate_detections(event_positions=event_positions,
                                                    event_times=event_times,
                                                    sensors_positions=sensors,
                                                    jitter_std=SIGMA_T,
                                                    simulate_complete_detections=True,
                                                    min_pts=CLUSTER_MIN_PTS)

        # tudo pronto, rodando a clusterização

        start_st = perf_counter()

        (tempos_ordenados,
            indices_sensores_ordenados,
            clusters_espaciais,
            ordered_indexes) = stela_phase_one(tempos_de_chegada=detection_times,
                                            indices_sensores=sensor_indexes,
                                            sensor_tt=sensor_tt,
                                            epsilon_t=EPSILON_T,
                                            min_pts=CLUSTER_MIN_PTS)


        end_st = perf_counter()

        print(
            f"Eventos: {num_events[i]}, Tempo gasto: {end_st - start_st} Segundos")
        len_clusterizados = len(
            np.unique(clusters_espaciais[clusters_espaciais >= 0]))
        len_reais = len(event_positions)
        print(len_clusterizados, len_reais)

        correct_association_index = np.mean(
            spatial_clusters == clusters_espaciais) * 100
        print(correct_association_index)
        try:
            assert len_clusterizados == len_reais
            assert spatial_clusters == clusters_espaciais
        except:
            print(f"Clusterizados: {len_clusterizados}, Reais: {len_reais}")

    # fase 2 - localização dos eventos
    sensors = get_sensors()
    min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors)

    # gerando os eventos
    min_alt = 935.0
    max_alt = 935.0
    min_time = 10000
    max_time = min_time + 72 * 3600

    event_positions, event_times = generate_events(10,
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

    verossimilhanca = stela_phase_two(event_positions,
                                      spatial_clusters,
                                      detection_times,
                                      detections,
                                      False,
                                      SIGMA_T,
                                      AVG_LIGHT_SPEED)

    maxima_verossimilhanca = maxima_log_verossimilhanca(
        len(detection_times), AVG_LIGHT_SPEED * SIGMA_T)
    print(verossimilhanca, maxima_verossimilhanca)
    print(len(detections))
    maxima_verossimilhanca = funcao_log_verossimilhanca(np.zeros(len(detections)), AVG_LIGHT_SPEED * SIGMA_T)
    print(maxima_verossimilhanca)
    assert np.isclose(verossimilhanca, maxima_verossimilhanca, 10)


if __name__ == "__main__":
    test_stela()
