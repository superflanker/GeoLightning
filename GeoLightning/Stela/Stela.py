"""
STELA Core algorithm
====================

Spatio-Temporal Event Likelihood Assignment (STELA) Algorithm

Summary
-------
This module implements the STELA algorithm, designed for spatio-temporal association 
of multisensory detections with simulated physical events, such as lightning strikes 
or impulsive acoustic sources.

The algorithm integrates spatial and temporal clustering with a likelihood function 
based on the consistency between the time-of-arrival (TOA) of signals and the estimated 
event positions. It refines candidate solutions from multilateration, adjusts 
search boundaries for meta-heuristics, and identifies plausible groupings 
of detections that correspond to real-world events.

This pipeline is compatible with TOA-based localization methods and applicable 
to geophysical sensing, radio frequency, underwater acoustics, and transient astronomy.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- STELA Algorithm (main routine - phase one and two)
- Temporal clustering
- Spatial clustering and likelihood estimation
- Search bounds generation

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
- GeoLightning.Stela.LogLikelihood
- GeoLightning.Stela.Common
"""

import numpy as np
from numba import jit
from GeoLightning.Utils.Constants import SIGMA_T, \
    EPSILON_T, \
    CLUSTER_MIN_PTS, \
    AVG_LIGHT_SPEED, \
    AVG_EARTH_RADIUS
from GeoLightning.Utils.Utils import computa_tempos_de_origem
from GeoLightning.Stela.DBSCAN import dbscan
from GeoLightning.Stela.LogLikelihood import funcao_log_verossimilhanca
from GeoLightning.Stela.Common import computa_residuos_temporais, calcular_media_clusters


@jit(nopython=True, cache=True, fastmath=True)
def stela_phase_one(tempos_de_chegada: np.ndarray,
                    indices_sensores: np.ndarray,
                    sensor_tt: np.ndarray,
                    epsilon_t: np.float64 = EPSILON_T,
                    min_pts: np.int32 = CLUSTER_MIN_PTS) -> np.ndarray:
    """
    Spatio-Temporal Event Likelihood Assignment (STELA) Algorithm - Clustering Phase.

    This function performs the core association and filtering step based on 
    the spatio-temporal consistency between multisensory detections using TDOA consistency criteria.

    Parameters
    ----------
    tempos_de_chegada : np.ndarray
        Array of shape (M,) with absolute signal arrival times at the sensors.
    indice_sensores: np.ndarray
        Array of sensor IDs associated with times
    sensor_tt: np.ndarray
        Association matrix with time-to-travel in light speed of the distances
        between sensors
    epsilon_t : float, optional
        Temporal tolerance for spatio-temporal clustering.
    min_pts : int, optional
        Minimum number of points to form a valid cluster (DBSCAN requirement).

    Returns
    -------
    clusters_espaciais: np.ndarray
        list of clusters of asssociated detections

    Notes
    -----
    - Optimized with Numba for high-performance execution.
    - Compatible with multiple events and multisensor contexts.
    - Suitable for pre-processing before global optimization with genetic algorithms,
      swarm intelligence, and other meta-heuristics.
    """
    clusters_espaciais = dbscan(tempos_de_chegada,
                                indices_sensores,
                                sensor_tt,
                                epsilon_t,
                                min_pts)
    return clusters_espaciais


@jit(nopython=True, cache=True, fastmath=True)
def stela_phase_two(solucoes: np.ndarray,
                    clusters_espaciais: np.ndarray,
                    tempos_de_chegada: np.ndarray,
                    pontos_de_deteccao: np.ndarray,
                    sistema_cartesiano: bool = False,
                    sigma_t: np.float64 = SIGMA_T,
                    c: np.float64 = AVG_LIGHT_SPEED) -> np.float64:
    """
    Spatio-Temporal Event Likelihood Assignment (STELA) Algorithm - Refinement Phase.

    This function performs the position index for a candidate event position.

    Parameters
    ----------
    solucao : np.ndarray
        Array of shape (N, 3) with candidate event positions in geographic 
        or Cartesian coordinates.
    clusters_espaciais: np.ndarray
        clusters of asssociated detections
    tempos_de_chegada : np.ndarray
        Array of shape (M,) with absolute signal arrival times at the sensors.
    pontos_de_deteccao : np.ndarray
        Array of shape (M, 3) with sensor positions, using the same coordinate system 
        as `solucoes`.
    sistema_cartesiano : bool, optional
        Indicates whether the coordinates are Cartesian (True) or geographic (False). Default is False.
    sigma_t : float, optional
        Temporal standard deviation (used in likelihood computation).

    Returns
    -------
    verossimilhanca : float
        Total log-likelihood value of the solution.

    Notes
    -----
    - Optimized with Numba for high-performance execution.
    - Compatible with multiple events and multisensor contexts.
    - Suitable for pre-processing before global optimization with genetic algorithms,
      swarm intelligence, and other meta-heuristics.
    """

    tempos_de_origem = computa_tempos_de_origem(solucoes,
                                                clusters_espaciais,
                                                tempos_de_chegada,
                                                pontos_de_deteccao,
                                                sistema_cartesiano)
    # média dos tempos por cluster
    centroides_temporais, _ = calcular_media_clusters(tempos_de_origem,
                                                   clusters_espaciais)
    # residuos temporais
    residuos_temporais = computa_residuos_temporais(centroides_temporais,
                                                    clusters_espaciais,
                                                    tempos_de_origem)    

    # distancias a partir do ponto de origem
    residuos_espaciais = c * np.abs(residuos_temporais)

    # verossimilhança
    verossimilhanca = funcao_log_verossimilhanca(residuos_espaciais, c * sigma_t)

    # tudo pronto, retornando
    return verossimilhanca


if __name__ == "__main__":

    from GeoLightning.Simulator.Simulator import (get_sensors,
                                                  get_random_sensors,
                                                  get_sensor_matrix,
                                                  get_lightning_limits,
                                                  generate_detections,
                                                  generate_events)
    from time import perf_counter

    num_events = [1, 2, 5, 10, 15, 20, 25,
                  30, 100, 500, 800, 1000, 5000, 20000]

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
         sensor_indexes,
         spatial_clusters) = generate_detections(event_positions,
                                                 event_times,
                                                 sensors)
        start_st = perf_counter()

        clusters_espaciais = stela_phase_one(detection_times,
                                             sensor_indexes,
                                             sensor_tt,
                                             2 * EPSILON_T,
                                             CLUSTER_MIN_PTS)

        end_st = perf_counter()
        print(
            f"Eventos: {num_events[i]}, Tempo gasto: {end_st - start_st} Segundos")
        len_clusterizados = len(
            np.unique(clusters_espaciais[clusters_espaciais >= 0]))
        len_reais = len(event_positions)
        print(len_clusterizados, len_reais)

        correct_association_index = np.mean(spatial_clusters == clusters_espaciais) * 100
        print(correct_association_index)
        try:
            assert len_clusterizados == len_reais
            assert spatial_clusters == clusters_espaciais
        except:
            print(f"Clusterizados: {len_clusterizados}, Reais: {len_reais}")
