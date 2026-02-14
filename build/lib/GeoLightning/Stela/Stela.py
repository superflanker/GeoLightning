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
from typing import Tuple
from GeoLightning.Utils.Constants import SIGMA_T, \
    EPSILON_T, \
    CLUSTER_MIN_PTS, \
    AVG_LIGHT_SPEED, \
    AVG_EARTH_RADIUS
from GeoLightning.Utils.Utils import computa_tempos_de_origem
from GeoLightning.Stela.PivotClustering import pivot_clustering
from GeoLightning.Stela.LogLikelihood import funcao_log_verossimilhanca
from GeoLightning.Stela.Common import computa_residuos_temporais, calcular_media_clusters


@jit(nopython=True, cache=True, fastmath=True)
def cluster_cleanup(labels: np.ndarray,
                    min_pts: np.int32) -> None:
    """
    Prune clusters with fewer than `min_pts` points (in place).

    This function assumes that `labels` is ordered such that equal labels form
    contiguous segments (i.e., each cluster appears as a single block). Each
    block whose length is strictly smaller than `min_pts` is relabeled as -1.

    Parameters
    ----------
    labels : np.ndarray
        One-dimensional array of integer labels. Clusters are identified by
        positive integers. The value -1 is treated as noise/outlier.
        The array is modified in place.

    min_pts : np.int32
        Minimum number of points required for a label block to be kept.
    """
    n = len(labels)
    if n == 0:
        return

    cluster_start = 0
    current_label = labels[0]

    # Scan transitions between contiguous label blocks
    for i in range(1, n):
        if labels[i] != current_label:
            # Close the block [cluster_start, i)
            if current_label != -1:
                num_points = i - cluster_start
                if num_points < min_pts:
                    labels[cluster_start:i] = -1

            # Start a new block
            cluster_start = i
            current_label = labels[i]

    # Process the last block [cluster_start, n)
    if current_label != -1:
        num_points = n - cluster_start
        if num_points < min_pts:
            labels[cluster_start:n] = -1


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

    Returns
    -------
    tempos_ordenados : np.ndarray
        Copy of ``tempos`` sorted in ascending order.

    indices_sensores_ordenados : np.ndarray
        Copy of ``indices_sensores`` permuted by the same ordering applied to
        ``tempos_ordenados``.

    labels : np.ndarray
        One-dimensional array of integer cluster labels aligned with
        ``tempos_ordenados``. Labels start at 1 and increase sequentially as new
        clusters are created.

    Notes
    -----
    - Optimized with Numba for high-performance execution.
    - Compatible with multiple events and multisensor contexts.
    - Suitable for pre-processing before global optimization with genetic algorithms,
      swarm intelligence, and other meta-heuristics.
    """

    (tempos_ordenados,
     indices_sensores_ordenados,
     labels,
     ordered_indexes) = pivot_clustering(tempos=tempos_de_chegada,
                                         indices_sensores=indices_sensores,
                                         sensor_tt=sensor_tt,
                                         eps=epsilon_t)

    # mais uma ordenação, agora por label

    label_ordered_indexes = np.argsort(labels)

    tempos_ordenados = tempos_ordenados[label_ordered_indexes]

    indices_sensores_ordenados = indices_sensores_ordenados[label_ordered_indexes]

    labels = labels[label_ordered_indexes]

    ordered_indexes = ordered_indexes[label_ordered_indexes]

    cluster_cleanup(labels=labels,
                    min_pts=min_pts)

    return (tempos_ordenados,
            indices_sensores_ordenados,
            labels,
            ordered_indexes)


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

    # distancias residuais a partir do ponto de origem
    residuos_espaciais = c * np.abs(residuos_temporais)

    # verossimilhança
    verossimilhanca = funcao_log_verossimilhanca(
        residuos_espaciais, c * sigma_t)

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
                  30, 100, 500, 800, 1000, 5000, 10000, 100000, 1000000]

    time_multipliers = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # num_events = [5, 10, 100]

    # recuperando o grupo de sensores
    sensors = get_sensors()
    sensor_tt = get_sensor_matrix(sensors, AVG_LIGHT_SPEED, False)
    min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors, 3000)

    # gerando os eventos
    min_alt = 935.0
    max_alt = 935.0
    min_time = 10000

    delta_time = 0.0

    for i in range(len(sensor_tt)):
        for j in range(i, len(sensor_tt[i])):
            if sensor_tt[i, j] > delta_time:
                delta_time = sensor_tt[i, j]

    for multiplier in time_multipliers:

        current_delta_time = delta_time * multiplier

        for i in range(len(num_events)):

            max_time = min_time + num_events[i] * current_delta_time

            event_positions, event_times = generate_events(num_events=num_events[i],
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
                                                     sensor_positions=sensors,
                                                     simulate_complete_detections=True,
                                                     fixed_seed=False,
                                                     min_pts=CLUSTER_MIN_PTS)

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
            print()
            print(f"Tempo Entre Eventos: {current_delta_time:.10f} segundos")
            print(
                f"Eventos: {num_events[i]}, Tempo gasto: {end_st - start_st:0.8f} segundos")
            len_clusterizados = len(
                np.unique(clusters_espaciais[clusters_espaciais >= 0]))
            len_reais = len(event_positions)

            print(
                f"Clusterizados: {len_clusterizados}, Reais: {len_reais} ({100 * len_clusterizados/len_reais:.4f} %)")
        print("------------------------------------------------------", "\n")
