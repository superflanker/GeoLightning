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
from GeoLightning.Utils.Utils import  coordenadas_esfericas_para_cartesianas_batelada, \
    computa_distancias, \
        computa_distancia_batelada
from GeoLightning.Stela.LogLikelihood import funcao_log_verossimilhanca
from GeoLightning.Stela.Common import calcula_distancias_ao_centroide, \
    calcular_centroides_espaciais
from sklearn.cluster import DBSCAN, OPTICS


def gera_novas_solucoes(solucoes: np.ndarray,
                        labels: np.ndarray,
                        centroides) -> np.ndarray:
    """
    Fill an array with centroids solutions
    Parameters
    ----------
    solucoes : np.ndarray
        Array of estimated solutions.
    labels : np.ndarray
        Array of cluster labels assigned to each solution point.
    centroides : np.ndarray
        Array of centroids for each clusters
    Returns
    -------
    novas_solucoes: np.ndarray
        Array containing new centroids solutions to each 
    """
    if np.any(labels == -1):
        c_labels = labels + 1
    else:
        c_labels = labels.copy()
    novas_solucoes = np.empty_like(solucoes)

    for i in range(len(solucoes)):
        novas_solucoes[i] = centroides[c_labels[i]]
    return novas_solucoes


def stela_phase_one(solucoes: np.ndarray,
                    tempos_de_chegada: np.ndarray,
                    pontos_de_deteccao: np.ndarray,
                    sistema_cartesiano: bool = False,
                    sigma_t: np.float64 = SIGMA_T,
                    epsilon_t: np.float64 = EPSILON_T,
                    min_pts: np.int32 = CLUSTER_MIN_PTS,
                    c: np.float64 = AVG_LIGHT_SPEED) -> tuple:
    """
    Spatio-Temporal Event Likelihood Assignment (STELA) Algorithm - Clustering Phase.

    This function performs the core association and filtering step based on 
    the spatio-temporal consistency between multisensory detections and 
    multilateration-generated candidate event positions.

    It applies temporal clustering to infer origin times, followed by a spatial 
    clustering step with compatibility checks, optimizing a log-likelihood function.

    Parameters
    ----------
    solucoes : np.ndarray
        Array of shape (N, 3) with candidate event positions in geographic 
        or Cartesian coordinates.
    tempos_de_chegada : np.ndarray
        Array of shape (M,) with absolute signal arrival times at the sensors.
    pontos_de_deteccao : np.ndarray
        Array of shape (M, 3) with sensor positions, using the same coordinate system 
        as `solucoes`.
    sistema_cartesiano : bool, optional
        Indicates whether the coordinates are Cartesian (True) or geographic (False). Default is False.
    sigma_t : float, optional
        Temporal standard deviation (used in likelihood computation).
    epsilon_t : float, optional
        Temporal tolerance for spatio-temporal clustering.
    min_pts : int, optional
        Minimum number of points to form a valid cluster (DBSCAN requirement).
    c: np.float64
        wave propagation velocity (default is speed of lught in vacuum)

    Returns
    -------
    tuple
        clusters_espaciais : np.ndarray
            Updated spatial cluster labels.
        verossimilhanca : float
            Total log-likelihood value of the solution.

    Notes
    -----
    - Optimized with Numba for high-performance execution.
    - Compatible with multiple events and multisensor contexts.
    - Suitable for pre-processing before global optimization with genetic algorithms,
      swarm intelligence, and other meta-heuristics.
    """

    # primeiro passo - clusters espaço-temporais

    o_distancias = computa_distancia_batelada(solucoes,
                                              pontos_de_deteccao,
                                              sistema_cartesiano)
    
    tempos_de_origem = tempos_de_chegada - o_distancias/AVG_LIGHT_SPEED

    # segundo passo - se o sistemas de coordenadas for esférico, converter para cartesiano
    # e "transformar" em tempo
    if sistema_cartesiano:
        solucoes_cartesianas = solucoes.copy()/c
    else:
        solucoes_cartesianas = coordenadas_esfericas_para_cartesianas_batelada(
            solucoes) / c

    # antes de aglutinar tudo - vou fazer um empilhamento de tempos
    solucoes_cartesianas = np.hstack((solucoes_cartesianas,
                                     tempos_de_origem.reshape(-1, 1)))

    clustering = DBSCAN(eps=epsilon_t,
                        min_samples=min_pts,
                        metric="euclidean").fit(solucoes_cartesianas)
    clusters_espaciais = clustering.labels_

    centroides_espaciais, _ = calcular_centroides_espaciais(solucoes,
                                                            clusters_espaciais)

    distancias = calcula_distancias_ao_centroide(solucoes,
                                                 clusters_espaciais,
                                                 centroides_espaciais,
                                                 sistema_cartesiano)

    verossimilhanca = funcao_log_verossimilhanca(
        distancias, c * sigma_t)  # \

    # tudo pronto, retornando
    return (clusters_espaciais,
            verossimilhanca)


def stela_phase_two(solucao: np.ndarray,
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

    o_distancias = computa_distancias(solucao,
                                      pontos_de_deteccao,
                                      sistema_cartesiano)

    tempos_de_origem = tempos_de_chegada - o_distancias/AVG_LIGHT_SPEED

    # média dos tempos
    tempo_de_origem= np.mean(tempos_de_origem)

    # distancias a partir do ponto de origem
    distancias = c * (tempos_de_origem - tempo_de_origem)
    
    # verossimilhança
    verossimilhanca = funcao_log_verossimilhanca(distancias, c * sigma_t) 
    + funcao_log_verossimilhanca(tempos_de_origem - tempo_de_origem, sigma_t)

    # tudo pronto, retornando
    return verossimilhanca


if __name__ == "__main__":

    from GeoLightning.Simulator.Simulator import (get_sensors,
                                                  get_random_sensors,
                                                  get_lightning_limits,
                                                  generate_detections,
                                                  generate_events)
    from time import perf_counter

    num_events = [1, 2, 5, 10, 15, 20, 25,
                  30, 100, 500, 800, 1000,
                  5000, 10000, 20000]

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
        start_st = perf_counter()

        (clusters_espaciais,
         verossimilhanca) = stela_phase_one(n_event_positions,
                                            detection_times,
                                            detections,
                                            sistema_cartesiano=False)

        end_st = perf_counter()
        print(
            f"Eventos: {num_events[i]}, Tempo gasto: {end_st - start_st} Segundos")
        print(verossimilhanca)
        len_clusterizados = len(
            np.unique(clusters_espaciais[clusters_espaciais >= 0]))
        len_reais = len(event_positions)
        try:
            assert len_clusterizados == len_reais
            assert spatial_clusters == clusters_espaciais
        except:
            print(f"Clusterizados: {len_clusterizados}, Reais: {len_reais}")
