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
- STELA Algorithm (main routine)
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
- GeoLightning.Stela.TemporalClustering
- GeoLightning.Stela.SpatialClustering
- GeoLightning.Stela.Bounds
"""


import numpy as np
from numba import jit
from GeoLightning.Utils.Constants import SIGMA_D, \
    EPSILON_D, \
    EPSILON_T, \
    LIMIT_D, \
    CLUSTER_MIN_PTS, \
    MAX_DISTANCE
from GeoLightning.Utils.Utils import computa_tempos_de_origem
from GeoLightning.Stela.TemporalClustering import clusterizacao_temporal_stela
from GeoLightning.Stela.SpatialClustering import clusterizacao_espacial_stela
from GeoLightning.Stela.Bounds import gera_limites


@jit(nopython=True, cache=True, fastmath=True)
def stela(solucoes: np.ndarray,
          tempos_de_chegada: np.ndarray,
          pontos_de_deteccao: np.ndarray,
          clusters_espaciais: np.ndarray,
          sistema_cartesiano: bool = False,
          sigma_d: np.float64 = SIGMA_D,
          epsilon_t: np.float64 = EPSILON_T,
          epsilon_d: np.float64 = EPSILON_D,
          limit_d: np.float64 = LIMIT_D,
          max_d: np.float64 = MAX_DISTANCE,
          min_pts: np.int32 = CLUSTER_MIN_PTS) -> tuple:
    """
    Spatio-Temporal Event Likelihood Assignment (STELA) Algorithm.

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
    clusters_espaciais : np.ndarray
        Array of shape (N,) with the spatial cluster identifiers of each candidate.
    sistema_cartesiano : bool, optional
        Indicates whether the coordinates are Cartesian (True) or geographic (False). Default is False.
    sigma_d : float, optional
        Spatial standard deviation (used in likelihood computation).
    epsilon_t : float, optional
        Temporal tolerance for temporal clustering.
    epsilon_d : float, optional
        Spatial tolerance for spatial clustering.
    limit_d : float, optional
        Search radius used to define bounding boxes for meta-heuristic optimization.
    max_d : float, optional
        Maximum allowable distance between events and sensors.
    min_pts : int, optional
        Minimum number of points to form a valid cluster (DBSCAN requirement).

    Returns
    -------
    tuple
        lb : np.ndarray
            Lower bounds for the optimization process.
        ub : np.ndarray
            Upper bounds for the optimization process.
        centroides : np.ndarray
            Average positions of the detected event clusters.
        detectores : np.ndarray
            Indices of sensors associated with the solution.
        clusters_espaciais : np.ndarray
            Updated spatial cluster labels.
        novas_solucoes : np.ndarray
            Refined spatial solutions.
        verossimilhanca : float
            Total log-likelihood value of the solution.
    
    Notes
    -----
    - Optimized with Numba for high-performance execution.
    - Compatible with multiple events and multisensor contexts.
    - Suitable for pre-processing before global optimization with genetic algorithms,
      swarm intelligence, and other meta-heuristics.
    """

    # primeiro passo - clusters temporais
    tempos_de_origem = computa_tempos_de_origem(solucoes,
                                                clusters_espaciais,
                                                tempos_de_chegada,
                                                pontos_de_deteccao,
                                                sistema_cartesiano)

    clusters_temporais = clusterizacao_temporal_stela(tempos_de_origem,
                                                      epsilon_t,
                                                      min_pts)

    verossimilhanca = 0.0
    # segundo passo - clusterização espacial e cálculo da função de fitness
    # adicionado: calculamos o remapeamento espacial aqui também

    (centroides,
     detectores,
     solucoes_unicas,
     clusters_espaciais,
     novas_solucoes,
     loglikelihood) = clusterizacao_espacial_stela(solucoes,
                                                   clusters_temporais,
                                                   tempos_de_origem,
                                                   epsilon_d,
                                                   sigma_d,
                                                   min_pts,
                                                   sistema_cartesiano)

    verossimilhanca += loglikelihood

    # calculando os limites para o algoritmo meta-heurístico

    lb, ub = gera_limites(novas_solucoes,
                          solucoes_unicas,
                          limit_d,
                          max_d,
                          sistema_cartesiano)

    # tudo pronto, retornando
    return (lb,
            ub,
            centroides,
            detectores,
            clusters_espaciais,
            novas_solucoes,
            verossimilhanca)


if __name__ == "__main__":

    num_events = [2, 5, 10, 15, 20, 25,
                  30, 100, 500, 800, 1000]

    from time import perf_counter

    for i in range(len(num_events)):
        print("Events: {:d}".format(num_events[i]))
        file_detections = "../../data/static_constellation_detections_{:06d}.npy".format(
            num_events[i])

        file_detections_times = "../../data/static_constelation_detection_times_{:06d}.npy".format(
            num_events[i])

        file_event_positions = "../../data/static_constelation_event_positions_{:06d}.npy".format(
            num_events[i])

        file_event_times = "../../data/static_constelation_event_times_{:06d}.npy".format(
            num_events[i])

        file_n_event_positions = "../../data/static_constelation_n_event_positions_{:06d}.npy".format(
            num_events[i])

        file_n_event_times = "../../data/static_constelation_n_event_times_{:06d}.npy".format(
            num_events[i])

        file_distances = "../../data/static_constelation_distances_{:06d}.npy".format(
            num_events[i])

        file_spatial_clusters = "../../data/static_constelation_spatial_clusters_{:06d}.npy".format(
            num_events[i])

        event_positions = np.load(file_event_positions)
        event_times = np.load(file_event_times)
        pontos_de_deteccao = np.load(file_detections)
        tempos_de_chegada = np.load(file_detections_times)
        solucoes = np.load(file_n_event_positions)
        # spatial_clusters = np.load(file_spatial_clusters)
        spatial_clusters = np.cumsum(
            np.ones(len(solucoes), dtype=np.int32)) - 1
        start_st = perf_counter()

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
        end_st = perf_counter()

        print(f"Elapsed time: {end_st - start_st:.6f} seconds")

        print(verossimilhanca)
        print(clusters_espaciais)
        print(lb)
        print(ub)

        # a repetição deve ser com um conjunto bem menor de soluções

        start_st = perf_counter()

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

        end_st = perf_counter()

        print(f"Elapsed time: {end_st - start_st:.6f} seconds")

        print(verossimilhanca)
        print(clusters_espaciais)
        print(lb)
        print(ub)
