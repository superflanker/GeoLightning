"""
ST-DBSCAN
=========

Numba-Optimized Spatio-Temporal DBSCAN Algorithm for Event Clustering

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Summary
-------
This module implements a spatio-temporal version of the DBSCAN (Density-Based 
Spatial Clustering of Applications with Noise) algorithm optimized using Numba. 
It is tailored for the geolocation of atmospheric events using spatial data from 
lightning detection sensors.

Notes
-----
This code is part of the academic activities of the course 
EELT 7019 - Applied Artificial Intelligence at the Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numba
- GeoLightning.Utils.Constants
- GeoLightning.Utils.Utils

"""

from numba import jit
import numpy as np
from GeoLightning.Utils.Utils import computa_distancia
from GeoLightning.Utils.Constants import SIGMA_T, \
    SIGMA_D, \
    EPSILON_T, \
    EPSILON_D, \
    CLUSTER_MIN_PTS
from GeoLightning.Stela.LogLikelihood import funcao_log_verossimilhanca
from GeoLightning.Stela.Entropy import calcular_entropia_local
from GeoLightning.Stela.Dimensions import remapeia_solucoes


@jit(nopython=True, cache=True, fastmath=True)
def calcular_media_clusters(tempos: np.ndarray,
                            labels: np.ndarray) -> tuple:
    """
    Computes the average origin time and the number of detectors for each cluster.

    This function calculates the temporal centroid (mean estimated origin time)
    and the count of sensor detections associated with each spatial cluster.

    Parameters
    ----------
    tempos : np.ndarray
        1D array containing the estimated origin times for all detections.
    labels : np.ndarray
        1D array containing the cluster labels assigned to each detection.

    Returns
    -------
    tuple of np.ndarray
        medias : np.ndarray
            Array of temporal centroids (mean origin times) for each cluster.
        detectores : np.ndarray
            Array with the number of detectors associated with each cluster.
    """
    n_clusters = np.max(labels) + 1
    medias = np.zeros(n_clusters, dtype=np.float64)
    detectores = np.zeros(n_clusters, dtype=np.int32)

    for i in range(len(tempos)):
        lbl = labels[i]
        if lbl >= 0:
            medias[lbl] += tempos[i]
            detectores[lbl] += 1

    for k in range(n_clusters):
        if detectores[k] > 0:
            medias[k] /= detectores[k]
    return medias, detectores


@jit(nopython=True, cache=True, fastmath=True)
def calcular_centroides(solucoes: np.ndarray,
                        mapeamento_tempos_para_solucoes: np.ndarray,
                        labels: np.ndarray) -> np.ndarray:
    """
    Computes the mean location of occurrence for each cluster and the number of sensors associated.

    This function aggregates the origin times of events based on cluster labels and 
    returns the temporal centroid (mean) for each cluster.

    Parameters
    ----------
    solucoes : np.ndarray
        Array of estimated solutions.
    mapeamento_tempos_para_solucoes: np.ndarray
        time to solutions mapping
    labels : np.ndarray
        Array of cluster labels assigned to each solution point.

    Returns
    -------
    medias : np.ndarray
        Array of temporal centroids (mean origin times) for each cluster.
    """

    n_clusters = np.int32(np.max(labels) + 1)
    medias = np.zeros((n_clusters, solucoes.shape[1]))
    detectores = np.zeros(n_clusters, dtype=np.int32)

    for i in range(mapeamento_tempos_para_solucoes.shape[0]):
        lbl = labels[i]
        if lbl >= 0:
            for j in range(solucoes.shape[1]):
                medias[lbl, j] += solucoes[mapeamento_tempos_para_solucoes[i], j]
            detectores[lbl] += 1

    for k in range(n_clusters):
        if detectores[k] > 0:
            for j in range(solucoes.shape[1]):
                medias[k, j] /= detectores[k]
                if np.abs(medias[k, j]) < 1e-12:
                    medias[k, j] = 0.0

    return medias, detectores



@jit(nopython=True, cache=True, fastmath=True)
def calcula_distancias_ao_centroide(solucoes: np.ndarray,
                                    mapeamento_tempos_para_solucoes: np.ndarray,
                                    labels: np.ndarray,
                                    centroides: np.ndarray,
                                    sistema_cartesiano: bool = False) -> np.ndarray:
    """
    Computes the delta D used in likelihood calculations.

    This function calculates the difference in distance (ΔD) between each point in 
    a cluster and its corresponding centroid.

    Parameters
    ----------
    solucoes : np.ndarray
        Array of estimated solutions.
    mapeamento_tempos_para_solucoes: np.ndarray
        time to solutions mapping
    labels : np.ndarray
        Array of cluster labels assigned to each solution point.
    centroides : np.ndarray
        Array of centroids for each cluster.
    sistema_cartesiano : bool
        Indicates whether the coordinate system is Cartesian (True) 
        or geographic (False)

    Returns
    -------
    distancias : np.ndarray
        Array containing the distance differences (ΔD) for each solution point 
        relative to its cluster centroid.
    """
    distancias = np.zeros(len(np.argwhere(labels >= 0)))
    d_idx = 0
    for i in range(labels.shape[0]):
        if labels[i] == -1:
            continue
        distancias[d_idx] = computa_distancia(solucoes[mapeamento_tempos_para_solucoes[i]],
                                              centroides[labels[i]],
                                              sistema_cartesiano)
        d_idx += 1

    return distancias


@jit(nopython=True, cache=True)
def region_query(solucoes: np.ndarray,
                 tempos_de_origem: np.ndarray,
                 mapeamento_tempos_para_solucoes: np.ndarray,
                 idx: np.int64,
                 eps_s: np.float64,
                 eps_t: np.float64,
                 sistema_cartesiano: bool) -> np.ndarray:
    """
    Returns the spatio-temporal neighbors of a point within an epsilon window.

    This function searches for all indices of points whose spatial values
    lie within a maximum distance `eps` from the point with index `i`.

    Parameters
    ----------
    solucoes : np.ndarray
        Array of solution points in space.
    tempos_de_origem: np.ndarray
        Array of origin times
    mapeamento_tempos_para_solucoes: np.ndarray
        time to solutions mapping
    idx : int
        Index of the central point around which neighbors are searched.
    eps_s : float
        Maximum distance (spatial window) used to define neighborhood.
    eps_t : float
        Maximum distance (temporal window) used to define neighborhood.
    sistema_cartesiano : bool
        Indicates whether the coordinate system is Cartesian (True) or geographic (False).

    Returns
    -------
    np.ndarray
        Array containing the indices of neighboring points.
    """
    vizinhos = []
    for j in range(mapeamento_tempos_para_solucoes.shape[0]):
        if idx == j:
            continue
        dist_espacial = computa_distancia(solucoes[mapeamento_tempos_para_solucoes[idx]],
                                          solucoes[mapeamento_tempos_para_solucoes[j]],
                                          sistema_cartesiano)
        dist_temporal = np.abs(tempos_de_origem[idx] - tempos_de_origem[j])
        if dist_espacial <= eps_s and dist_temporal <= eps_t:
            vizinhos.append(j)
    return np.array(vizinhos)


@jit(nopython=True, cache=True)
def expand_cluster(solucoes: np.ndarray,
                   tempos_de_origem: np.ndarray,
                   mapeamento_tempos_para_solucoes: np.ndarray,
                   labels: np.ndarray,
                   visitado: np.ndarray,
                   vizinhos: np.ndarray,
                   ponto_idx: np.int64,
                   cluster_id: np.int64,
                   eps_s: np.float64,
                   eps_t: np.float64,
                   min_pts: np.int32,
                   sistema_cartesiano: bool = False):
    """
    Expands a cluster from a core point by assigning labels to neighboring points.

    This function performs the cluster expansion step of the DBSCAN algorithm. 
    It iteratively adds new points to the current cluster based on spatial proximity 
    and density requirements. Points are only included if they meet the minimum 
    density criterion.

    Parameters
    ----------
    solucoes : np.ndarray
        Array of solution points in space.
    tempos_de_origem: np.ndarray
        Array of origin times
    mapeamento_tempos_para_solucoes: np.ndarray
        time to solutions mapping    
    labels : np.ndarray
        Array containing the cluster labels assigned to each point.
    visitado : np.ndarray
        Boolean array indicating whether each point has already been visited.
    vizinhos : np.ndarray
        Array of indices of the initial neighboring points.
    ponto_idx : int
        Index of the core point from which the cluster expansion starts.
    cluster_id : int
        Numeric identifier of the current cluster.
    eps_s : float
        Maximum distance (spatial window) used to define neighborhood.
    eps_t : float
        Maximum distance (temporal window) used to define neighborhood.
    min_pts : int
        Minimum number of points required to form a valid cluster.
    sistema_cartesiano : bool
        Indicates whether the coordinate system is Cartesian (True) or geographic (False).

    Returns
    -------
    None
        This function modifies `labels` and `visitado` in-place.
    """
    labels[ponto_idx] = cluster_id
    i = 0
    while i < len(vizinhos):
        viz_idx = vizinhos[i]
        if not visitado[viz_idx]:
            visitado[viz_idx] = True
            novos_vizinhos = region_query(solucoes,
                                          tempos_de_origem,
                                          mapeamento_tempos_para_solucoes,
                                          viz_idx,
                                          eps_s,
                                          eps_t,
                                          sistema_cartesiano)
            if len(novos_vizinhos) + 1 >= min_pts:
                for nv in novos_vizinhos:
                    if nv not in vizinhos:
                        np.concatenate((vizinhos, nv * np.ones(1)))
        if labels[viz_idx] == -1:
            labels[viz_idx] = cluster_id
        i += 1


@jit(nopython=True, cache=True)
def st_dbscan(solucoes: np.ndarray,
              tempos_de_origem: np.ndarray,
              mapeamento_tempos_para_solucoes: np.ndarray,
              eps_s: np.float64 = EPSILON_D,
              eps_t: np.float64 = EPSILON_T,
              sigma_d: np.float64 = SIGMA_D,
              min_pts: np.int32 = CLUSTER_MIN_PTS,
              sistema_cartesiano: bool = False) -> tuple:
    """
    Main algorithm of the ST-DBSCAN clustering.

    This function implements the DBSCAN algorithm for spatio-temporal clustering 
    in three-dimensional space, supporting both Cartesian and geographic coordinates.

    Parameters
    ----------
    solucoes : np.ndarray
        Array of solution points in space.
    tempos_de_origem: np.ndarray
        Array of origin times
    mapeamento_tempos_para_solucoes: np.ndarray
        time to solutions mapping   
    eps_s : float
        Maximum distance (spatial window) used to define neighborhood.
    eps_t : float
        Maximum distance (temporal window) used to define neighborhood.
    sigma_d: float
        Spatial standard deviation to compute log-likelihood
    min_pts : int
        Minimum number of points required to form a valid cluster.
    sistema_cartesiano : bool
        Indicates whether the coordinate system is Cartesian (True) or geographic (False).

    Returns
    -------
    tuple
        labels : np.ndarray
            Array with cluster labels assigned to each point. Noise points are labeled as -1.
        distancias : np.ndarray
            Array containing the distance differences (ΔD) for each solution point 
            relative to its cluster centroid.
        centroides : np.ndarray
            Average positions of the detected event clusters.
        detectores : np.ndarray
            Indices of sensors associated with the solution.
        novas_solucoes : np.ndarray
            Refined spatial solutions.
        verossimilhanca : float
            Total log-likelihood value of the solution
    
    """

    N = mapeamento_tempos_para_solucoes.shape[0]
    labels = -np.ones(N, dtype=np.int32)
    visitado = np.zeros(N, dtype=np.bool_)
    cluster_id = 0

    for i in range(N):
        if visitado[i]:
            continue

        visitado[i] = True
        vizinhos = region_query(solucoes,
                                tempos_de_origem,
                                mapeamento_tempos_para_solucoes,
                                i,
                                eps_s,
                                eps_t,
                                sistema_cartesiano)

        if len(vizinhos) + 1 < min_pts:
            labels[i] = -1  # Ruído
        else:

            expand_cluster(solucoes,
                           tempos_de_origem,
                           mapeamento_tempos_para_solucoes,
                           labels,
                           visitado,
                           vizinhos,
                           i,
                           cluster_id,
                           eps_s,
                           eps_t,
                           min_pts,
                           sistema_cartesiano)
            cluster_id += 1

    # cálculo de média temporal e espacial

    centroides, detectores = calcular_centroides(solucoes,
                                                 mapeamento_tempos_para_solucoes,
                                                 labels)

    distancias = calcula_distancias_ao_centroide(solucoes,
                                                 mapeamento_tempos_para_solucoes,
                                                 labels,
                                                 centroides)

    verossimilhanca = calcular_entropia_local(tempos_de_origem[labels >= 0]) \
        + funcao_log_verossimilhanca(distancias, sigma_d)

    novas_solucoes = remapeia_solucoes(solucoes, 
                                       labels, 
                                       centroides)

    return (labels,
            distancias,
            centroides,
            detectores,
            novas_solucoes,
            verossimilhanca)


if __name__ == "__main__":

    from GeoLightning.Utils.Utils import computa_tempos_de_origem
    from time import perf_counter

    num_events = [2, 5, 10, 15, 20, 25,
                  30, 100, 500, 800, 1000]
    
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

        n_event_times = np.load(file_n_event_times)
        solucoes = np.load(file_n_event_positions)
        detection_times = np.load(file_detections_times)
        detection_positions = np.load(file_detections)
        mapeamento_tempos_para_solucoes = np.load(file_spatial_clusters)

        start_st = perf_counter()

        tempos_de_origem = computa_tempos_de_origem(solucoes,
                                                    mapeamento_tempos_para_solucoes,
                                                    detection_times,
                                                    detection_positions)

        (labels,
         distancias,
         centroides,
         detectores,
         novas_solucoes,
         verossimilhanca) = st_dbscan(solucoes,
                                      tempos_de_origem,
                                      mapeamento_tempos_para_solucoes,
                                      EPSILON_D,
                                      EPSILON_T,
                                      SIGMA_D,
                                      CLUSTER_MIN_PTS,
                                      False)

        end_st = perf_counter()

        print(f"Elapsed time: {end_st - start_st:.6f} seconds")

        print(len(np.unique(labels)), verossimilhanca)
