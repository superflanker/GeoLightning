"""
TEmporal Clustering
===================

Numba-Optimized 1D DBSCAN Algorithm for Event Clustering

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Summary
-------
This module implements a one-dimensional version of the DBSCAN (Density-Based 
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
from GeoLightning.Utils.Constants import EPSILON_T, CLUSTER_MIN_PTS


@jit(nopython=True, cache=True, fastmath=True)
def region_query(tempos: np.ndarray,
                 i: np.int32,
                 eps: np.float64) -> np.ndarray:
    """
    Returns the temporal neighbors of a point within an epsilon window.

    This function searches for all indices of points whose temporal values
    lie within a maximum distance `eps` from the point with index `i`.

    Parameters
    ----------
    solucoes : np.ndarray
        Array of solution points in space.
    i : int
        Index of the central point around which neighbors are searched.
    eps : float
        Maximum distance (temporal window) used to define neighborhood.
    Returns
    -------
    np.ndarray
        Array containing the indices of neighboring points.
    """
    vizinhos = []
    for j in range(len(tempos)):
        if np.abs(tempos[j] - tempos[i]) <= eps:
            vizinhos.append(j)
    return np.array(vizinhos)
   

@jit(nopython=True, cache=True, fastmath=True)
def expand_cluster(tempos: np.ndarray,
                   labels: np.ndarray,
                   visitado: np.ndarray,
                   i: np.int32,
                   vizinhos: np.ndarray,
                   cluster_id: np.int32,
                   eps: np.float64,
                   min_pts: np.int32) -> None:
    """
    Expands a cluster from a core point by assigning labels to neighboring points.

    This function performs the cluster expansion step of the DBSCAN algorithm. 
    It iteratively adds new points to the current cluster based on spatial proximity 
    and density requirements. Points are only included if they meet the minimum 
    density criterion.

    Parameters
    ----------
    solucoes : np.ndarray
        1D array of estimated solutions (points in space).
    labels : np.ndarray
        Array containing the cluster labels assigned to each point.
    visitado : np.ndarray
        Boolean array indicating whether each point has already been visited.
    i : int
        Index of the core point from which the cluster expansion starts.
    vizinhos : np.ndarray
        Array of indices of the initial neighboring points.
    cluster_id : int
        Numeric identifier of the current cluster.
    eps : float
        Maximum spatial distance (epsilon neighborhood).
    min_pts : int
        Minimum number of points required to form a valid cluster.

    Returns
    -------
    None
        This function modifies `labels` and `visitado` in-place.
    """

    labels[i] = cluster_id
    k = 0
    while k < len(vizinhos):
        j = vizinhos[k]
        if not visitado[j]:
            visitado[j] = True
            new_vizinhos = region_query(tempos, j, eps)
            if len(new_vizinhos) >= min_pts:
                for nb in new_vizinhos:
                    already_in = False
                    for existing in vizinhos:
                        if nb == existing:
                            already_in = True
                            break
                    if not already_in:
                        np.concatenate((vizinhos, nb * np.ones(1)))
        if labels[j] == -1:
            labels[j] = cluster_id
        k += 1


@jit(nopython=True, cache=True, fastmath=True)
def clusterizacao_temporal_stela(tempos: np.ndarray,
                                 eps: np.float64 = EPSILON_T,
                                 min_pts: np.int32 = CLUSTER_MIN_PTS) -> np.ndarray:
    """
    Main algorithm of the 1D DBSCAN clustering.

    This function implements the DBSCAN algorithm for spatial clustering 
    in three-dimensional space, supporting both Cartesian and geographic coordinates.

    Parameters
    ----------
    solucoes : np.ndarray
        Array of estimated origin solutions (3D points).
    eps : float
        Maximum spatial distance (epsilon) in meters for neighborhood definition.
    min_pts : int
        Minimum number of points required to form a valid cluster.

    Returns
    -------
    labels : np.ndarray
        Array with cluster labels assigned to each point. Noise points are labeled as -1.
    """
    n = len(tempos)
    labels = -1 * np.ones(n, dtype=np.int32)
    cluster_id = 0
    visitado = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        if visitado[i]:
            continue
        visitado[i] = True
        vizinhos = region_query(tempos, i, eps)
        if len(vizinhos) < min_pts:
            labels[i] = -1  # ruído
        else:
            expand_cluster(tempos, labels, visitado, i, vizinhos,
                           cluster_id, eps, min_pts)
            cluster_id += 1

    return labels


# Exemplo de uso
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
        n_event_positions = np.load(file_n_event_positions)
        detection_times = np.load(file_detections_times)
        detection_positions = np.load(file_detections)
        spatial_clustering = np.load(file_spatial_clusters)

        # calculando os tempos de origem
        start_st = perf_counter()

        tempos_de_origem = computa_tempos_de_origem(n_event_positions, 
                                           spatial_clustering, 
                                           detection_times, 
                                           detection_positions)
        labels = clusterizacao_temporal_stela(tempos_de_origem)
        
        end_st = perf_counter()

        print(f"Elapsed time: {end_st - start_st:.6f} seconds")

        print(len(np.unique(labels)))

