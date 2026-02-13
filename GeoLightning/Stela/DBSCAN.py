"""
EELT 7019 - Applied Artificial Intelligence
===========================================

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
from GeoLightning.Stela.Common import clustering_metric


@jit(nopython=True, cache=True, fastmath=True)
def region_query(tempos: np.ndarray,
                 indices_sensores: np.ndarray,
                 sensor_tt: np.ndarray,
                 i: np.int32,
                 eps: np.float64) -> np.ndarray:
    """
    Returns the temporal neighbors of a point within an epsilon window.

    This function searches for all indices of points whose temporal values
    lie within a maximum distance `eps` from the point with index `i`.

    Parameters
    ----------
    tempos : np.ndarray
        Array of detection times associated with each detector
    indice_sensores: np.ndarray
        Array of sensor IDs associated with times
    sensor_tt: np.ndarray
        Association matrix with time-to-travel in light speed of the distances
        between sensors
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
        if i == j:
            continue
        if clustering_metric(tempos[i],
                             tempos[j],
                             indices_sensores[i],
                             indices_sensores[j],
                             sensor_tt) <= eps:
            vizinhos.append(j)
    return np.array(vizinhos)


@jit(nopython=True, cache=True, fastmath=True)
def expand_cluster(tempos: np.ndarray,
                   indices_sensores: np.ndarray,
                   sensor_tt: np.ndarray,
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
    tempos : np.ndarray
        Array of detection times associated with each detector
    indice_sensores: np.ndarray
        Array of sensor IDs associated with times
    sensor_tt: np.ndarray
        Association matrix with time-to-travel in light speed of the distances
        between sensors
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
            new_vizinhos = region_query(tempos,
                                        indices_sensores,
                                        sensor_tt,
                                        j,
                                        eps)
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
def dbscan(tempos: np.ndarray,
           indices_sensores: np.ndarray,
           sensor_tt: np.ndarray,
           eps: np.float64 = EPSILON_T,
           min_pts: np.int32 = CLUSTER_MIN_PTS) -> np.ndarray:
    """
    Main algorithm of the 1D DBSCAN clustering.

    This function implements the DBSCAN algorithm for spatial clustering 
    in three-dimensional space, supporting both Cartesian and geographic coordinates.

    Parameters
    ----------
    tempos : np.ndarray
        Array of detection times associated with each detector
    indice_sensores: np.ndarray
        Array of sensor IDs associated with times
    sensor_tt: np.ndarray
        Association matrix with time-to-travel in light speed of the distances
        between sensors
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
        vizinhos = region_query(tempos,
                                indices_sensores,
                                sensor_tt,
                                i,
                                eps)
        if len(vizinhos) < min_pts:
            labels[i] = -1  # ruído
        else:
            expand_cluster(tempos,
                           indices_sensores,
                           sensor_tt,
                           labels,
                           visitado,
                           i,
                           vizinhos,
                           cluster_id,
                           eps,
                           min_pts)
            cluster_id += 1

    return labels
