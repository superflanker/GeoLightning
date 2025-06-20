"""
Spatial DBSCAN Clustering
=========================

Numba-Optimized 3D DBSCAN Algorithm for Event Clustering

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Summary
-------
This module implements a three-dimensional version of the DBSCAN (Density-Based 
Spatial Clustering of Applications with Noise) algorithm optimized using Numba. 
It is tailored for the geolocation of atmospheric events using spatial data from 
lightning detection sensors.

The algorithm supports both Cartesian and geographic coordinate systems and 
is intended for high-performance clustering in applications requiring real-time 
or large-scale event separation.

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
from GeoLightning.Utils.Constants import EPSILON_D, CLUSTER_MIN_PTS
from GeoLightning.Utils.Utils import computa_distancia


@jit(nopython=True, cache=True, fastmath=True)
def region_query_3D(solucoes: np.ndarray,
                    i: np.int32,
                    eps: np.float64,
                    sistema_cartesiano: bool = False) -> np.ndarray:
    """
    Returns the spatial neighbors of a point within an epsilon window.

    This function searches for all indices of points whose spatial values
    lie within a maximum distance `eps` from the point with index `i`.

    Parameters
    ----------
    solucoes : np.ndarray
        Array of solution points in space.
    i : int
        Index of the central point around which neighbors are searched.
    eps : float
        Maximum distance (spatial window) used to define neighborhood.
    sistema_cartesiano : bool
        Indicates whether the coordinate system is Cartesian (True) or geographic (False).

    Returns
    -------
    np.ndarray
        Array containing the indices of neighboring points.
    """
    vizinhos = []
    for j in range(len(solucoes)):
        if computa_distancia(solucoes[i],
                             solucoes[j],
                             sistema_cartesiano) <= eps:
            vizinhos.append(j)
    return np.array(vizinhos)


@jit(nopython=True, cache=True, fastmath=True)
def expand_cluster_3D(solucoes: np.ndarray,
                      labels: np.ndarray,
                      visitado: np.ndarray,
                      i: np.int32,
                      vizinhos: np.ndarray,
                      cluster_id: np.int32,
                      eps: np.float64,
                      min_pts: np.int32,
                      sistema_cartesiano: bool = False) -> None:
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
            new_vizinhos = region_query_3D(
                solucoes, j, eps, sistema_cartesiano)
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
def clusterizacao_DBSCAN3D(solucoes: np.ndarray,
                           eps: np.float64 = EPSILON_D,
                           min_pts: np.int32 = CLUSTER_MIN_PTS,
                           sistema_cartesiano: bool = False) -> np.ndarray:
    """
    Main algorithm of the 3D DBSCAN clustering.

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
    sistema_cartesiano : bool
        Indicates whether the coordinate system is Cartesian (True) 
        or geographic (False).

    Returns
    -------
    labels : np.ndarray
        Array with cluster labels assigned to each point. Noise points are labeled as -1.
    """

    n = len(solucoes)
    labels = -1 * np.ones(n, dtype=np.int32)
    cluster_id = 0
    visitado = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        if visitado[i]:
            continue
        visitado[i] = True
        vizinhos = region_query_3D(solucoes,
                                   i,
                                   eps,
                                   sistema_cartesiano)
        if len(vizinhos) < min_pts:
            labels[i] = -1  # ruído
        else:
            expand_cluster_3D(solucoes,
                              labels,
                              visitado,
                              i,
                              vizinhos,
                              cluster_id,
                              eps,
                              min_pts,
                              sistema_cartesiano)
            cluster_id += 1

    return labels


if __name__ == "__main__":
    cluster1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float64)
    cluster2 = np.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]], dtype=np.float64)
    cluster3 = np.array([[6, 6, 6], [6, 6, 6], [6, 6, 6]], dtype=np.float64)
    solucoes = np.vstack((cluster1, cluster2, cluster3))
    labels = clusterizacao_DBSCAN3D(solucoes,
                                    eps=1.0,
                                    min_pts=3,
                                    sistema_cartesiano=True)
    print(labels)
