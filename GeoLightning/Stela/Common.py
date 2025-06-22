"""
STELA Common Utilities
======================

Common Functions of STELA Algorithms

Author
-------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Summary
-------
This module provides common utilities for STELA-based
clustering, centroid estimation, and spatial likelihood evaluation.

Notes
-----
This module is part of the coursework for EELT 7019 - Applied Artificial Intelligence,
Federal University of Paraná (UFPR). 

Dependencies
------------
- numpy
- numba
- GeoLightning.Utils.Utils
"""
import numpy as np
from numba import jit
from GeoLightning.Utils.Utils import computa_distancia


@jit(nopython=True, cache=True, fastmath=True)
def calcula_centroides_temporais(tempos: np.ndarray,
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
def calcula_residuos_temporais(tempos: np.ndarray,
                               labels: np.ndarray,
                               centroides: np.ndarray) -> np.ndarray:
    """
    Computes the temporal resuduals around the mean solution time

    Parameters
    ----------
    tempos: np.ndarray
        vetor of origin times ([t1, t2, t3, ...])
    labels: np.ndarray 
        clusterings labels ([la, la, lb, ...])
    centroides: np.ndarray
        clusterings centroids ([ta, tb, tc, ...])

    Returns
    -------
    residuos_de_tempo : np.ndarray
        Array containing mean times for each solution point 
        relative to its cluster centroid. 

    """
    residuos_de_tempo = np.zeros(len(np.argwhere(labels >= 0)), dtype=np.float64)
    d_idx = 0
    for i in range(labels.shape[0]):
        if labels[i] == -1:
            continue
        residuos_de_tempo[d_idx] = np.abs(tempos[i] - centroides[labels[i]])
        d_idx += 1

    return residuos_de_tempo


@jit(nopython=True, cache=True, fastmath=True)
def calcular_centroides_espaciais(solucoes: np.ndarray,
                                  labels: np.ndarray) -> np.ndarray:
    """
    Computes the mean location of occurrence for each cluster and the number of sensors associated.

    This function aggregates the origin times of events based on cluster labels and 
    returns the temporal centroid (mean) for each cluster.

    Parameters
    ----------
    solucoes : np.ndarray
        Array of estimated solutions.
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

    for i in range(labels.shape[0]):
        lbl = labels[i]
        if lbl >= 0:
            for j in range(solucoes.shape[1]):
                medias[lbl, j] += solucoes[lbl, j]
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
    labels : np.ndarray
        Array of cluster labels assigned to each solution point.
    centroides : np.ndarray
        Array of centroids for each cluster.    s
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
        distancias[d_idx] = computa_distancia(solucoes[labels[i]],
                                              centroides[labels[i]],
                                              sistema_cartesiano)
        d_idx += 1

    return distancias
