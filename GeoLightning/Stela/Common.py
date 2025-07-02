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
def computa_residuos_temporais(centroides_temporais: np.ndarray,
                               labels: np.ndarray,
                               tempos_de_origem: np.ndarray,
                               eps: np.float64 = 1e-8) -> np.ndarray:
    """
    Computes the temporal residuals between assigned cluster centroids and 
    the original event times.

    This function calculates, for each detection, the deviation between 
    the centroid time of its associated cluster and its actual time of origin. 
    These residuals can be used to evaluate temporal coherence within clusters.

    Parameters
    ----------
    centroides_temporais : np.ndarray
        Array of temporal centroids, where each entry corresponds to the 
        estimated central time of a cluster (in seconds).

    labels : np.ndarray
        Array of integer labels assigning each detection to a specific cluster. 
        It must be the same length as `tempos_de_origem`.

    tempos_de_origem : np.ndarray
        Array of original event times (in seconds), one per detection.

    Returns
    -------
    np.ndarray
        Array of temporal residuals, where each value corresponds to the 
        difference between the assigned centroid time and the detection's 
        original time. Has the same shape as `tempos_de_origem`.
    """
    residuos = np.empty(len(tempos_de_origem))
    labels = labels.astype(np.int32)
    for i in range(len(tempos_de_origem)):
        residuos[i] = centroides_temporais[labels[i]] - tempos_de_origem[i]
    residuos[np.abs(residuos) <= eps] = 0.0
    return residuos