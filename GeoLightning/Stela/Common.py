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
from GeoLightning.Utils.Constants import SIGMA_T, SIGMA_D


@jit(nopython=True, cache=True, fastmath=True)
def clustering_metric(detection_time_s1: np.float64,
                      detection_time_s2: np.float64,
                      s1_index: np.int32,
                      s2_index: np.int32,
                      sensor_tt: np.ndarray) -> np.float64:
    """
    Computes the Spatio-Temporal Consistency Metric based on the TDOA 
    between two detection times and the physical distance between their 
    respective sensors.

    This metric evaluates how well the observed time difference between 
    two detections matches the expected time difference implied by the 
    distance between the sensors and the signal propagation speed.

    Parameters
    ----------
    detection_time_s1 : np.float64
        Detection timestamp (in seconds) from sensor s1.

    detection_time_s2 : np.float64
        Detection timestamp (in seconds) from sensor s2.

    s1_index : np.int32
        Index of the first sensor in the sensor array.

    s2_index : np.int32
        Index of the second sensor in the sensor array.

    sensor_tt : np.ndarray
        Precomputed symmetric matrix containing the pairwise distances 
        between all sensor positions divided by the signal propagation 
        speed (i.e., time-of-travel between sensors), in seconds.

    Returns
    -------
    np.float64
        The absolute deviation between the observed time difference and 
        the expected time-of-travel between the two sensors. Lower values 
        indicate higher spatio-temporal consistency and, hence, greater 
        likelihood that the detections are from the same physical event.
    """
    return np.abs(np.abs(detection_time_s1 - detection_time_s2) - sensor_tt[s1_index, s2_index])


@jit(nopython=True, cache=True, fastmath=True)
def affinity_weight(delta: np.float64,
                    sigma: np.float64 = SIGMA_T,
                    max_eps: np.float64 = (SIGMA_T ** 2)/2) -> np.float64:
    """
    Compute an affinity weight from a spatio-temporal inconsistency metric.

    This function converts a spatio-temporal discrepancy between two detections
    into a dimensionless affinity (attraction) weight using a Gaussian (RBF) kernel.
    The discrepancy is obtained from ``clustering_metric`` and typically represents
    the absolute mismatch between the observed time-difference-of-arrival (TDOA)
    and the expected inter-sensor time-of-travel (ToT) derived from the sensor pair.

    Formally, if ``m`` denotes the metric returned by ``clustering_metric`` and
    ``sigma`` is a positive scale parameter, this function returns:

        w = exp( - m^2 / (2 * sigma^2) )

    Smaller discrepancies yield weights closer to 1, while larger discrepancies
    yield weights closer to 0. This weight is intended to be used as a local kernel
    within attraction-based clustering procedures.

    Parameters
    ----------
    detection_time_s1 : np.float64
      
    sigma : np.float64, optional
        Positive scale (standard deviation) of the Gaussian kernel, expressed in the same 
        units as delta. Default is ``1e-6``.

    Returns
    -------
    np.float64
        Attraction (affinity) weight in the interval (0, 1], computed from the
        spatio-temporal inconsistency metric. Values closer to 1 indicate stronger
        consistency (higher affinity) between the two detections.

    See Also
    --------
    clustering_metric
        Computes the underlying spatio-temporal inconsistency metric from which the
        attraction weight is derived.
    """
    
    z = (delta ** 2) / (2.0 * sigma ** 2)

    return np.exp(-z)


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
