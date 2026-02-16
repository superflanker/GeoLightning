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
from typing import Tuple
from GeoLightning.Utils.Utils import computa_distancias
from GeoLightning.Utils.Constants import SIGMA_T, SIGMA_D, AVG_LIGHT_SPEED


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
def tempo_do_evento_ponderado(tempos: np.ndarray,
                              distancias: np.ndarray,
                              pesos: np.ndarray) -> np.float64:
    """
    Estimate the event origin time using a weighted TOA back-projection.

    This routine computes a weighted estimate of the emission (origin) time
    ``t0`` from time-of-arrival (TOA) measurements. For each detection ``i``,
    an individual origin-time estimate is formed as:

        t0_i = tempos[i] - distancias[i] / c,

    where ``c`` is the assumed propagation speed (``AVG_LIGHT_SPEED``). The
    returned estimate is the weighted mean:

        t0 = sum_i pesos[i] * t0_i / sum_i pesos[i].

    If the total weight is nonpositive (e.g., all weights are zero), the
    function falls back to the unweighted mean of ``t0_i``.

    Parameters
    ----------
    tempos : np.ndarray
        One-dimensional array of TOA timestamps (seconds).

    distancias : np.ndarray
        One-dimensional array of propagation distances (meters) aligned with
        ``tempos``. Each entry corresponds to the distance between the current
        event hypothesis and the sensor that produced the associated TOA.

    pesos : np.ndarray
        One-dimensional array of nonnegative weights aligned with ``tempos``.
        These weights typically originate from a robust weighting rule (e.g.,
        IRLS) or a kernel-based affinity function.

    Returns
    -------
    np.float64
        Weighted estimate of the event origin time ``t0`` (seconds). If the
        sum of weights is nonpositive, an unweighted estimate is returned.

    Notes
    -----
    - This routine assumes consistent units: seconds for ``tempos``, meters for
      ``distancias``, and meters/second for ``AVG_LIGHT_SPEED``.
    - Callers should ensure that ``tempos``, ``distancias``, and ``pesos`` have
      identical lengths.
    """

    num = 0.0
    den = 0.0
    for i in range(len(tempos)):
        num += pesos[i] * (tempos[i] - distancias[i] / AVG_LIGHT_SPEED)
        den += pesos[i]
    if den <= 0.0:
        # solução muito longe do desejado - pesos iguais
        num = 0.0
        for i in range(len(tempos)):
            num += (tempos[i] - distancias[i] / AVG_LIGHT_SPEED)
        return num / len(tempos)
    return num / den


@jit(nopython=True, cache=True, fastmath=True)
def residuos_espaciais(solucao_inicial: np.ndarray,
                       tempos_de_chegada: np.ndarray,
                       pontos_de_chegada: np.ndarray,
                       pesos: np.ndarray,
                       sistema_cartesiano: bool = False) -> Tuple[np.ndarray,
                                                                  np.float64]:
    """
    Compute spatial residuals and a weighted origin-time estimate for a TOA model.

    For a given candidate event location, this function computes the propagation
    distances from the candidate to each sensor location and estimates the event
    origin time ``t0`` via a weighted back-projection. It then returns spatial
    residuals (meters) defined as:

        r_i = c * (t_i - t0) - d_i,

    where ``t_i`` are the measured arrival times, ``d_i`` are the predicted
    propagation distances, and ``c`` is the assumed propagation speed
    (``AVG_LIGHT_SPEED``).

    Parameters
    ----------
    solucao_inicial : np.ndarray
        Candidate event location. For spherical mode, it must follow the
        geodetic format ``[latitude, longitude, altitude]`` (degrees, degrees,
        meters). For Cartesian mode, it must be ``[x, y, z]`` in meters.

    tempos_de_chegada : np.ndarray
        One-dimensional array of absolute arrival times (seconds), one per
        detection.

    pontos_de_chegada : np.ndarray
        Array of sensor coordinates aligned with ``tempos_de_chegada``. Each row
        must follow the same coordinate convention implied by
        ``sistema_cartesiano`` (geodetic degrees/meters or Cartesian meters).

    pesos : np.ndarray
        One-dimensional array of weights aligned with ``tempos_de_chegada`` used
        to compute the weighted origin time estimate ``t0``.

    sistema_cartesiano : bool, optional
        If True, Cartesian distances are used; otherwise spherical distances are
        used. Default is False.

    Returns
    -------
    residuos : np.ndarray
        One-dimensional array of spatial residuals (meters), aligned with
        ``tempos_de_chegada``.

    t0 : np.float64
        Weighted estimate of the event origin time (seconds).

    Notes
    -----
    - The spatial residual formulation is convenient when the likelihood is
      expressed in distance units (e.g., by converting temporal errors via
      multiplication by ``AVG_LIGHT_SPEED``).
    - The function assumes that the coordinate system chosen is consistent
      between ``solucao_inicial`` and ``pontos_de_chegada``.
    """
    d = computa_distancias(origem=solucao_inicial,
                           destinos=pontos_de_chegada,
                           sistema_cartesiano=sistema_cartesiano)

    t0 = tempo_do_evento_ponderado(tempos=tempos_de_chegada,
                                   distancias=d,
                                   pesos=pesos)

    n = len(tempos_de_chegada)
    r = np.empty(n, dtype=np.float64)
    for i in range(n):
        r[i] = AVG_LIGHT_SPEED * (tempos_de_chegada[i] - t0) - d[i]
    return r, t0


@jit(nopython=True, cache=True, fastmath=True)
def computa_pesos(residuos: np.ndarray,
                  sigma: np.float64,
                  k: np.float64,
                  pesos_alg: str = "rbf") -> np.ndarray:
    """
    Compute per-observation weights from residuals using a selected weighting rule.

    This function maps a vector of residuals to a vector of nonnegative weights,
    typically for use in robust refinement procedures such as Iteratively
    Reweighted Least Squares (IRLS). The weighting rule is selected by the
    ``pesos_alg`` identifier:

    - ``"rbf"``  : Gaussian (RBF) kernel weights.
    - ``"huber"``: Huber IRLS weights.
    - ``"tukey"``: Tukey biweight (bisquare) IRLS weights.

    If an unknown identifier is provided, the routine defaults to the RBF rule.

    Parameters
    ----------
    residuos : np.ndarray
        One-dimensional array of residuals. Residuals may be expressed in time
        (seconds) or distance (meters), provided ``sigma`` is given in the same
        units.

    sigma : np.float64
        Positive scale parameter used by the weighting rule (same units as
        ``residuos``). For TOA geolocation in spatial units, a common choice is
        ``sigma = AVG_LIGHT_SPEED * sigma_t``.

    k : np.float64
        Positive, dimensionless tuning constant for robust weighting rules
        (Huber/Tukey), expressed in units of the normalized residual magnitude.

    pesos_alg : str, optional
        Identifier of the weighting rule. Supported values are ``"rbf"``,
        ``"huber"``, and ``"tukey"``. Default is ``"rbf"``.

    Returns
    -------
    np.ndarray
        One-dimensional array of weights aligned with ``residuos``. The exact
        range depends on the rule (e.g., (0, 1] for Huber, [0, 1] for Tukey).

    Notes
    -----
    - This function assumes that the called weighting routines are available in
      the current module namespace.
    - Callers must ensure ``sigma > 0`` and, for robust rules, ``k > 0``.
    """
    match pesos_alg:
        case "rbf":
            return afinidde_rbf(delta=residuos,
                                sigma=sigma)
        case "huber":
            return afinidade_huber(delta=residuos,
                                   sigma=sigma,
                                   k=k)
        case "tukey":
            return afinidade_tukey(delta=residuos,
                                   sigma=sigma,
                                   k=k)
        case _:
            return afinidde_rbf(delta=residuos,
                                sigma=sigma)


@jit(nopython=True, cache=True, fastmath=True)
def solver_2x2_simples(a11: np.float64,
                       a12: np.float64,
                       a22: np.float64,
                       b1: np.float64,
                       b2: np.float64) -> Tuple[np.float64,
                                                np.float64]:
    """
    Solve a symmetric 2x2 linear system in closed form.

    This routine solves the linear system:

        [a11  a12] [x1] = [b1]
        [a12  a22] [x2]   [b2]

    using an explicit determinant-based formula. If the determinant magnitude
    is below a small threshold, the system is treated as (near) singular and
    the function returns zeros.

    Parameters
    ----------
    a11 : np.float64
        (1, 1) entry of the symmetric coefficient matrix.

    a12 : np.float64
        (1, 2) and (2, 1) entry of the symmetric coefficient matrix.

    a22 : np.float64
        (2, 2) entry of the symmetric coefficient matrix.

    b1 : np.float64
        First entry of the right-hand side vector.

    b2 : np.float64
        Second entry of the right-hand side vector.

    Returns
    -------
    x1 : np.float64
        First component of the solution vector.

    x2 : np.float64
        Second component of the solution vector.

    Notes
    -----
    - The determinant is ``det = a11 * a22 - a12^2``.
    - When ``|det|`` is very small, returning zeros avoids numerical blow-up,
      but callers should interpret this as a degeneracy condition.
    """
    det = a11 * a22 - a12 * a12
    if np.abs(det) < 1e-18:
        return 0.0, 0.0
    x1 = (b1 * a22 - b2 * a12) / det
    x2 = (-b1 * a12 + b2 * a11) / det
    return x1, x2


@jit(nopython=True, cache=True, fastmath=True)
def afinidde_rbf(delta: np.float64,
                 sigma: np.float64 = SIGMA_T) -> np.float64:
    """
    Compute weights using the Radial-Based Gaussian function.

    This function converts a spatio-temporal discrepancy between two values
    into a dimensionless affinity weight using a Gaussian (RBF) kernel.

    Formally, if ``m`` denotes the deviation` and
    ``sigma`` is a positive scale parameter, this function returns:

        w = exp( - m^2 / (2 * sigma^2) )

    Smaller discrepancies yield weights closer to 1, while larger discrepancies
    yield weights closer to 0. 

    Parameters
    ----------
    delta : np.float64
        differences or deviations from a central value

    sigma : np.float64, optional
        Positive scale (standard deviation) of the Gaussian kernel, expressed in the same 
        units as delta. Default is ``1e-6``.

    Returns
    -------
    np.float64
        Affinity weight in the interval (0, 1], computed from the
        deviation metric. Values closer to 1 indicate stronger
        consistency (higher affinity) between the two differences.

    Notes
    -----
    - This routine does not guard against ``sigma <= 0``. Callers must ensure
      ``sigma`` is strictly positive to avoid division by zero.
    - The returned weights correspond to the standard IRLS form
      ``w(u) = psi(u) / u`` associated with the RBF.

    See Also
    --------
    afinidade_tukey
        Computes weights using Tukey's biweight influence function.
    afinidade_huber
        Computes weights using the Huber influence function.
    """

    z = (delta ** 2) / (2.0 * sigma ** 2)

    return np.exp(-z)


@jit(nopython=True, cache=True, fastmath=True)
def afinidade_huber(delta: np.ndarray,
                    sigma: np.float64,
                    k: np.float64) -> np.ndarray:
    """
    Compute weights using the Huber influence function.

    This function maps residuals to nonnegative weights. Let ``u_i = |delta_i| / sigma``
    denote the normalized residual magnitude for observation ``i``. The Huber
    weight is defined as:

        w_i = 1,           if u_i <= k
        w_i = k / u_i,     if u_i >  k

    Hence, small residuals are treated with full weight (quadratic regime),
    while large residuals are downweighted (approximately linear regime).

    Parameters
    ----------
    delta : np.ndarray
        One-dimensional array of residuals. The residuals may be expressed in
        time (seconds) or distance (meters), provided that ``sigma`` is given in
        the same units.

    sigma : np.float64
        Positive scale parameter used to normalize residuals (same units as
        ``delta``). In TOA-based geolocation, a common choice is
        ``sigma = c * sigma_t`` when ``delta`` is expressed in meters.

    k : np.float64
        Positive, dimensionless Huber threshold expressed in units of the
        normalized residual. Values of ``u_i`` above ``k`` are downweighted.

    Returns
    -------
    np.ndarray
        One-dimensional array of weights with the same length as ``delta``.
        Each weight lies in the interval (0, 1].

    Notes
    -----
    - This routine does not guard against ``sigma <= 0``. Callers must ensure
      ``sigma`` is strictly positive to avoid division by zero.
    - The returned weights correspond to the standard IRLS form
      ``w(u) = psi(u) / u`` associated with the Huber loss.

    See Also
    --------
    afinidade_tukey
        Computes weights using Tukey's biweight influence function.
    """

    n = len(delta)
    w = np.empty(n, dtype=np.float64)
    inv_sigma = 1.0 / sigma  # assumes sigma > 0
    for i in range(n):
        u = np.abs(delta[i]) * inv_sigma
        if u <= k:
            w[i] = 1.0
        else:
            w[i] = k / u
    return w


@jit(nopython=True, cache=True, fastmath=True)
def afinidade_tukey(delta: np.ndarray,
                    sigma: np.float64,
                    k: np.float64) -> np.ndarray:
    """
    Compute weights using Tukey's biweight (bisquare) influence function.

    This function maps residuals to nonnegative weights suitable for
    Iteratively Reweighted Least Squares (IRLS). Let ``u_i = |delta_i| / sigma``
    denote the normalized residual magnitude for observation ``i``. Tukey's
    biweight assigns:

        w_i = (1 - (u_i / k)^2)^2,   if u_i <= k
        w_i = 0,                    if u_i >  k

    Thus, residuals beyond the cutoff ``k`` are rejected (zero weight), while
    inliers are smoothly downweighted as they approach the cutoff.

    Parameters
    ----------
    delta : np.ndarray
        One-dimensional array of residuals. The residuals may be expressed in
        time (seconds) or distance (meters), provided that ``sigma`` is given in
        the same units.

    sigma : np.float64
        Positive scale parameter used to normalize residuals (same units as
        ``delta``). In TOA-based geolocation, a common choice is
        ``sigma = c * sigma_t`` when ``delta`` is expressed in meters.

    k : np.float64
        Positive, dimensionless cutoff expressed in units of the normalized
        residual. Observations with ``u_i > k`` receive zero weight.

    Returns
    -------
    np.ndarray
        One-dimensional array of IRLS weights with the same length as ``delta``.
        Each weight lies in the interval [0, 1].

    Notes
    -----
    - This routine does not guard against ``sigma <= 0`` or ``k <= 0``. Callers
      must ensure both parameters are strictly positive.
    - The returned weights correspond to the standard IRLS form
      ``w(u) = psi(u) / u`` associated with Tukey's biweight loss.

    See Also
    --------
    afinidade_huber
        Computes IRLS weights using the Huber influence function.

    """
    n = len(delta)
    w = np.empty(n, dtype=np.float64)
    inv_sigma = 1.0 / sigma  # assumes sigma > 0
    inv_k = 1.0 / k          # assumes k > 0
    for i in range(n):
        u = np.abs(delta[i]) * inv_sigma
        if u <= k:
            t = u * inv_k
            one_minus = 1.0 - t * t
            w[i] = one_minus * one_minus
        else:
            w[i] = 0.0
    return w


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
