"""
EELT 7019 - Applied Artificial Intelligence
==========================================

Numba-Optimized 1D Pivoting Algorithm for Event Clustering

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Summary
-------
This module implements a one-dimensional, pivot-based clustering routine
optimized with Numba for event separation in time-of-arrival (TOA) data
collected by spatially distributed sensors. The clustering criterion is
physics-informed: it relies on an upper bound on feasible time differences
between two sensors derived from the inter-sensor time-of-travel (ToT).

Given a detection timestamp ``t_i`` and a pivot timestamp ``t_p`` (both in
seconds), a detection is assigned to the cluster represented by that pivot if:

    |t_i - t_p| <= ToT(s_p, s_i) + eps,

where ``ToT(s_p, s_i)`` is obtained from a precomputed symmetric matrix
(`sensor_tt`) and ``eps`` is a nonnegative tolerance accounting for timestamp
jitter and modeling error.

If a detection is not compatible with any currently active pivot, a new cluster
is created and that detection becomes the pivot of the new cluster.

Notes
-----
Input ordering and asynchronous streams
    The clustering logic is order-dependent. To ensure consistent behavior in
    asynchronous streams, the implementation sorts detections by timestamp
    before clustering. When label alignment with the original input order is
    required, the sorting permutation must be retained (or labels must be
    remapped accordingly).

Star (pivot) constraint vs. complete pairwise constraint
    Membership is enforced with respect to pivots only. Each cluster is
    represented by a single pivot detection (typically the first element that
    instantiated the cluster), and a detection is admitted into a cluster if it
    is compatible with at least one pivot under the TDOA bound.

    This constitutes a star-shaped (pivot-centered) feasibility condition and
    does not, in general, imply full pairwise feasibility among all detections
    within the same cluster. Two non-pivot members may, in principle, violate
    the same bound with each other even if each individually satisfies the
    bound relative to the pivot. If strict pairwise feasibility is required, an
    additional intra-cluster validation step should be applied.

Pivot expiration (safe pivot discard)
    After sorting by timestamp, pivots can be safely discarded once they become
    permanently incompatible with all future detections. For a pivot whose
    sensor index is ``s_p``, define:

        ToT_max(s_p) = max_s sensor_tt[s_p, s].

    A pivot at time ``t_p`` is expired when:

        t_current - t_p > ToT_max(s_p) + eps.

    Expired pivots are excluded from subsequent affinity checks, reducing the
    number of pivot comparisons in long streams.

Computational complexity
    Let N be the number of detections and K be the total number of clusters.
    The runtime is approximately O(N * K_active), where K_active is the number
    of non-expired pivots at each step. Without pivot expiration, the worst
    case may approach O(N^2). With expiration under sorted timestamps, the
    active pivot set is typically bounded by a sliding time window.

Implementation details
    Pivot metadata is stored in preallocated arrays sized for the worst case
    (one pivot per detection), avoiding repeated dynamic reallocations during
    execution.

Intended usage
    This routine is well suited as a fast, physics-informed temporal
    segmentation stage prior to event localization and likelihood-based
    refinement.

Dependencies
------------
- numpy
- numba
- GeoLightning.Utils.Constants
- GeoLightning.Stela.Common
"""

from numba import jit
from numba.typed import List as nList
from typing import Tuple
import numpy as np
from GeoLightning.Utils.Constants import EPSILON_T


@jit(nopython=True, cache=True, fastmath=True)
def order_events(tempos: np.ndarray,
                 indices_sensores: np.ndarray) -> Tuple[np.ndarray,
                                                        np.ndarray]:
    """
    Sort detections by timestamp and reorder associated sensor indices accordingly.

    This function orders detection timestamps in ascending order and applies the
    same permutation to the corresponding sensor-index array, preserving the
    one-to-one association between each timestamp and its originating sensor.
    It is typically used as a preprocessing step in spatio-temporal clustering
    pipelines (including attraction-based clustering), where chronological ordering
    simplifies subsequent pairwise consistency evaluations and event formation.

    Parameters
    ----------
    tempos : np.ndarray
        One-dimensional array of detection timestamps (seconds). The array is
        expected to contain one timestamp per detection.

    indices_sensores : np.ndarray
        One-dimensional array of sensor indices aligned with ``tempos``, such that
        ``indices_sensores[i]`` identifies the sensor that produced ``tempos[i]``.

    Returns
    -------
    tempos_ordenados : np.ndarray
        Copy of ``tempos`` sorted in ascending order.

    indices_sensores_ordenados : np.ndarray
        Copy of ``indices_sensores`` permuted by the same ordering applied to
        ``tempos``, preserving the timestamp-to-sensor association.

    Notes
    -----
    This function performs an ``argsort`` over ``tempos`` and returns copies of the
    reordered arrays to ensure that downstream operations may safely mutate the
    outputs without affecting the inputs.

    Examples
    --------
    >>> tempos = np.array([3.0, 1.0, 2.0])
    >>> indices_sensores = np.array([10, 11, 12])
    >>> t_ord, s_ord = order_events(tempos, indices_sensores)
    >>> t_ord
    array([1., 2., 3.])
    >>> s_ord
    array([11, 12, 10])
    """

    ordered_indexes = np.argsort(tempos)
    tempos_ordenados = tempos[ordered_indexes].copy()
    indices_sensores_ordenados = indices_sensores[ordered_indexes].copy()
    return (tempos_ordenados,
            indices_sensores_ordenados)


@jit(nopython=True, cache=True, fastmath=True)
def get_pivot_affinity(pivots: np.ndarray,
                       pivot_start: np.int32,
                       n_pivots: np.int32,
                       tempo_idx: np.int32,
                       tempos: np.ndarray,
                       indices_sensores: np.ndarray,
                       sensor_tt: np.ndarray,
                       eps: np.float64) -> np.int32:
    """
    Return the index of the first active pivot compatible with a detection.

    Parameters
    ----------
    pivots : np.ndarray
        Preallocated integer array of shape (N, 2) storing pivot metadata.
        Each valid row k (0 <= k < n_pivots) stores:
        - pivots[k, 0] : pivot detection index
        - pivots[k, 1] : cluster label (cluster_id)

    pivot_start : np.int32
        Index of the first active pivot in `pivots`. Pivots in [0, pivot_start)
        are considered expired and must not be tested.

    n_pivots : np.int32
        Current number of registered pivots (valid rows in `pivots`).

    tempo_idx : np.int32
        Index of the detection being evaluated.

    tempos : np.ndarray
        Detection timestamps (seconds), aligned with `indices_sensores`.

    indices_sensores : np.ndarray
        Sensor indices aligned with `tempos`.

    sensor_tt : np.ndarray
        Symmetric matrix of inter-sensor time-of-travel values (seconds).

    eps : np.float64
        Nonnegative tolerance (seconds) added to the TDOA feasibility bound.

    Returns
    -------
    np.int32
        Row index in `pivots` of the first compatible pivot among active pivots.
        Returns -1 if no compatible pivot is found.
    """
    t_i = tempos[tempo_idx]
    s_i = indices_sensores[tempo_idx]

    for p in range(pivot_start, n_pivots):
        pivot_det = pivots[p, 0]
        s_p = indices_sensores[pivot_det]
        t_ij = sensor_tt[s_p, s_i]

        if np.abs(t_i - tempos[pivot_det]) <= (t_ij + eps):
            return p

    return -1


@jit(nopython=True, cache=True, fastmath=True)
def pivot_clustering(tempos: np.ndarray,
                     indices_sensores: np.ndarray,
                     sensor_tt: np.ndarray,
                     eps: np.float64 = EPSILON_T) -> Tuple[np.ndarray,
                                                           np.ndarray,
                                                           np.ndarray]:

    """
    Cluster detections using pivot-based gating with safe pivot expiration.

    This routine performs a timestamp sort and then assigns each detection to the
    first active pivot that satisfies a physics-informed feasibility bound derived
    from the maximum admissible time-difference-of-arrival (TDOA) between two
    sensors:

        |t_i - t_p| <= ToT(s_p, s_i) + eps,

    where ``t_p`` is the pivot timestamp, ``s_p`` is the pivot sensor index,
    ``s_i`` is the current detection sensor index, and ``ToT(s_p, s_i)`` is the
    inter-sensor time-of-travel value obtained from the symmetric matrix
    ``sensor_tt``.

    If a detection is not compatible with any active pivot, a new cluster is
    created and that detection becomes the pivot of the new cluster.

    Pivot expiration
    ---------------
    After sorting, pivots can be safely discarded once they become permanently
    incompatible with all future detections. For a pivot produced by sensor
    ``s_p``, define:

        ToT_max(s_p) = max_s sensor_tt[s_p, s].

    A pivot at time ``t_p`` is expired when:

        t_current - t_p > ToT_max(s_p) + eps.

    Expired pivots are excluded from subsequent affinity checks by advancing an
    active-window pointer.

    Parameters
    ----------
    tempos : np.ndarray
        One-dimensional array of detection timestamps (seconds). The input order
        may be arbitrary; the routine internally sorts timestamps prior to
        clustering.

    indices_sensores : np.ndarray
        One-dimensional array of sensor indices aligned with ``tempos``.
        The element ``indices_sensores[i]`` identifies the sensor that produced
        ``tempos[i]``.

    sensor_tt : np.ndarray
        Symmetric matrix of inter-sensor time-of-travel values (seconds), where
        ``sensor_tt[a, b]`` corresponds to the propagation time between sensors
        ``a`` and ``b`` under the adopted propagation speed.

    eps : np.float64, optional
        Nonnegative tolerance (seconds) added to the feasibility bound to account
        for timestamp jitter and modeling error. Default is ``EPSILON_T``.

    Returns
    -------
    tempos_ordenados : np.ndarray
        Copy of ``tempos`` sorted in ascending order.

    indices_sensores_ordenados : np.ndarray
        Copy of ``indices_sensores`` permuted by the same ordering applied to
        ``tempos_ordenados``.

    labels : np.ndarray
        One-dimensional array of integer cluster labels aligned with
        ``tempos_ordenados``. Labels start at 1 and increase sequentially as new
        clusters are created.

    Notes
    -----
    - The clustering criterion enforces compatibility with at least one pivot
    (star-shaped constraint). It does not, in general, guarantee full pairwise
    feasibility among all members of a cluster.
    - Because outputs are aligned with the sorted order, downstream routines that
    require labels in the original input order must retain the sorting
    permutation (or remap labels accordingly).

    See Also
    --------
    order_events
        Sorts detections by timestamp and permutes sensor indices accordingly.
    get_pivot_affinity
        Tests a detection against the current active set of pivots under the TDOA
        feasibility bound.
    """

    (tempos_ordenados,
     indices_sensores_ordenados) = order_events(tempos=tempos,
                                                indices_sensores=indices_sensores)
    
    n = len(tempos_ordenados)
    labels = -np.ones(n, dtype=np.int32)

    n_sensors = sensor_tt.shape[0]
    max_tt = np.empty(n_sensors, dtype=np.float64)
    for s in range(n_sensors):
        m = 0.0
        for j in range(n_sensors):
            v = sensor_tt[s, j]
            if v > m:
                m = v
        max_tt[s] = m

    pivots = np.empty((n, 2), dtype=np.int32)

    cluster_id = np.int32(1)
    n_pivots = np.int32(1)
    pivot_start = np.int32(0)

    pivots[0, 0] = np.int32(0)         # pivot detection index
    pivots[0, 1] = cluster_id          # cluster label

    for i in range(n):
        t_i = tempos_ordenados[i]

        while pivot_start < n_pivots:
            p_det = pivots[pivot_start, 0]
            s_p = indices_sensores_ordenados[p_det]
            if (t_i - tempos_ordenados[p_det]) > (max_tt[s_p] + eps):
                pivot_start += np.int32(1)
            else:
                break

        p_row = get_pivot_affinity(pivots=pivots,
                                   pivot_start=pivot_start,
                                   n_pivots=n_pivots,
                                   tempo_idx=np.int32(i),
                                   tempos=tempos_ordenados,
                                   indices_sensores=indices_sensores_ordenados,
                                   sensor_tt=sensor_tt,
                                   eps=eps)

        if p_row == -1:
            # New cluster, current detection becomes a new pivot
            cluster_id += np.int32(1)
            pivots[n_pivots, 0] = np.int32(i)
            pivots[n_pivots, 1] = cluster_id
            p_row = n_pivots
            n_pivots += np.int32(1)

        labels[i] = pivots[p_row, 1]

    return (tempos_ordenados,
            indices_sensores_ordenados,
            labels)
