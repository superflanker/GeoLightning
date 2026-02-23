"""
Numba-Friendly Convex Hull Computation
======================================

Convex Hull Utilities for Numba-Compatible Spatial Bounding

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Summary
-------
This module provides Numba-compatible utilities for lexicographic sorting and
convex hull construction in two dimensions. The primary purpose is to support
the derivation of data-driven spatial bounds and polygonal envelopes from sets
of candidate points (e.g., solutions, detections, or intermediate populations)
within meta-heuristic pipelines for atmospheric-event geolocation.

The implementation is designed to remain compatible with Numba's ``nopython``
mode by avoiding Python object allocations and relying on in-place operations
over NumPy arrays. It includes:
- A lexicographic (x, then y) in-place sorting routine,
- A 2D signed cross-product predicate for orientation testing,
- An Andrew monotone-chain convex hull algorithm returning the hull vertices.

Notes
-----
This module is part of the academic activities of the discipline
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numba
"""
import numpy as np
from numba import jit


@jit(nopython=True, cache=True, fastmath=True)
def is_less_than(p1: np.ndarray,
                 p2: np.ndarray) -> bool:
    """
    Compare two 2D points under lexicographic (x, then y) ordering.

    This predicate returns True if and only if ``p1`` precedes ``p2`` in
    lexicographic order, using the first coordinate as the primary key and the
    second coordinate as the secondary key.

    Parameters
    ----------
    p1 : np.ndarray
        One-dimensional array of length 2 representing a 2D point
        ``[x, y]``.

    p2 : np.ndarray
        One-dimensional array of length 2 representing a 2D point
        ``[x, y]``.

    Returns
    -------
    bool
        True if ``p1`` is lexicographically smaller than ``p2``; otherwise False.

    Notes
    -----
    This routine performs a strict ordering check:
    - ``p1 < p2`` if ``p1[0] < p2[0]``; or
    - ``p1 < p2`` if ``p1[0] == p2[0]`` and ``p1[1] < p2[1]``.
    """
    if p1[0] < p2[0]:
        return True
    if p1[0] == p2[0] and p1[1] < p2[1]:
        return True
    return False


@jit(nopython=True, cache=True, fastmath=True)
def partition(arr: np.ndarray,
              low: np.int32,
              high: np.int32) -> np.int32:
    """
    Partition a 2D point array using Hoare's partition scheme under lexicographic order.

    This function partitions the subarray ``arr[low:high+1]`` around a pivot point
    selected as the middle element. Points are compared using lexicographic order
    (x, then y), as implemented by ``is_less_than``.

    The routine is intended to be used as the partition step of an in-place
    QuickSort.

    Parameters
    ----------
    arr : np.ndarray
        Array of shape (N, 2) containing 2D points to be partitioned in place.
        Each row is interpreted as ``[x, y]``.

    low : np.int32
        Lower index (inclusive) of the subarray to partition.

    high : np.int32
        Upper index (inclusive) of the subarray to partition.

    Returns
    -------
    np.int32
        Partition index ``p`` such that the subarray is separated into two parts:
        ``arr[low:p+1]`` and ``arr[p+1:high+1]`` according to Hoare's scheme.

    Notes
    -----
    - This implementation uses Hoare's partition algorithm (not Lomuto).
    - The pivot is copied to avoid aliasing issues during swapping.
    - The function performs swaps in place and does not allocate additional
      arrays beyond small temporary copies of points for swapping.
    """
    mid = (low + high) // 2
    pivot = arr[mid].copy()

    i = low - 1
    j = high + 1

    while True:
        while True:
            i += 1
            if not is_less_than(arr[i], pivot):
                break
        while True:
            j -= 1
            if not is_less_than(pivot, arr[j]):
                break

        if i >= j:
            return j

        tmp = arr[i].copy()
        arr[i] = arr[j]
        arr[j] = tmp


@jit(nopython=True, cache=True, fastmath=True)
def quicksort_lex(arr: np.ndarray,
                  low: np.int32,
                  high: np.int32):
    """
    Sort a 2D point array in place using QuickSort under lexicographic order.

    This routine applies a recursive QuickSort to the subarray
    ``arr[low:high+1]``, using ``partition`` (Hoare partition scheme) and the
    lexicographic comparator ``is_less_than``.

    Parameters
    ----------
    arr : np.ndarray
        Array of shape (N, 2) containing 2D points to be sorted in place.
        Each row is interpreted as ``[x, y]``.

    low : np.int32
        Lower index (inclusive) of the subarray to sort.

    high : np.int32
        Upper index (inclusive) of the subarray to sort.

    Returns
    -------
    None
        The input array is modified in place.

    Notes
    -----
    - This implementation is recursive and may hit recursion limits for very
      large arrays or adversarial inputs.
    - The ordering is lexicographic: primary key is x, secondary key is y.
    """
    if low < high:
        p = partition(arr, low, high)
        quicksort_lex(arr, low, p)
        quicksort_lex(arr, p + 1, high)


@jit(nopython=True, cache=True, fastmath=True)
def lex_sort_points(points: np.ndarray) -> np.ndarray:
    
    """
    Sort 2D points in lexicographic order (x, then y) using in-place QuickSort.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) containing 2D points. Each row is interpreted as
        ``[x, y]``. The array is sorted in place.

    Returns
    -------
    np.ndarray
        Reference to the sorted ``points`` array (sorted in place).

    Notes
    -----
    This function is a thin wrapper around ``quicksort_lex`` and therefore
    inherits its algorithmic characteristics (in-place, recursive QuickSort).
    """
    
    quicksort_lex(points, 0, len(points) - 1)
    return points


@jit(nopython=True, cache=True, fastmath=True)
def cross_product(o: np.ndarray,
                  a: np.ndarray,
                  b: np.ndarray) -> np.float64:
    
    """
    Compute the 2D cross product (signed area) for the turn formed by three points.

    This function evaluates the scalar cross product:

        (a - o) x (b - o)

    in two dimensions, which is proportional to the signed area of the triangle
    (o, a, b). The sign indicates the orientation of the turn:
    positive for counterclockwise, negative for clockwise, and zero for collinear
    points.

    Parameters
    ----------
    o : np.ndarray
        One-dimensional array of length 2 representing the origin point ``[x, y]``.

    a : np.ndarray
        One-dimensional array of length 2 representing the first point ``[x, y]``.

    b : np.ndarray
        One-dimensional array of length 2 representing the second point ``[x, y]``.

    Returns
    -------
    np.float64
        Signed cross product value. Positive indicates a left turn
        (counterclockwise), negative indicates a right turn (clockwise), and
        zero indicates collinearity.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


@jit(nopython=True, cache=True, fastmath=True)
def convex_hull(points: np.ndarray) -> np.ndarray:
    """
    Compute the convex hull of a set of 2D points using the monotone chain algorithm.

    This routine computes the convex hull (as a polygonal chain) for a set of
    planar points using Andrew's monotone chain method. The input points are
    first sorted in lexicographic order (x, then y). The algorithm then builds
    the lower and upper hulls by repeatedly enforcing convexity using the signed
    cross product test.

    The hull is returned as an array of vertices in counterclockwise order,
    without repeating the first vertex at the end.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) containing 2D points. Each row is interpreted as
        ``[x, y]``. The array is sorted in place by this function.

    Returns
    -------
    np.ndarray
        Array of shape (H, 2) containing the hull vertices in counterclockwise
        order, where H is the number of hull vertices. If ``N <= 2``, the input
        points (after sorting) are returned.

    Notes
    -----
    - Collinear points on the boundary are discarded by the use of the
      condition ``cross_product(...) <= 0`` when maintaining convexity.
      Consequently, the returned hull contains only the extreme vertices.
    - The input array is modified in place because ``lex_sort_points`` sorts
      it in place.
    - The complexity is O(N log N) due to sorting, followed by an O(N) hull
      construction step.
    """
    points = lex_sort_points(points)

    n = points.shape[0]
    if n <= 2:
        return points

    lower = np.empty((n, 2), dtype=points.dtype)
    l_idx = 0

    for i in range(n):
        p = points[i]
        while l_idx >= 2 and cross_product(lower[l_idx-2], lower[l_idx-1], p) <= 0:
            l_idx -= 1
        lower[l_idx] = p
        l_idx += 1

    upper = np.empty((n, 2), dtype=points.dtype)
    u_idx = 0
    for i in range(n - 1, -1, -1):
        p = points[i]
        while u_idx >= 2 and cross_product(upper[u_idx-2], upper[u_idx-1], p) <= 0:
            u_idx -= 1
        upper[u_idx] = p
        u_idx += 1

    full_hull = np.empty((l_idx + u_idx - 2, 2), dtype=points.dtype)
    full_hull[:l_idx-1] = lower[:l_idx-1]
    full_hull[l_idx-1:] = upper[:u_idx-1]

    return full_hull
