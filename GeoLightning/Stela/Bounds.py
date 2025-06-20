"""
Bound Limits Generation
=======================

Bound Generator for Meta-Heuristic Search Spaces

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Summary
-------
This module defines and computes dynamic spatial bounds for meta-heuristic algorithms
used in the geolocation of atmospheric events. It provides an optimized routine for 
generating localized search limits around clustered points, based on spatial and 
cartesian coordinate systems.

The bounding regions are adaptive: tight around unique solutions and wider for 
ambiguous (non-unique) detections. The formulation respects physical constraints 
and enables integration with global optimization strategies.

Notes
-----
This module is part of the activities of the discipline 
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numba
- GeoLightning.Utils.Constants

"""
from numba import jit
import numpy as np
from GeoLightning.Utils.Constants import EPSILON_D, \
    MAX_DISTANCE, \
    R_LAT


@jit(nopython=True, cache=True, fastmath=True)
def gera_limites(pontos_clusterizados: np.ndarray,
                 clusters: np.ndarray,
                 raio_metros: np.float64 = EPSILON_D,
                 raio_maximo: np.float64 = MAX_DISTANCE,
                 sistema_cartesiano: bool = False) -> tuple:
    """
    Optimized version using Numba for generating local search bounds
    around clustered points.

    Parameters
    ----------
    pontos_clusterizados : np.ndarray
        A (N, 3) matrix with columns [lat, lon, alt], representing clustered points 
        in degrees (lat/lon) and meters (altitude).
    clusters : np.ndarray
        Array of cluster identifiers for each point in the reduced solution.
    raio_metros : float
        Search radius in meters around each uniquely identified solution point.
    raio_maximo : float
        Maximum radius in meters around each non-unique solution point.
    sistema_cartesiano : bool
        Indicates whether the coordinate system is Cartesian (True) or geographic (False).

    Returns
    -------
    tuple of np.ndarray
        lb : np.ndarray
            Lower bounds of the search region for each cluster.
        ub : np.ndarray
            Upper bounds of the search region for each cluster.
    """

    n = pontos_clusterizados.shape[0]

    ub = np.zeros((n, 3), dtype=np.float64)

    lb = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        lat = pontos_clusterizados[i, 0]
        lon = pontos_clusterizados[i, 1]
        alt = pontos_clusterizados[i, 2]

        d_raio = raio_metros
        if clusters[i] == -1:
            # se o ponto for (-1,-1,-1) a solução já é descartada
            if lat != -1 and lon != -1 and alt != -1:
                d_raio = raio_maximo
            else:
                lat = 0
                lon = 0
                alt = 0
                d_raio = 0

        if sistema_cartesiano:
            dlat = d_raio
            dlon = d_raio
            dalt = d_raio
        else:
            dlat = d_raio / R_LAT
            dlon = d_raio / (R_LAT * np.cos(np.radians(lat)))
            dalt = 5 * d_raio
            if dalt > 30000:
                dalt = 30000

        lb[i, 0] = lat - dlat
        lb[i, 1] = lon - dlon
        lb[i, 2] = alt - dalt

        if not sistema_cartesiano:

            if lb[i, 2] < 0:
                lb[i, 2] = 0

        ub[i, 0] = lat + dlat
        ub[i, 1] = lon + dlon
        ub[i, 2] = alt + dalt

    return lb.flatten(), ub.flatten()

@jit(nopython=True, cache=True, fastmath=True)
def gera_limites_iniciais(detections: np.ndarray,
                          min_lat: np.float64,
                          max_lat: np.float64,
                          min_lon: np.float64,
                          max_lon: np.float64,
                          min_alt: np.float64,
                          max_alt: np.float64) -> tuple:
    """
    Optimized version using Numba for generating initial search bounds
    
    Parameters
    ----------
    detections: np.ndarray
        detections array (Initial M=N)
    num_sensors : np.int32
        Number of sensors.
    min_lat : np.float64
        Minimum latitude.
    max_lat : np.float64
        Maximum latitude.
    min_lon : np.float64
        Minimum longitude.
    max_lon : np.float64
        Maximum longitude.
    min_alt : np.float64
        Minimum altitude.
    max_alt : np.float64
        Maximum altitude.

    Returns
    -------
    tuple of np.ndarray
        lb : np.ndarray
            Lower bounds of the search region for each cluster.
        ub : np.ndarray
            Upper bounds of the search region for each cluster.

    """
    ub_elem = np.array([max_lat, max_lon, max_alt], dtype=np.float64)
    lb_elem = np.array([min_lat, min_lon, min_alt], dtype=np.float64)
    ub = np.empty(detections.shape, dtype=np.float64)
    lb = np.empty(detections.shape, dtype=np.float64)
    for i in range(detections.shape[0]):
        ub[i] = ub_elem.copy()
        lb[i] = lb_elem.copy()
    
    return lb.flatten(), ub.flatten()


if __name__ == "__main__":

    # Recriar os pontos exemplo
    pontos_exemplo = np.array([
        [-25.0, -49.0, 800.0],
        [-25.1, -49.1, 1000.0],
        [-25.1, -49.1, 1000.0],
        [-25.1, -49.1, 1000.0],
        [-25.1, -49.1, 1000.0],
        [-25.1, -49.1, 1000.0],
        [-25.1, -49.1, 1000.0],
        [-25.2, -49.2, 1200.0]
    ])

    solucoes_unicas = np.array([1, 1, 1, 0, 0, 0, 0, 0])

    # Teste de verificação
    lb, ub = gera_limites(pontos_exemplo, solucoes_unicas)
    print(lb, ub)
