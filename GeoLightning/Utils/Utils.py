"""
Utilities
=========

Utilities - Atmospheric Event Geolocation
------------------------------------------

This module implements utility functions for computing distances, 
coordinate transformations, and time-of-origin estimations 
in geolocation systems based on Time-of-Arrival (TOA) data.

Author
-------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Summary
--------
The functions included in this module support core tasks such as:
- Conversion of coordinates from degrees to radians
- Computation of distances in spherical and Cartesian systems
- Minimal angular representation
- Estimation of event origin times given sensor positions and signal arrival times

These utilities are optimized using Numba and designed to be compatible 
with large-scale simulations of geophysical or atmospheric detection systems.

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

import numpy as np
from numba import jit
from .Constants import AVG_EARTH_RADIUS, AVG_LIGHT_SPEED

#################################################################
# Utilitários comuns
#################################################################


@jit(nopython=True, cache=True, fastmath=True)
def coordenadas_em_radianos(coordenadas: np.ndarray) -> np.ndarray:
    """
    Converts a vector of spherical coordinates from degrees to radians.

    Parameters
    ----------
    coordenadas : np.ndarray
        Array containing coordinates in degrees, typically in the format 
        [latitude, longitude, altitude].

    Returns
    -------
    np.ndarray
        Array of coordinates in radians (latitude and longitude) with altitude preserved.

    Notes
    -----
    This function assumes angular values are in degrees and performs the conversion
    for latitude and longitude only. The altitude (third component) remains unchanged.
    """
    novas_coordenadas = np.array([np.deg2rad(coordenadas[0]),
                                  np.deg2rad(coordenadas[1]),
                                  coordenadas[2]])
    return novas_coordenadas


@jit(nopython=True, cache=True, fastmath=True)
def coordenadas_em_radianos_batelada(s_coordinates: np.ndarray) -> np.ndarray:
    """
    Converts a batch of spherical coordinates from degrees to radians.

    Parameters
    ----------
    coordenadas : np.ndarray
        Array of shape (N, 3) containing N coordinate vectors in degrees, 
        typically in the format [latitude, longitude, altitude].

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) with coordinates converted to radians 
        (latitude and longitude). Altitude values remain unchanged.

    Notes
    -----
    This function performs the conversion for each vector in the batch. 
    It is optimized for processing multiple coordinate sets efficiently.
    """
    novas_coordenadas = np.copy(s_coordinates)
    for i in range(0, len(s_coordinates)):
        novas_coordenadas[i] = coordenadas_em_radianos(s_coordinates[i])
    return novas_coordenadas


@jit(nopython=True, cache=True, fastmath=True)
def determinacao_angular_minima(angulo: np.float64) -> np.float64:
    """
    Reduces an angle to its minimum principal determination.

    This function normalizes an angle (in radians) to the interval (-π, π].

    Parameters
    ----------
    angulo : np.float64
        Angle in radians to be normalized.

    Returns
    -------
    np.float64
        Normalized angle within the range (-π, π].

    Notes
    -----
    This operation is commonly used in angular difference calculations
    to ensure minimal angular deviation.
    """
    angulo = angulo % (2 * np.pi)
    if angulo > np.pi:
        angulo -= 2 * np.pi
    elif angulo < -np.pi:
        angulo += 2 * np.pi
    return angulo

#################################################################
# Distâncias
#################################################################


@jit(nopython=True, cache=True, fastmath=True)
def coordenadas_esfericas_para_cartesianas(coordinates: np.ndarray) -> np.ndarray:
    """
    Convert spherical (geodetic) coordinates to Cartesian ECEF coordinates.

    Parameters
    ----------
    coordinates : ndarray
        Input vector [latitude, longitude, altitude] in degrees and meters.

    Returns
    -------
    ndarray
        Output vector [x, y, z] in meters (ECEF Cartesian coordinates).
    """
    radius = AVG_EARTH_RADIUS + coordinates[2]
    new_coordinates = np.array([
        np.cos(np.radians(coordinates[0])) *
        np.cos(np.radians(coordinates[1])) * radius,
        np.cos(np.radians(coordinates[0])) *
        np.sin(np.radians(coordinates[1])) * radius,
        np.sin(np.radians(coordinates[0])) * radius
    ], dtype=np.float64)
    return new_coordinates


@jit(nopython=True, cache=True, fastmath=True)
def coordenadas_esfericas_para_cartesianas_batelada(coordinates: np.ndarray) -> np.ndarray:
    """
    Convert spherical (geodetic) coordinates to Cartesian ECEF coordinates - batch version.

    Parameters
    ----------
    coordinates : ndarray
        Input vector [[latitude, longitude, altitude],...] in degrees and meters.

    Returns
    -------
    ndarray
        Output vector [[x, y, z],...] in meters (ECEF Cartesian coordinates).
    """
    new_coordinates = np.empty(coordinates.shape, dtype=coordinates.dtype)
    for i in range(coordinates.shape[0]):
        new_coordinates[i] = coordenadas_esfericas_para_cartesianas(
            coordinates[i])
    return new_coordinates


@jit(nopython=True, cache=True, fastmath=True)
def coordenadas_cartesianas_para_esfericas(coordinates: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian ECEF coordinates to spherical (geodetic) coordinates.

    Parameters
    ----------
    coordinates : ndarray
        Input vector [x, y, z] in meters (ECEF Cartesian coordinates).

    Returns
    -------
    ndarray
        Output vector [latitude, longitude, altitude] in degrees and meters.
    """
    radius = np.sqrt(coordinates.T.dot(coordinates))
    new_coordinates = np.array([
        np.degrees(np.arcsin(coordinates[2] / radius)),
        np.degrees(np.arctan2(coordinates[1], coordinates[0])),
        radius - AVG_EARTH_RADIUS
    ], dtype=np.float64)
    return new_coordinates


@jit(nopython=True, cache=True, fastmath=True)
def coordenadas_cartesianas_para_esfericas_batelada(coordinates: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian ECEF coordinates to spherical (geodetic) coordinates - batch version.

    Parameters
    ----------
    coordinates : ndarray
        Input vector [[x, y, z], ...] in meters (ECEF Cartesian coordinates).

    Returns
    -------
    ndarray
        Output vector [[latitude, longitude, altitude], ...] in degrees and meters.
    """
    new_coordinates = np.empty(coordinates.shape, dtype=coordinates.dtype)
    for i in range(coordinates.shape[0]):
        new_coordinates[i] = coordenadas_cartesianas_para_esfericas(
            coordinates[i])
    return new_coordinates


@jit(nopython=True, cache=True, fastmath=True)
def distancia_esferica_entre_pontos(a: np.ndarray,
                                    s: np.ndarray) -> np.float64:
    """
    Computes the distance between two points in spherical coordinates ([lat, lon, alt]).

    This function calculates the 3D distance between two geographic locations
    considering the Earth's curvature and elevation, based on spherical geometry.

    Parameters
    ----------
    a : np.ndarray
        Coordinate vector [latitude, longitude, altitude] of the event (in degrees and meters).
    s : np.ndarray
        Coordinate vector [latitude, longitude, altitude] of the station or detector (in degrees and meters).

    Returns
    -------
    np.float64
        Estimated distance between the two points `a` and `s` (in meters).

    Notes
    -----
    Altitudes are added to the average Earth radius to compute an effective radial component.
    Latitude and longitude are internally converted to radians.
    """

    r_a = coordenadas_em_radianos(a)
    r_s = coordenadas_em_radianos(s)
    alt_a = AVG_EARTH_RADIUS + a[2]
    alt_s = AVG_EARTH_RADIUS + s[2]
    delta_alt = alt_a - alt_s
    half_delta_lat = (r_a[0] - r_s[0]) / 2.0
    half_delta_long = (r_a[1] - r_s[1]) / 2.0
    lat_a = r_a[0]
    lat_s = r_s[0]
    arg = 4 * alt_a * alt_s * (np.power(np.sin(half_delta_lat), 2.0)
                               + np.cos(lat_a) * np.cos(lat_s) *
                               np.power(np.sin(half_delta_long), 2.0))
    + np.power(delta_alt, 2.0)
    return np.sqrt(arg)


@jit(nopython=True, cache=True, fastmath=True)
def distancia_cartesiana_entre_pontos(a: np.ndarray,
                                      s: np.ndarray) -> np.float64:
    """
    Computes the distance between two points in Cartesian coordinates ([x, y, z]).

    This function calculates the Euclidean distance between two points in 3D space,
    commonly used in systems where coordinates are expressed in meters.

    Parameters
    ----------
    a : np.ndarray
        Coordinate vector [x, y, z] of the event (in meters).
    s : np.ndarray
        Coordinate vector [x, y, z] of the station or detector (in meters).

    Returns
    -------
    np.float64
        Estimated Euclidean distance between the two points `a` and `s` (in meters).

    Notes
    -----
    This implementation assumes both vectors are in the same Cartesian reference frame.
    """
    temp = np.subtract(a, s)
    return np.sqrt(temp.dot(temp.T))


@jit(nopython=True, cache=True, fastmath=True)
def computa_distancia(a: np.ndarray,
                      s: np.ndarray,
                      sistema_cartesiano: bool = False) -> np.float64:
    """
    Computes the distance between two points, depending on the coordinate system.

    This function selects either spherical or Cartesian distance calculation based
    on the `sistema_cartesiano` flag, enabling compatibility with both geodetic and 
    Euclidean spatial models.

    Parameters
    ----------
    a : np.ndarray
        Coordinate vector of the event (either [lat, lon, alt] in degrees/meters or [x, y, z] in meters).
    s : np.ndarray
        Coordinate vector of the station or detector (same format as `a`).
    sistema_cartesiano : bool, optional
        If True, uses Euclidean (Cartesian) distance. If False, uses spherical distance (default: False).

    Returns
    -------
    np.float64
        Estimated distance between the two points `a` and `s`, in meters.

    Notes
    -----
    - For spherical coordinates, this function assumes latitude and longitude in degrees and altitude in meters.
    - For Cartesian coordinates, standard Euclidean geometry is used.
    """
    if sistema_cartesiano:
        return distancia_cartesiana_entre_pontos(
            a, s)
    return distancia_esferica_entre_pontos(a, s)


@jit(nopython=True, cache=True, fastmath=True)
def computa_distancias(origem: np.ndarray,
                       destinos: np.ndarray,
                       sistema_cartesiano: bool = False) -> np.ndarray:
    """
    Computes the distances between a single origin point and multiple destination points.

    This function is typically used in the computation of spatio-temporal likelihood,
    where distance between event candidates and sensors must be evaluated either in
    spherical or Cartesian coordinates.

    Parameters
    ----------
    origem : np.ndarray
        The origin point coordinates (either [lat, lon, alt] in degrees/meters or [x, y, z] in meters).
    destinos : np.ndarray
        An array of destination point coordinates. Each row must follow the same format as `origem`.
    sistema_cartesiano : bool, optional
        If True, Euclidean distance is used. If False, spherical distance is used (default: False).

    Returns
    -------
    distancias : np.ndarray
        A 1D array of distances (in meters) from `origem` to each point in `destinos`.

    Notes
    -----
    - This function is optimized for performance using Numba JIT compilation.
    - Compatible with both geographical (spherical) and Cartesian coordinate systems.
    """

    distancias = np.zeros(len(destinos), dtype=np.float64)

    for i in range(len(destinos)):
        if sistema_cartesiano:
            distancias[i] = distancia_cartesiana_entre_pontos(
                origem, destinos[i])
        else:
            distancias[i] = distancia_esferica_entre_pontos(
                origem, destinos[i])

    return distancias


@jit(nopython=True, cache=True, fastmath=True)
def computa_tempos_de_origem(solucoes: np.ndarray,
                             tempos_de_chegada: np.ndarray,
                             pontos_de_deteccao: np.ndarray,
                             sistema_cartesiano: bool = False) -> np.ndarray:
    """
    Computes the origin times for each (solution, sensor, arrival_time) triplet.

    This function estimates the event emission time by subtracting the travel time 
    (based on the distance between the event candidate and the detection point) from 
    the signal's arrival time at each sensor.

    Parameters
    ----------
    solucoes : np.ndarray
        Array of shape (N, 3) containing the spatial coordinates of candidate event locations..
    tempos_de_chegada : np.ndarray
        Array of shape (M,) with the absolute arrival times of signals at the sensors.
    pontos_de_deteccao : np.ndarray
        Array of shape (M, 3) containing the spatial coordinates of the sensors (may include repetitions).
    sistema_cartesiano : bool, optional
        If True, uses Cartesian distances; if False, uses spherical distances (default: False).

    Returns
    -------
    np.ndarray
        A 1D array of shape (M,) with the estimated origin times for each detection.

    Notes
    -----
    - The origin time is computed as: `t_origin = t_arrival - distance / c`, 
      where `c` is the average propagation speed (typically the speed of light).
    - Suitable for use in spatio-temporal clustering and likelihood evaluation routines.
    - Optimized using Numba for high-performance numerical computation.
    """

    N = len(solucoes)
    distancias = np.zeros(N, dtype=np.float64)

    for i in range(N):
        if sistema_cartesiano:
            distancias[i] = distancia_cartesiana_entre_pontos(
                solucoes[i], pontos_de_deteccao[i])
        else:
            distancias[i] = distancia_esferica_entre_pontos(
                solucoes[i], pontos_de_deteccao[i])

    tempos_de_origem = tempos_de_chegada - distancias / AVG_LIGHT_SPEED

    return tempos_de_origem
