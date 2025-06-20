"""
EELT 7019 - Applied Artificial Intelligence
===========================================

Bearing Calculation - Destination Point on a Sphere

Summary
-------
This module implements the spherical bearing calculation, allowing the estimation of 
a destination point on a sphere (e.g., Earth) given an initial position, a distance, and an azimuth.

The altitude is assumed to be constant throughout the computation. This function 
is essential in geolocation applications, navigation, and spatial simulation of 
movement or dispersion from known coordinates.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- Conversion of geodetic coordinates to radians
- Spherical forward geodesic problem
- Latitude and longitude destination computation
- Preservation of original altitude

Notes
-----
This module is part of the activities of the discipline 
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numba
- GeoLightning.Utils.Utils
- GeoLightning.Utils.Constants
"""

import numpy as np
from numba import jit
from GeoLightning.Utils.Utils import coordenadas_em_radianos
from GeoLightning.Utils.Constants import AVG_EARTH_RADIUS


@jit(nopython=True, cache=True, fastmath=True)
def destino_esferico(posicao: np.ndarray,
                     distancia: np.float64,
                     azimute_deg: np.float64,
                     raio: np.float64 = AVG_EARTH_RADIUS) -> np.ndarray:
    """
    Computes the destination point over a spherical surface given an initial position,
    a distance, and an azimuth (bearing), assuming constant altitude.

    This function solves the forward geodesic problem on a sphere. It calculates
    the destination latitude and longitude from a given point, distance, and bearing.

    Parameters
    ----------
    posicao : np.ndarray
        Array of shape (3,) representing the initial position in the format 
        [latitude, longitude, altitude], with latitude and longitude in degrees and altitude in meters.
    distancia : np.float64
        Distance to be traveled over the sphere's surface (in meters).
    azimute_deg : np.float64
        Azimuth (bearing) in degrees, measured clockwise from the north.
    raio : np.float64, optional
        Radius of the sphere in meters. Default is the Earth's average radius (6371000 m).

    Returns
    -------
    np.ndarray
        Array of shape (3,) with the destination point in the format 
        [latitude, longitude, altitude], where latitude and longitude are in degrees and 
        the altitude remains the same as the input.
    """

    # Converte para radianos
    r_posicao = coordenadas_em_radianos(posicao)
    lat1 = r_posicao[0]
    long1 = r_posicao[1]
    theta = np.radians(azimute_deg)
    delta = distancia / raio

    # Cálculos
    lat2 = np.arcsin(np.sin(lat1) * np.cos(delta) +
                     np.cos(long1) * np.sin(delta) * np.cos(theta))
    long2 = long1 + np.arctan2(np.sin(theta) * np.sin(delta) * np.cos(lat1),
                               np.cos(delta) - np.sin(lat1) * np.sin(lat2))

    # Converte de volta para graus
    lat2_deg = np.degrees(lat2)
    lon2_deg = np.degrees(long2)
    destino = np.empty(len(posicao))
    destino[0] = lat2_deg
    destino[1] = lon2_deg
    destino[2] = posicao[2]
    return destino


if __name__ == "__main__":

    # Exemplo de teste com vetor de posições
    pontos_iniciais = np.array(
        [[-45.0, 45.0, 100.0], [45.0, -45.0, 100.0]], dtype=np.float64)
    distancias = np.array([200000, 200000], dtype=np.float64)  # em metros
    azimutes = np.array([-45.0, 45.0], dtype=np.float64)        # em graus

    for i in range(len(pontos_iniciais)):
        destino = destino_esferico(pontos_iniciais[i],
                                   distancias[i],
                                   azimutes[i])
        print(destino)
