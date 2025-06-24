"""
Log-Likelihood
==============

Log-Likelihood Function - Objective Function

Summary
-------
This module defines the computation of the log-likelihood function
based on the Gaussian distribution, used as a fitness or objective 
function in spatio-temporal event assignment and optimization procedures.

The log-likelihood is computed under the assumption of Gaussian residuals 
centered at zero with known standard deviation, commonly used in 
signal localization, parameter estimation, and inference models.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- Log-likelihood function for normal distribution
- Fitness evaluation for Gaussian residuals
- Likelihood gradients

Notes
-----
This module is part of the activities of the discipline 
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numba
- GeoLightning.Utils.Constants
- GeoLightning.Utils.Utils
"""

from numba import jit
import numpy as np
from GeoLightning.Utils.Constants import AVG_LIGHT_SPEED, SIGMA_D, SIGMA_T, AVG_EARTH_RADIUS
from GeoLightning.Utils.Utils import coordenadas_esfericas_para_cartesianas_batelada

@jit(nopython=True, cache=True, fastmath=True)
def numba_clip(value: np.float64, 
               min_bound: np.float64, 
               max_bound: np.float64) -> np.float64:
    """
    Clamps a scalar value between specified lower and upper bounds.

    This function returns the input value clipped to the interval 
    [`min_bound`, `max_bound`]. It is implemented using Numba for 
    high-performance scenarios involving scalar bound enforcement 
    within JIT-compiled functions.

    Parameters
    ----------
    value : float
        Input scalar value to be clipped.
    min_bound : float
        Lower bound of the valid interval.
    max_bound : float
        Upper bound of the valid interval.

    Returns
    -------
    float
        The clipped value, satisfying min_bound <= result <= max_bound.

    Notes
    -----
    - Equivalent to `np.clip(value, min_bound, max_bound)` for scalars,
      but compatible with Numba's `nopython` mode.
    - This function is particularly useful when used inside other 
      JIT-compiled routines where `np.clip` may not be supported.
    """
    if value < min_bound:
        out = min_bound
    elif value > max_bound:
        out = max_bound
    else:
        out = value
    return out

@jit(nopython=True, cache=True, fastmath=True)
def maxima_log_verossimilhanca(N: np.int32,
                               sigma: np.float64) -> np.float64:
    """
    Computes the maximum log-likelihood under a standard normal distribution 
    with zero mean and standard deviation sigma.

    This function evaluates the sum of log-likelihoods for a number of observations,
    assuming they follow a Gaussian distribution N(0, σ²) and have a Δ = 0 each.

    Formula
    -------
    log(ℒ) = -0.5 * log(2π * σ²) - (Δ² / (2σ²))

    Parameters
    ----------
    N : np.int32
        number of observations
    sigma : float
        Standard deviation σ > 0.

    Returns
    -------
    float
        Total maximum log-likelihood value for the observed deviations.
    """
    const = -np.log(sigma) - 0.5 * np.log(np.pi) - 0.5 * np.log(2)
    return N * const


@jit(nopython=True, cache=True, fastmath=True)
def funcao_log_verossimilhanca(deltas: np.ndarray,
                               sigma: np.float64) -> np.float64:
    """
    Computes the log-likelihood under a standard normal distribution 
    with zero mean and standard deviation sigma.

    This function evaluates the sum of log-likelihoods for a given array 
    of deviations, assuming they follow a Gaussian distribution N(0, σ²).

    Formula
    -------
    log(ℒ) = -0.5 * log(2π * σ²) - (Δ² / (2σ²))

    Parameters
    ----------
    deltas : np.ndarray
        Array of observed deviations Δ.
    sigma : float
        Standard deviation σ > 0.

    Returns
    -------
    float
        Total log-likelihood value for the observed deviations.
    """
    const = -np.log(sigma) - 0.5 * np.log(np.pi) - 0.5 * np.log(2)
    denom = 2 * (sigma ** 2)
    log_likelihoods = np.sum(const - ((deltas ** 2) / denom))
    # log_likelihoods = np.sum(-(deltas ** 2))
    return log_likelihoods


@jit(nopython=True, cache=True, fastmath=True)
def grad_verossimilanca_cartesiana(events_pos: np.ndarray,  # (N, 3)
                                   detectors_pos: np.ndarray,  # (N, 3)
                                   times_of_origin: np.ndarray,  # (N,)
                                   detection_times: np.ndarray,  # (N,)
                                   sigma_d: np.float64 = SIGMA_D,
                                   c: np.float64 = AVG_LIGHT_SPEED) -> tuple:
    """
    Computes the gradient of the spatio-temporal log-likelihood for a 1-to-1 mapping
    between events and detections in Cartesian coordinates.

    Parameters
    ----------
    events_pos : np.ndarray
        Estimated event positions (N, 3)
    detectors_pos : np.ndarray
        Detector positions (N, 3)
    times_of_origin : np.ndarray
        Estimated times of origin (N,)
    detection_times : np.ndarray
        Observed times of arrival (N,)
    sigma_d : float
        Spatial uncertainty
    c : float
        Propagation speed

    Returns
    -------
    grad_pos : np.ndarray
        Gradient of the log-likelihood with respect to position (N, 3)
    """
    N = events_pos.shape[0]
    grad_pos = np.zeros((N, 3))

    for i in range(N):
        diff_pos = detectors_pos[i] - events_pos[i]
        d = np.sqrt(np.sum(diff_pos ** 2)) + 1e-12  # Evita divisão por zero
        residual_d = d - c * (detection_times[i] - times_of_origin[i])

        # Gradiente espacial
        dL_dd = residual_d / (sigma_d ** 2)
        grad_pos[i] = dL_dd * (diff_pos / d)

    return grad_pos


@jit(nopython=True, cache=True, fastmath=True)
def grad_cartesiano_para_esferico(coordinates: np.ndarray,
                                  grad_xyz: np.ndarray,
                                  R: np.float64 = AVG_EARTH_RADIUS) -> np.ndarray:
    """
    Converts the likelihood gradient from Cartesian to spherical coordinates.

    Parameters
    ----------
    coordinates: np.ndarray
        coordinates ([lat, long, alt]) with angles in degrees
    grad_xyz : np.ndarray
        Gradient in Cartesian system (shape (3,))
    R : float
        Mean radius of the Earth (default: 6371000 m)

    Return
    -------
    grad_esf : np.ndarray
        Gradient in spherical system (dφ, dλ, dh), shape (3,)
    """
    rad_lat = np.radians(coordinates[0])
    rad_lon = np.radians(coordinates[1])
    cos_phi = np.cos(rad_lat)
    sin_phi = np.sin(rad_lat)
    cos_lambda = np.cos(rad_lon)
    sin_lambda = np.sin(rad_lon)
    r = R + coordinates[2]

    # Matriz jacobiana transposta
    J_T = np.array([
        [-r * sin_phi * cos_lambda, -r * sin_phi * sin_lambda, r * cos_phi],
        [-r * cos_phi * sin_lambda,  r * cos_phi * cos_lambda, 0],
        [cos_phi * cos_lambda,       cos_phi * sin_lambda,    sin_phi]
    ])

    # Gradiente em coordenadas esféricas (radianos para lat/lon, metros para alt)
    grad_esf_rad = J_T @ grad_xyz

    # Conversão de radianos para graus nas duas primeiras componentes
    grad_esf_deg = np.array([
        np.degrees(grad_esf_rad[0]),
        np.degrees(grad_esf_rad[1]),
        grad_esf_rad[2]  # altitude permanece em metros
    ])

    return grad_esf_deg

@jit(nopython=True, cache=True, fastmath=True)
def raw_amend_positions(events_pos: np.ndarray,
                        detectors_pos: np.ndarray,
                        times_of_origin: np.ndarray,
                        detection_times: np.ndarray,
                        sigma_d: np.float64 = SIGMA_D,
                        c: np.float64 = AVG_LIGHT_SPEED,
                        R: np.float64 = AVG_EARTH_RADIUS,
                        learning_rate: np.float64 = 1e-3) -> np.ndarray:
    
    """
    Amends events positions towards more promissing regions

    Parameters
    ----------
    events_pos : np.ndarray
        Estimated event positions (N, 3)
    detectors_pos : np.ndarray
        Detector positions (N, 3)
    times_of_origin : np.ndarray
        Estimated times of origin (N,)
    detection_times : np.ndarray
        Observed times of arrival (N,)
    sigma_d : float
        Spatial uncertainty
    c : float
        Propagation speed
    R: np.float64
        Average Earth Radius
    learning_rate: np.float64
        learning rate to avoid overflows
    Returns
    -------
    amended_positions : np.ndarray
        amended positions towards more promissing regions
    """
  
    cart_events = coordenadas_esfericas_para_cartesianas_batelada(events_pos)
    cart_detectors = coordenadas_esfericas_para_cartesianas_batelada(detectors_pos)

    grads = grad_verossimilanca_cartesiana(cart_events,
                                           cart_detectors,
                                           times_of_origin,
                                           detection_times,
                                           sigma_d, 
                                           c)

    new_pos = np.empty_like(events_pos)
    for i in range(len(events_pos)):
        spherical_grad = grad_cartesiano_para_esferico(events_pos[i], grads[i], R)
        updated = events_pos[i] + learning_rate * spherical_grad

        # Proteções
        updated[0] = numba_clip(updated[0], -90.0, 90.0)    # latitude
        updated[1] = (updated[1] + 180.0) % 360.0 - 180.0 # longitude normalizada
        updated[2] = numba_clip(updated[2], R, R + 3000)
        new_pos[i] = updated

    return new_pos
