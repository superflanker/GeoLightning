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
def funcao_log_verossimilhanca_ponderada(deltas: np.ndarray,
                                         pesos: np.ndarray,
                                         sigma:np.ndarray) -> np.float64:
    """
    Computes the log-likelihood under a standard normal distribution 
    with zero mean and standard deviation sigma, ponderated by affinity or attraction.

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
    log_likelihoods = np.sum(const - ((np.dot(pesos, deltas) ** 2) / denom))
    return log_likelihoods