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

Notes
-----
This module is part of the activities of the discipline 
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numba
"""

from numba import jit
import numpy as np


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
    const = -0.5 * np.log(2 * np.pi * sigma ** 2)
    denom = 2 * sigma ** 2
    log_likelihoods = np.sum(const - (deltas ** 2) / denom)
    return log_likelihoods