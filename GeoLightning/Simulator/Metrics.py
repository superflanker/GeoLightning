"""
Performance Metrics
===================

This module defines the performance metrics used in atmospheric discharge geolocation experiments.

Summary
-------
The functions implemented in this module are designed to evaluate the 
accuracy and consistency of geolocation algorithms applied to atmospheric 
discharges (e.g., lightning events). It supports metrics such as 
positioning error, timing residuals, likelihood values, and statistical 
criteria for evaluating candidate solutions.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- Metric evaluation functions for estimated positions and times.
- Support for spatial and temporal residuals.
- Adapted for use with likelihood-based geolocation pipelines.

Notes
-----
This module is part of the activities of the discipline 
EELT 7019 - Applied Artificial Intelligence, 
Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numpy.linalg.inv
- numba
- GeoLightning.Utils.Constants (SIGMA_D, SIGMA_T)
"""


import numpy as np
from numpy.linalg import inv
from numba import jit
from GeoLightning.Utils.Constants import SIGMA_D, SIGMA_T


@jit(nopython=True, cache=True, fastmath=True)
def rmse(estimadas: np.ndarray,
         reais: np.ndarray) -> np.float64:
    """
    Calculates the root mean square error (RMSE) between estimated and real values.

    A fundamental parameter for evaluating the accuracy of event location algorithms, the RMSE provides a measure of the average discrepancy between the estimated and real position vectors, and is particularly useful in quantifying spatial accuracy.

    Parameters
    ----------
    estimated : np.ndarray
        Vector of estimated positions
    real : np.ndarray
        Vector of real positions

    Returns
    -------
    np.float64
        Value of the root mean square error (RMSE) between the given vectors.
    """
    return np.sqrt(np.mean((estimadas - reais) ** 2))


@jit(nopython=True, cache=True, fastmath=True)
def mae(estimados: np.ndarray,
        reais: np.ndarray) -> np.float64:
    """
    Calculates the mean absolute error (MAE) between the estimated values ​​and the true values.

    MAE is a widely used metric in the evaluation of prediction models, providing a direct measure of the average magnitude of the errors, without considering their direction.

    Parameters
    ----------
    estimated : np.ndarray
        Vector of estimated values, representing the model predictions.
    actual : np.ndarray
        Vector of actual values, representing the reference or true values.

    Returns
    -------
    np.float64
        Value of the mean absolute error (MAE) between the given vectors.
    """
    return np.mean(np.abs(estimados - reais))


@jit(nopython=True, cache=True, fastmath=True)
def average_mean_squared_error(estimados: np.ndarray,
                               reais: np.ndarray) -> np.float64:
    """
    Calculates the mean squared error (AMSE) between estimated values ​​and real values.

    AMSE (Average Mean Squared Error) is a metric that quantifies the average of the squares of the differences between estimated values ​​and real values, penalizing larger errors with greater severity.

    Parameters
    ----------
    estimated : np.ndarray
        Vector of estimated values, usually from predictions of a model.
    real : np.ndarray
        Vector of real values, considered as reference or ground truth.

    Returns
    -------
    np.float64
        Value of the mean squared error (AMSE) between the provided vectors.
    """
    return np.mean((estimados - reais) ** 2)


@jit(nopython=True, cache=True, fastmath=True)
def mean_location_error(estimados: np.ndarray,
                        reais: np.ndarray) -> np.float64:
    """
    Calculates the mean localization error (MLE) between the estimated values ​​and the real values.

    The MLE (Mean Localization Error) is defined as the average of the Euclidean distances between pairs of estimated and real positions, and is widely used in geolocation problems to quantify spatial accuracy.

    Parameters
    ----------
    estimated : np.ndarray
        Vector of estimated positions
    real : np.ndarray
        Vector of real reference positions, with the same dimension as `estimated`.

    Returns
    -------
    np.float64
        Value of the mean localization error (MLE), expressed in the same unit as the given spatial coordinates.
    """

    return np.mean(estimados - reais)


@jit(nopython=True, cache=True, fastmath=True)
def calcula_prmse(rmse: float,
                  referencia: float) -> float:
    """
    Calculates the percentage root mean square error (PRMSE).

    PRMSE (Percentage Root Mean Square Error) is defined as the percentage ratio between the RMSE value and a reference value (usually the full scale), and is useful for relative performance analysis.

    Parameters
    ----------
    rmse : float
        Absolute value of the root mean square error (RMSE).
    reference : float
        Full scale value or reference adopted for normalization.

    Returns
    -------
    float
        Value of the root mean square error expressed as a percentage (PRMSE), given by: `(rmse / reference) * 100`.
    """

    return 100.0 * rmse / referencia


@jit(nopython=True, cache=True, fastmath=True)
def acuracia_associacao(associacoes_estimadas: np.ndarray,
                        associacoes_reais: np.ndarray) -> np.float64:
    """
    Calculates the accuracy of the association between detections and events.

    The accuracy of the association is defined as the proportion of correct associations between the estimated and real indices, expressing the effectiveness of the algorithm in the task of identifying corresponding events.

    Parameters
    ----------
    estimated_associations : np.ndarray
        One-dimensional vector containing the estimated indices of the associations for each detection.
    real_associations : np.ndarray
        One-dimensional vector containing the real indices of the corresponding associations.

    Returns
    -------
    np.float64
    Value of the accuracy of the association, defined as the ratio between the number of correct associations and the total number of detections.
    """
    return np.mean(associacoes_estimadas == associacoes_reais)


@jit(nopython=True, cache=True, fastmath=True)
def erro_relativo_funcao_ajuste(F_estimado: float,
                                F_referencia: float) -> float:
    """
    Calculates the percentage relative error between the estimated fitting function and a reference.

    The percentage relative error is defined as the percentage difference between the estimated value of the fitting function and a reference value, the latter usually being a benchmark, known optimum value or ideal solution.

    Parameters
    ----------
    F_estimado : float
        Value of the fitting function obtained by the algorithm.
    F_referencia : float
        Reference value used for comparison, such as the theoretical optimum or a benchmark.

    Returns
    -------
    float
    Percentage relative error between the given values, calculated as
    ``100 * abs(F_estimado - F_referencia) / abs(F_referencia)``.
    """

    return np.abs(F_estimado - F_referencia) / np.abs(F_referencia) * 100.0


@jit(nopython=True, cache=True, fastmath=True)
def tempo_execucao(tempo_inicial: float,
                   tempo_final: float) -> float:
    """
    Calculates the total execution time between two time instants.

    This method computes the difference between the end time and the start time,
    returning the total elapsed time in seconds.

    Parameters
    ----------
    start_time : float
        Timestamp of the start of the execution, expressed in seconds.
    end_time : float
        Timestamp of the end of the execution, expressed in seconds.

    Returns
    -------
    float
        Total execution time, in seconds, calculated as the difference
        between `end_time` and `start_time`.
    """
    return tempo_final - tempo_inicial


@jit(nopython=True, cache=True, fastmath=True)
def calcular_crlb_espacial(sigma_d: float = SIGMA_D,
                           N: int = 7) -> np.ndarray:
    """
    Calculates the Cramér-Rao Lower Bound (CRLB) matrix for estimating the position of an event.

    A model of isotropic spatial variance is assumed and distance measurements are subject to Gaussian noise with constant standard deviation. The CRLB matrix expresses the lower bound of the variance of any unbiased estimator of the event's position in three-dimensional space.

    Parameters
    ----------
    sigma_d : float
        Standard deviation of distance measurements (in meters), assumed to be the same for all sensors.
    N : int
        Total number of sensors used in the geolocation system.

    Returns
    -------
    np.ndarray
        CRLB matrix of dimension (3, 3), corresponding to the [x, y, z] components of the position.
    """

    return (sigma_d ** 2 / N) * np.eye(3)


@jit(nopython=True, cache=True, fastmath=True)
def calcular_crlb_temporal(sigma_t: float = SIGMA_T,
                           N: int = 7) -> np.ndarray:
    """
    Computes the Cramér-Rao Lower Bound (CRLB) for estimating the time of origin of an event.

    The CRLB represents the theoretical lower bound on the variance of any unbiased estimator 
    for the temporal origin, under Gaussian noise with standard deviation `sigma_t`.

    Parameters
    ----------
    sigma_t : float, optional
        Standard deviation of the time measurements (in seconds). Default is `SIGMA_T`.
    N : int, optional
        Number of sensors involved in the estimation. Default is 7.

    Returns
    -------
    np.ndarray
        A (1, 1) array representing the CRLB for the origin time estimate.
    """
    return (sigma_t ** 2 / N) * np.eye(1)


@jit(nopython=True, cache=True, fastmath=True)
def calcular_crlb_rmse(crlb: np.ndarray) -> float:
    """
    Computes the root mean square error (RMSE) derived from the CRLB matrix.

    This function calculates the quadratic mean of the trace of the product of 
    the CRLB matrix with itself, normalized by its dimension.

    Parameters
    ----------
    crlb : np.ndarray
        CRLB matrix from which the RMSE is to be computed.

    Returns
    -------
    float
        RMSE value computed from the CRLB matrix.
    """
    return np.sqrt(np.trace(crlb @ crlb)) / crlb.shape[0]



@jit(nopython=True, cache=True, fastmath=True)
def calcular_mean_crlb(crlb: np.ndarray) -> float:
    """
    Computes the mean of the variances along the diagonal of the CRLB matrix.

    This metric represents the average uncertainty (variance) associated with 
    the estimation of each parameter under the CRLB model.

    Parameters
    ----------
    crlb : np.ndarray
        CRLB matrix obtained from the estimation process.

    Returns
    -------
    float
        Mean value of the variances in the CRLB matrix.
    """
    return np.trace(crlb) / crlb.shape[0]

