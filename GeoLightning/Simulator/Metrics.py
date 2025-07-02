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
def rmse(deltas: np.ndarray) -> np.float64:
    """
    Calculates the root mean square error (RMSE) between estimated and real values.

    A fundamental parameter for evaluating the accuracy of event location algorithms, the RMSE provides a measure of the average discrepancy between the estimated and real position vectors, and is particularly useful in quantifying spatial accuracy.

    Parameters
    ----------
    deltas : np.ndarray
        Vector of measurement deltas
    real : np.ndarray
        Vector of real positions

    Returns
    -------
    np.float64
        Value of the root mean square error (RMSE) between the given vectors.
    """
    return np.sqrt(np.mean((deltas) ** 2))


@jit(nopython=True, cache=True, fastmath=True)
def mae(deltas: np.ndarray) -> np.float64:
    """
    Calculates the mean absolute error (MAE) between the estimated values ​​and the true values.

    MAE is a widely used metric in the evaluation of prediction models, providing a direct measure of the average magnitude of the errors, without considering their direction.

    Parameters
    ----------
    deltas : np.ndarray
        Vector of estimated values, representing the model predictions.

    Returns
    -------
    np.float64
        Value of the mean absolute error (MAE) between the given vectors.
    """
    return np.mean(np.abs(deltas))


@jit(nopython=True, cache=True, fastmath=True)
def average_mean_squared_error(deltas: np.ndarray) -> np.float64:
    """
    Calculates the mean squared error (AMSE) between estimated values ​​and real values.

    AMSE (Average Mean Squared Error) is a metric that quantifies the average of the squares of the differences between estimated values ​​and real values, penalizing larger errors with greater severity.

    Parameters
    ----------
    deltas : np.ndarray
        Vector of measurement deltas

    Returns
    -------
    np.float64
        Value of the mean squared error (AMSE) between the provided vectors.
    """
    return np.mean((deltas) ** 2)


@jit(nopython=True, cache=True, fastmath=True)
def mean_location_error(deltas: np.ndarray) -> np.float64:
    """
    Calculates the mean localization error (MLE) between the estimated values ​​and the real values.

    The MLE (Mean Localization Error) is defined as the average of the Euclidean distances between pairs of estimated and real positions, and is widely used in geolocation problems to quantify spatial accuracy.

    Parameters
    ----------
    deltas : np.ndarray
        Vector of measurement deltas

    Returns
    -------
    np.float64
        Value of the mean localization error (MLE), expressed in the same unit as the given spatial coordinates.
    """

    return np.mean(deltas)


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
