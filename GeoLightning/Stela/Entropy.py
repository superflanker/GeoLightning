"""
EELT 7019 - Applied Artificial Intelligence
===========================================

Shannon Entropy Calculation

Summary
-------
This module provides a function to compute the local Shannon entropy 
of a time-of-arrival vector. The entropy value reflects the temporal 
dispersion of the arrivals and can be used as a regularization or 
penalty term in likelihood-based optimization procedures.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- Local entropy computation based on histogram binning.
- Numerical stabilization using epsilon for log(0) avoidance.

Notes
-----
This module is part of the activities of the discipline 
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numba
"""


import numpy as np
from numba import jit

@jit(nopython=True, cache=True, fastmath=True)
def calcular_entropia_local(tempos: np.ndarray, 
                            n_bins: np.int32 = 10, 
                            epsilon: np.float64 = 1e-12) -> np.float64:
    """
    Computes the local Shannon entropy of a vector of arrival times.

    This function estimates the entropy of a 1D time series by discretizing
    it into a histogram and evaluating the distribution of counts. The entropy
    reflects the temporal spread of detections and is useful as a penalty term
    in likelihood optimization.

    Parameters
    ----------
    tempos : np.ndarray
        1D array of time-of-arrival values (in seconds).
    n_bins : int, optional
        Number of histogram bins for discretization. Default is 10.
    epsilon : float, optional
        Small numerical constant added to avoid log(0). Default is 1e-12.

    Returns
    -------
    float
        The estimated Shannon entropy of the arrival time distribution.
    """
    n = len(tempos)
    if n <= 1:
        return 0.0

    # Determinar limites do histograma
    t_min = np.min(tempos)
    t_max = np.max(tempos)
    if t_max == t_min:
        return 0.0

    # Inicializar bins
    hist = np.zeros(n_bins, dtype=np.int32)
    bin_width = (t_max - t_min) / n_bins

    # Contar frequência em cada bin
    for i in range(n):
        idx = int((tempos[i] - t_min) / bin_width)
        if idx == n_bins:  # borda superior
            idx -= 1
        hist[idx] += 1

    # Normalizar para probabilidades
    entropia = 0.0
    total = np.sum(hist)
    for i in range(n_bins):
        if hist[i] > 0:
            p = hist[i] / total
            entropia -= p * np.log(p + epsilon)

    return entropia
if __name__ == "__main__":
    # Exemplo de uso
    tempos_exemplo = np.array([1.001e-5, 1.002e-5, 1.003e-5, 1.100e-5, 1.200e-5])
    entropia_resultado = calcular_entropia_local(tempos_exemplo)
    print(entropia_resultado)