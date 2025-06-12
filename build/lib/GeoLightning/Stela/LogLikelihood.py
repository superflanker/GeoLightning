"""
    EELT 7019 - Inteligência Artificial Aplicada
    Log da verossimilhança - Função de Ajuste
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
from numba import jit
import numpy as np


@jit(nopython=True, cache=True, fastmath=True)
def funcao_log_verossimilhanca(deltas: np.ndarray,
                               sigma: float) -> np.float64:
    """
    Calcula o logaritmo da verossimilhança de uma normal padrão com média zero
    e desvio padrão sigma para um vetor de desvios.

    Fórmula:
        log(𝓛) = -0.5 * log(2π * σ²) - (Δ² / (2σ²))

    Args:
        deltas (np.ndarray): vetor de desvios observados (Δ)
        sigma (float): desvio padrão (σ > 0)

    Returns:
        np.ndarray: vetor dos logaritmos das verossimilhanças
    """
    const = -0.5 * np.log(2 * np.pi * sigma ** 2)
    denom = 2 * sigma ** 2
    log_likelihoods = np.sum(const - (deltas ** 2) / denom)
    return log_likelihoods