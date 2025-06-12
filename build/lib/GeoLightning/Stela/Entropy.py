"""
    EELT 7019 - Inteligência Artificial Aplicada
    Remapeamento das soluções candidatas
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

import numpy as np
from numba import jit

@jit(nopython=True, cache=True, fastmath=True)
def calcular_entropia_local(tempos: np.ndarray, 
                            n_bins: int = 10, 
                            epsilon: float = 1e-12) -> float:
    """
    Calcula a entropia local de um vetor de tempos de chegada, como medida
    da dispersão temporal. Usado como penalidade para verossimilhança.

    Args:
        tempos (np.ndarray): vetor 1D de tempos de chegada (em segundos)
        n_bins (int): número de bins para discretização do histograma
        epsilon (float): pequeno valor para evitar log(0)

    Returns:
        float: valor da entropia de Shannon
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