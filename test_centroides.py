import numpy as np
from numba import jit


@jit(nopython=True, cache=True, fastmath=True)
def computar_medias(solucoes: np.ndarray,
                    labels: np.ndarray) -> np.ndarray:
    """
    Obtém o tempo médio de cada cluster e o número de sensores
    """
    n_clusters = np.int32(np.max(labels) + 1)
    medias = np.zeros((n_clusters, solucoes.shape[1]))
    detectores = np.zeros(n_clusters, dtype=np.int32)

    for i in range(solucoes.shape[0]):
        lbl = labels[i]
        if lbl >= 0:
            for j in range(solucoes.shape[1]):
                medias[lbl, j] += solucoes[i, j]
            detectores[lbl] += 1

    for k in range(n_clusters):
        if detectores[k] > 0:
            for j in range(solucoes.shape[1]):
                medias[k, j] /= detectores[k]
                if np.abs(medias[k, j]) < 1e-12:
                    medias[k, j] = 0.0

    return medias


solucoes = np.array([[0, 0, 0], [0, 0, 0], [2, 2, 2], [3, 3, 3]])
labels = np.array([0, 0, 1, 1])

medias = computar_medias(solucoes, labels)
print(medias)

sensores = np.load("data/sensors.npy")
sensores[:,2] *= 1000
print(sensores)
