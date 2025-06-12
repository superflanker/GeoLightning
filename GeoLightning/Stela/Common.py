"""
    EELT 7019 - Inteligência Artificial Aplicada
    Funções Comuns a todos os componentes do algoritmo
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from numba import jit
from GeoLightning.Utils.Utils import computa_distancia


@jit(nopython=True, cache=True, fastmath=True)
def calcular_media_clusters_ak(tempos: np.ndarray,
                               labels: np.ndarray) -> tuple:
    """
    Obtém o tempo médio de cada cluster e o número de sensores
    Args: 
        tempos (np.ndarray): vetor de tempos de origem estimados (1D)
        labels (np.ndarray): vetor com os rótulos de cluster atribuídos 
        a cada ponto
    Returns:
        tuple => medias (np.ndarray): os centróides temporais (médias)
                                      de cada cluster
                 detectores (np.ndarray): o número de detectores que
                                      participam da solução
    """
    n_clusters = np.max(labels) + 1
    medias = np.zeros(n_clusters, dtype=np.float64)
    detectores = np.zeros(n_clusters, dtype=np.int32)

    for i in range(len(tempos)):
        lbl = labels[i]
        if lbl >= 0:
            medias[lbl] += tempos[i]
            detectores[lbl] += 1

    for k in range(n_clusters):
        if detectores[k] > 0:
            medias[k] /= detectores[k]
    return medias, detectores


@jit(nopython=True, cache=True, fastmath=True)
def calcular_centroides_ak(solucoes: np.ndarray,
                           labels: np.ndarray) -> np.ndarray:
    """
    Obtém o tempo médio de cada cluster e o número de sensores
    Args:
        solucoes (np.ndarray): vetor de soluções
        mapeamento_tempos_para_solucoes (np.ndarray): 
        labels (np.ndarray): vetor com os rótulos de cluster atribuídos
                a cada ponto
    Returns:
        medias (np.ndarray): os centróides temporais (médias)
                                      de cada cluster
    """
    """mapeamento_tempos_para_solucoes = mapeamento_tempos_para_solucoes.astype(
        dtype=np.int64)"""
    n_clusters = np.int32(np.max(labels) + 1)
    medias = np.zeros((n_clusters, solucoes.shape[1]))
    detectores = np.zeros(n_clusters, dtype=np.int32)

    for i in range(labels.shape[0]):
        lbl = labels[i]
        if lbl >= 0:
            for j in range(solucoes.shape[1]):
                medias[lbl, j] += solucoes[lbl, j]
            detectores[lbl] += 1

    for k in range(n_clusters):
        if detectores[k] > 0:
            for j in range(solucoes.shape[1]):
                medias[k, j] /= detectores[k]
                if np.abs(medias[k, j]) < 1e-12:
                    medias[k, j] = 0.0

    return medias, detectores


@jit(nopython=True, cache=True, fastmath=True)
def calcula_distancias_ao_centroide_ak(solucoes: np.ndarray,
                                       labels: np.ndarray,
                                       centroides: np.ndarray,
                                       sistema_cartesiano: bool = False) -> np.ndarray:
    """
        Calcula o delta D para cálculos de verossimilhança
        Args:
            solucoes (np.ndarray): vetor de soluções
            labels (np.ndarray): vetor com os rótulos de cluster atribuídos
                    a cada ponto
            centroides (np.ndarray): os centrpoides de cada cluster
        Returns:
            delta_d (np.ndarray): vetor com as diferenças de distância
    """
    distancias = np.zeros(len(np.argwhere(labels >= 0)))
    d_idx = 0
    for i in range(labels.shape[0]):
        if labels[i] == -1:
            continue
        distancias[d_idx] = computa_distancia(solucoes[labels[i]],
                                              centroides[labels[i]],
                                              sistema_cartesiano)
        d_idx += 1

    return distancias
