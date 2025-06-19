"""
    EELT 7019 - Inteligência Artificial Aplicada
    Remapeamento das soluções candidatas
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from numba import jit


@jit(nopython=True, cache=True, fastmath=True)
def remapeia_solucoes(solucoes: np.ndarray,
                      labels: np.ndarray,
                      centroides: np.ndarray) -> tuple:
    """
        repameia soluções candidatas - retirando as soluções que
        indicam convergência prévia, reduzindo o espaço de busca
        Args:
            solucões (np.ndarray): vetor de soluções candidatas
            labels: (np.ndarray): os novos labels da estrutura 
                representada por soluções
            centroides (np.ndarray): as soluções únicas existentes 
                no espaço de busca
        Returns:
            tuple =>
                novas_soluções (np.ndarray): as soluções com redução 
                    no espaço de busca
                solucoes_unicas (np.ndarray): vetor indicativo
                    de soluções unicas existentes
    """
    solucoes_nao_unicas = solucoes[labels == -1]
    if solucoes_nao_unicas.shape[0] != 0:
        centroides = np.concatenate((centroides, solucoes_nao_unicas))

    if centroides.shape[0] != solucoes.shape[0]:
        left_solucoes = np.zeros((solucoes.shape[0] -
                                  centroides.shape[0],
                                  centroides.shape[1]),
                                 dtype=centroides.dtype)
        novas_solucoes = np.concatenate((centroides,
                                         left_solucoes))
    else:
        novas_solucoes = centroides.copy()
    return novas_solucoes


@jit(nopython=True, cache=True, fastmath=True)
def remapeia_solucoes_unicas(clusters: np.ndarray) -> np.ndarray:
    """
        Remapeamento intermediário para a construção de limites
        Args:
            clusters (np.ndarray): clusters ativos
        Returns:
            solucoes_unicas (np.ndarray): os clusters unicos seguidos das
            soluções não conergentes marcadas com -1
    """
    n_clusters = np.max(clusters) + 1
    new_clusters = np.ones(n_clusters)
    if new_clusters.shape[0] != clusters.shape[0]:
        left_clusters = -np.ones(clusters.shape[0] - new_clusters.shape[0])
        new_clusters = np.concatenate((new_clusters, left_clusters))
    return new_clusters


if __name__ == "__main__":
    solucoes = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                         [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                         [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0],
                         [3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0],
                         [4.0, 4.0, 4.0], [4.0, 4.0, 4.0], [4.0, 4.0, 4.0],
                         [4.0, 4.0, 4.0], [4.0, 4.0, 4.0], [4.0, 4.0, 4.0]])
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2,
                      3, 3, 3, 4, 4, 4, -1, -1, -1])
    centroides = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0],
                           [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    novas_solucoes = remapeia_solucoes(solucoes,
                                       labels,
                                       centroides)
    
    solucoes_unicas = remapeia_solucoes_unicas(labels)
    
    print(novas_solucoes)
    print(solucoes_unicas)
    print(labels)
