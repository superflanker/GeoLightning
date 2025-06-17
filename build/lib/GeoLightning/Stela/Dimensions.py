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
    if len(solucoes_nao_unicas) > 0:
        novas_solucoes = np.concatenate((centroides,
                                         solucoes_nao_unicas))
    else:
        novas_solucoes = centroides.copy()
    return novas_solucoes


@jit(nopython=True, cache=True, fastmath=True)
def remapeia_clusters(clusters: np.ndarray) -> None:
    """
    Remapeia os clusters para baterem com a ordem das soluções
    Args:
        clusters (np.ndarray): o array dos clusters original
    Returns:
        None (modifica in-place)
    """
    n_clusters = np.max(clusters) + 1
    for i in range(clusters.shape[0]):
        if clusters[i] == -1:
            clusters[i] = n_clusters
            n_clusters += 1


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
    remapeia_clusters(labels)

    print(novas_solucoes)
    print(labels)