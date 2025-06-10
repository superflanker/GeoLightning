"""
    EELT 7019 - Inteligência Artificial Aplicada
    Remapeamento das soluções candidatas
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from numba import jit
from GeoLightning.Utils.Utils import concat_manual


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
    novas_solucoes = concat_manual(centroides, solucoes_nao_unicas)
    solucoes_unicas = concat_manual(np.ones(len(centroides)),
                                    np.zeros(len(solucoes_nao_unicas)))
    return novas_solucoes, solucoes_unicas
