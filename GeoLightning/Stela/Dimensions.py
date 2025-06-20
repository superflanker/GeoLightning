"""
EELT 7019 - Applied Artificial Intelligence
===========================================

Remapping of Candidate Solutions

Summary
-------
This module provides utility functions for remapping candidate solutions 
after clustering, aiming to reduce the search space for meta-heuristic 
optimization by filtering out converged or duplicate solutions.

These functions are used in the post-processing stage of spatio-temporal 
clustering to refine and prepare inputs for the optimization phase.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- Remapping of candidate solutions using updated cluster labels.
- Construction of uniqueness masks to define solution bounds.

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
def remapeia_solucoes(solucoes: np.ndarray,
                      labels: np.ndarray,
                      centroides: np.ndarray) -> np.ndarray:
    """
    Remaps candidate solutions by filtering out converged ones 
    and reducing the search space.

    This function builds a new array of solutions by preserving only 
    the centroids of clusters and optionally including non-converged 
    solutions marked as noise (-1).

    Parameters
    ----------
    solucoes : np.ndarray
        Array of shape (N, D) with all candidate solutions.
    labels : np.ndarray
        Cluster labels for each solution (-1 indicates noise or non-converged).
    centroides : np.ndarray
        Array of centroids (unique solutions after clustering).

    Returns
    -------
    np.ndarray
        New array of remapped solutions with reduced dimensionality, 
        combining valid centroids and noise points.
    """
    solucoes_nao_unicas = solucoes[labels == -1]
    if solucoes_nao_unicas.shape[0] != 0:
        new_centroides = np.concatenate((centroides, solucoes_nao_unicas))
    else:
        new_centroides = centroides.copy()
    if new_centroides.shape[0] != solucoes.shape[0]:
        left_solucoes = np.zeros((solucoes.shape[0] -
                                  new_centroides.shape[0],
                                  new_centroides.shape[1]),
                                 dtype=centroides.dtype)
        novas_solucoes = np.concatenate((new_centroides,
                                         left_solucoes))
    else:
        novas_solucoes = new_centroides.copy()
    return novas_solucoes


@jit(nopython=True, cache=True, fastmath=True)
def remapeia_solucoes_unicas(clusters: np.ndarray) -> np.ndarray:
    """
    Constructs a binary uniqueness mask to define active clusters 
    and prepare search boundaries.

    This function creates an array marking valid clusters with 1 
    and padding the remaining positions with -1 for compatibility 
    with bounding box computation.

    Parameters
    ----------
    clusters : np.ndarray
        Array of active cluster labels for each solution.

    Returns
    -------
    np.ndarray
        A remapped array of size equal to the original, where unique clusters 
        are marked with 1 and unused slots are filled with -1.
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
