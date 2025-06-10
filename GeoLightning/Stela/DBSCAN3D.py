"""
    EELT 7019 - Inteligência Artificial Aplicada
    Algoritmo STELA - Clusterização Temporal (1. Fase)
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
from numba import jit
import numpy as np
from GeoLightning.Utils.Constants import EPSILON_D, CLUSTER_MIN_PTS
from GeoLightning.Utils.Utils import computa_distancia, \
    computa_distancias, \
    concat_manual
from .TemporalClustering import calcular_media_clusters


@jit(nopython=True, cache=True, fastmath=True)
def calcular_centroides(solucoes: np.ndarray,
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

    for i in range(len(solucoes)):
        lbl = labels[i]
        if lbl >= 0:
            medias[lbl] += solucoes[i]
            detectores[lbl] += 1

    for k in range(n_clusters):
        if detectores[k] > 0:
            medias[k] /= detectores[k]
    return medias


@jit(nopython=True, cache=True, fastmath=True)
def region_query_3D(solucoes: np.ndarray,
                    i: np.int32,
                    eps: np.float64,
                    sistema_cartesiano: bool = False) -> np.ndarray:
    """
        Retorna os vizinhos espaciais de um ponto dentro de uma janela eps.

        Esta função busca todos os índices de pontos cujos valores estão dentro
        de uma tolerância temporal `eps` a partir do ponto de índice `i`.

        Args:
            solucoes (np.ndarray): vetor de soluções
            i (np.int32): índice do ponto central da busca
            eps (np.float64): tolerância máxima (janela espacial) 
                        para definição de vizinhança
            sistema_cartesiano (bool): 

        Returns:
            vizinhos (np.ndarray): vetor contendo os índices dos pontos vizinhos
    """
    vizinhos = []
    for j in range(len(solucoes)):
        if computa_distancia(solucoes[i],
                             solucoes[j],
                             sistema_cartesiano) <= eps:
            vizinhos.append(j)
    return np.array(vizinhos)


@jit(nopython=True, cache=True, fastmath=True)
def expand_cluster_3D(solucoes: np.ndarray,
                      labels: np.ndarray,
                      visitado: np.ndarray,
                      i: np.int32,
                      vizinhos: np.ndarray,
                      cluster_id: np.int32,
                      eps: np.float64,
                      min_pts: np.int32,
                      sistema_cartesiano: bool = False) -> None:
    """
    Expande um cluster a partir de um ponto núcleo, atribuindo rótulos aos vizinhos.

    Esta função executa a etapa de expansão do algoritmo DBSCAN, incluindo novos
    pontos ao cluster atual com base na densidade de vizinhos e nos critérios de
    proximidade espacial. A expansão é feita iterativamente e inclui novos pontos
    apenas se satisfizerem os requisitos mínimos de densidade.

    Args:
        solucoes (np.ndarray): vetor de solucoes estimados (1D)
        labels (np.ndarray): vetor com os rótulos de cluster atribuídos a cada ponto
        visitado (np.ndarray): vetor booleano que indica se o ponto já foi visitado
        i (np.int32): índice do ponto núcleo a partir do qual o cluster é expandido
        vizinhos (np.ndarray): vetor com os índices dos pontos vizinhos iniciais
        cluster_id (np.int32): identificador numérico do cluster atual
        eps (np.float64): tolerância máxima (janela espacial)
        min_pts (np.int32): número mínimo de pontos para formar um cluster válido

    Returns:
        None: a função modifica os vetores `labels` e `visitado` in-place
    """
    labels[i] = cluster_id
    k = 0
    while k < len(vizinhos):
        j = vizinhos[k]
        if not visitado[j]:
            visitado[j] = True
            new_vizinhos = region_query_3D(
                solucoes, j, eps, sistema_cartesiano)
            if len(new_vizinhos) >= min_pts:
                for nb in new_vizinhos:
                    already_in = False
                    for existing in vizinhos:
                        if nb == existing:
                            already_in = True
                            break
                    if not already_in:
                        vizinhos.append(nb)
        if labels[j] == -1:
            labels[j] = cluster_id
        k += 1


@jit(nopython=True, cache=True, fastmath=True)
def clusterizacao_DBSCAN3D(solucoes: np.ndarray,
                           eps: np.float64 = EPSILON_D,
                           min_pts: np.int32 = CLUSTER_MIN_PTS,
                           sistema_cartesiano: bool = False) -> tuple:
    """
        Algoritmo de clusterização temporal (fase 1 do STELA) usando DBSCAN 1D.
        Parâmetros:
            solucoes (np.ndarray): vetor de solucoes de origem estimados (1D)
            eps (np.float64): tolerância máxima em segundos (default = 1.26 microssegundos)
            min_pts (np.int32): número mínimo de pontos para formar um cluster
        Retorna:
            tuple => clusters (np.ndarray): vetor com os rótulos de cluster 
                               atribuídos a cada ponto
                     solucoes_medios (np.ndarray): solucoes médios de cada cluster
                     detectores (np.ndarray): detectores em cada cluster
    """
    n = len(solucoes)
    labels = -1 * np.ones(n, dtype=np.int32)
    cluster_id = 0
    visitado = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        if visitado[i]:
            continue
        visitado[i] = True
        vizinhos = region_query_3D(solucoes,
                                   i,
                                   eps,
                                   sistema_cartesiano)
        if len(vizinhos) < min_pts:
            labels[i] = -1  # ruído
        else:
            expand_cluster_3D(solucoes,
                              labels,
                              visitado,
                              i,
                              vizinhos,
                              cluster_id,
                              eps,
                              min_pts,
                              sistema_cartesiano)
            cluster_id += 1

    centroides = calcular_centroides(solucoes, labels)
    distancias = None
    for i in range(len(centroides)):
        cluster_solucoes = solucoes[labels == i]
        n = len(cluster_solucoes)
        for j in range(len(cluster_solucoes)):
            if distancias is None:
                distancias = computa_distancias(centroides[i],
                                                cluster_solucoes,
                                                sistema_cartesiano)
            else:
                temp_distancias = computa_distancias(centroides[i],
                                                     cluster_solucoes,
                                                     sistema_cartesiano)
                distancias = concat_manual(distancias, temp_distancias)

    (distancias_medias,
     detectores) = calcular_media_clusters(distancias,
                                           labels)
    return labels, \
        centroides, \
        distancias_medias, \
        detectores
