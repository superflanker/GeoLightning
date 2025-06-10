"""
    EELT 7019 - Inteligência Artificial Aplicada
    Algoritmo STELA - Clusterização Temporal (1. Fase)
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
from numba import jit
import numpy as np
from GeoLightning.Utils.Constants import EPSILON_T, CLUSTER_MIN_PTS


@jit(nopython=True, cache=True, fastmath=True)
def calcular_media_clusters(tempos: np.ndarray,
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
def region_query(tempos: np.ndarray,
                 i: np.int32,
                 eps: np.float64) -> np.ndarray:
    """
    Retorna os vizinhos temporais de um ponto dentro de uma janela eps.

    Esta função busca todos os índices de pontos cujos valores estão dentro
    de uma tolerância temporal `eps` a partir do ponto de índice `i`.

    Args:
        tempos (np.ndarray): vetor de tempos estimados (1D)
        i (np.int32): índice do ponto central da busca
        eps (np.float64): tolerância máxima (janela temporal) 
                     para definição de vizinhança

    Returns:
        vizinhos (np.ndarray): vetor contendo os índices dos pontos vizinhos
    """
    vizinhos = []
    for j in range(len(tempos)):
        if np.abs(tempos[j] - tempos[i]) <= eps:
            vizinhos.append(j)
    return np.array(vizinhos)


@jit(nopython=True, cache=True, fastmath=True)
def expand_cluster(tempos: np.ndarray,
                   labels: np.ndarray,
                   visitado: np.ndarray,
                   i: np.int32,
                   vizinhos: np.ndarray,
                   cluster_id: np.int32,
                   eps: np.float64,
                   min_pts: np.int32) -> None:
    """
    Expande um cluster a partir de um ponto núcleo, atribuindo rótulos aos vizinhos.

    Esta função executa a etapa de expansão do algoritmo DBSCAN, incluindo novos
    pontos ao cluster atual com base na densidade de vizinhos e nos critérios de
    proximidade temporal. A expansão é feita iterativamente e inclui novos pontos
    apenas se satisfizerem os requisitos mínimos de densidade.

    Args:
        tempos (np.ndarray): vetor de tempos estimados (1D)
        labels (np.ndarray): vetor com os rótulos de cluster atribuídos a cada ponto
        visitado (np.ndarray): vetor booleano que indica se o ponto já foi visitado
        i (np.int32): índice do ponto núcleo a partir do qual o cluster é expandido
        vizinhos (np.ndarray): vetor com os índices dos pontos vizinhos iniciais
        cluster_id (np.int32): identificador numérico do cluster atual
        eps (np.float64): tolerância máxima (janela temporal)
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
            new_vizinhos = region_query(tempos, j, eps)
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
def clusterizacao_temporal_stela(tempos: np.ndarray,
                                 eps: np.float64 = EPSILON_T,
                                 min_pts: np.int32 = CLUSTER_MIN_PTS) -> tuple:
    """
        Algoritmo de clusterização temporal (fase 1 do STELA) usando DBSCAN 1D.
        Parâmetros:
            tempos (np.ndarray): vetor de tempos de origem estimados (1D)
            eps (np.float64): tolerância máxima em segundos (default = 1.26 microssegundos)
            min_pts (np.int32): número mínimo de pontos para formar um cluster
        Retorna:
            tuple => clusters (np.ndarray): vetor com os rótulos de cluster 
                               atribuídos a cada ponto
                     tempos_medios (np.ndarray): tempos médios de cada cluster
                     detectores (np.ndarray): detectores em cada cluster
    """
    n = len(tempos)
    labels = -1 * np.ones(n, dtype=np.int32)
    cluster_id = 0
    visitado = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        if visitado[i]:
            continue
        visitado[i] = True
        vizinhos = region_query(tempos, i, eps)
        if len(vizinhos) < min_pts:
            labels[i] = -1  # ruído
        else:
            expand_cluster(tempos, labels, visitado, i, vizinhos,
                           cluster_id, eps, min_pts)
            cluster_id += 1

    tempos_medios, detectores = calcular_media_clusters(tempos, labels)
    return labels, tempos_medios, detectores


# Exemplo de uso
if __name__ == "__main__":
    tempos = np.array(
        [1.0e-3, 1.0000012e-3, 1.003e-3, 1.0000024e-3, 1.002e-3, 1.003e-3])
    labels, tempos_medios, detectores = clusterizacao_temporal_stela(tempos)

    for tempo, label in zip(tempos, labels):
        print(f"{tempo:.9f} s -> cluster {label}")

    for i, (t_med, n) in enumerate(zip(tempos_medios, detectores)):
        print(f"Cluster {i}: média = {t_med:.9f} s, detectores = {n}")
