"""
    EELT 7019 - Inteligência Artificial Aplicada
    Algoritmo DBSCAN adaptado para numba e 1 dimensão
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
from numba import jit
import numpy as np
from GeoLightning.Utils.Constants import EPSILON_T, CLUSTER_MIN_PTS


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
                        np.concatenate((vizinhos, nb * np.ones(1)))
        if labels[j] == -1:
            labels[j] = cluster_id
        k += 1


@jit(nopython=True, cache=True, fastmath=True)
def clusterizacao_temporal_stela(tempos: np.ndarray,
                                 eps: np.float64 = EPSILON_T,
                                 min_pts: np.int32 = CLUSTER_MIN_PTS) -> np.ndarray:
    """
        Algoritmo de clusterização temporal (fase 1 do STELA) usando DBSCAN 1D.
        Parâmetros:
            tempos (np.ndarray): vetor de tempos de origem estimados (1D)
            eps (np.float64): tolerância máxima em segundos (default = 1.26 microssegundos)
            min_pts (np.int32): número mínimo de pontos para formar um cluster
        Retorna:
            lusters (np.ndarray): vetor com os rótulos de cluster 
                               atribuídos a cada ponto
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

    return labels


# Exemplo de uso
if __name__ == "__main__":

    from GeoLightning.Utils.Utils import computa_tempos_de_origem
    from time import perf_counter

    num_events = [2, 5, 10, 15, 20, 25, 
                30, 100, 500, 800, 1000, 
                2000, 3000, 4000, 5000, 6000, 
                7000, 8000, 9000, 10000]

    for i in range(len(num_events)):

        print("Events: {:d}".format(num_events[i]))
        
        file_detections = "../../data/static_constellation_detections_{:06d}.npy".format(
            num_events[i])

        file_detections_times = "../../data/static_constelation_detection_times_{:06d}.npy".format(
            num_events[i])

        file_event_positions = "../../data/static_constelation_event_positions_{:06d}.npy".format(
            num_events[i])

        file_event_times = "../../data/static_constelation_event_times_{:06d}.npy".format(
            num_events[i])

        file_n_event_positions = "../../data/static_constelation_n_event_positions_{:06d}.npy".format(
            num_events[i])

        file_n_event_times = "../../data/static_constelation_n_event_times_{:06d}.npy".format(
            num_events[i])

        file_distances = "../../data/static_constelation_distances_{:06d}.npy".format(
            num_events[i])
        
        file_spatial_clusters = "../../data/static_constelation_spatial_clusters_{:06d}.npy".format(
            num_events[i])

        n_event_times = np.load(file_n_event_times)
        n_event_positions = np.load(file_n_event_positions)
        detection_times = np.load(file_detections_times)
        detection_positions = np.load(file_detections)
        spatial_clustering = np.load(file_spatial_clusters)

        # calculando os tempos de origem
        start_st = perf_counter()

        tempos_de_origem = computa_tempos_de_origem(n_event_positions, 
                                           spatial_clustering, 
                                           detection_times, 
                                           detection_positions)
        labels = clusterizacao_temporal_stela(tempos_de_origem)
        
        end_st = perf_counter()

        print(f"Elapsed time: {end_st - start_st:.6f} seconds")

        print(len(np.unique(labels)))

