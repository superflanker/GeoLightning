"""
    EELT 7019 - Inteligência Artificial Aplicada
    Clusterização Espaço-Temporal
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

from numba import jit
import numpy as np
from GeoLightning.Utils.Utils import computa_distancia
from GeoLightning.Utils.Constants import SIGMA_T, \
    SIGMA_D, \
    EPSILON_T, \
    EPSILON_D, \
    CLUSTER_MIN_PTS
from GeoLightning.Stela.LogLikelihood import funcao_log_verossimilhanca
from GeoLightning.Stela.Entropy import calcular_entropia_local
from GeoLightning.Stela.Dimensions import remapeia_solucoes


@jit(nopython=True, cache=True, fastmath=True)
def calcular_centroides(solucoes: np.ndarray,
                        mapeamento_tempos_para_solucoes: np.ndarray,
                        labels: np.ndarray) -> np.ndarray:
    """
    Obtém o tempo médio de cada cluster e o número de sensores
    Args:
        solucoes (np.ndarray): vetor de soluções
        mapeamento_tempos_para_solucoes (np.ndarray): mapeamento de tempos para soluções
                devido ao remapeamento do espaço de buscas
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

    for i in range(mapeamento_tempos_para_solucoes.shape[0]):
        lbl = labels[i]
        if lbl >= 0:
            for j in range(solucoes.shape[1]):
                medias[lbl, j] += solucoes[mapeamento_tempos_para_solucoes[i], j]
            detectores[lbl] += 1

    for k in range(n_clusters):
        if detectores[k] > 0:
            for j in range(solucoes.shape[1]):
                medias[k, j] /= detectores[k]
                if np.abs(medias[k, j]) < 1e-12:
                    medias[k, j] = 0.0

    return medias, detectores


@jit(nopython=True, cache=True, fastmath=True)
def calcula_distancias_ao_centroide(solucoes: np.ndarray,
                                    mapeamento_tempos_para_solucoes: np.ndarray,
                                    labels: np.ndarray,
                                    centroides: np.ndarray,
                                    sistema_cartesiano: bool = False) -> np.ndarray:
    """
        Calcula o delta D para cálculos de verossimilhança
        Args:
            solucoes (np.ndarray): vetor de soluções
            labels (np.ndarray): vetor com os rótulos de cluster atribuídos
                    a cada ponto
            mapeamento_tempos_para_solucoes (np.ndarray): mapeamento de tempos para soluções
                    devido ao remapeamento do espaço de buscas
            centroides (np.ndarray): os centrpoides de cada cluster
        Returns:
            delta_d (np.ndarray): vetor com as diferenças de distância
    """
    distancias = np.zeros(len(np.argwhere(labels >= 0)))
    d_idx = 0
    for i in range(labels.shape[0]):
        if labels[i] == -1:
            continue
        distancias[d_idx] = computa_distancia(solucoes[mapeamento_tempos_para_solucoes[i]],
                                              centroides[labels[i]],
                                              sistema_cartesiano)
        d_idx += 1

    return distancias


@jit(nopython=True, cache=True)
def region_query(solucoes: np.ndarray,
                 tempos_de_origem: np.ndarray,
                 mapeamento_tempos_para_solucoes: np.ndarray,
                 idx: np.int64,
                 eps_s: np.float64,
                 eps_t: np.float64,
                 sistema_cartesiano: bool) -> np.ndarray:
    """
        Retorna os vizinhos espaço-temporais de um ponto dentro de uma janela (eps_s, eps_t).

        Esta função busca todos os índices de pontos cujos valores estão dentro
        de uma tolerância `eps` a partir do ponto de índice `idx`.

        Args:
            solucoes (np.ndarray): vetor de soluções
            tempos_de_origem (np.ndarray): vetor de tempos de origem
            mapeamento_tempos_para_solucoes (np.ndarray): mapeamento de tempos para soluções
                devido ao remapeamento do espaço de buscas
            idx (np.int32): índice do ponto central da busca
            eps_s (np.float64): tolerância máxima (janela espacial)
                        para definição de vizinhança
            eps_t (np.float64): tolerância máxima (janela temporal)
                        para definição de vizinhança
            sistema_cartesiano (bool): indicativo de sistema geo-referenciado a utilizar

        Returns:
            vizinhos (np.ndarray): vetor contendo os índices dos pontos vizinhos
    """
    vizinhos = []
    for j in range(mapeamento_tempos_para_solucoes.shape[0]):
        if idx == j:
            continue
        dist_espacial = computa_distancia(solucoes[mapeamento_tempos_para_solucoes[idx]],
                                          solucoes[mapeamento_tempos_para_solucoes[j]],
                                          sistema_cartesiano)
        dist_temporal = np.abs(tempos_de_origem[idx] - tempos_de_origem[j])
        if dist_espacial <= eps_s and dist_temporal <= eps_t:
            vizinhos.append(j)
    return np.array(vizinhos)


@jit(nopython=True, cache=True)
def expand_cluster(solucoes: np.ndarray,
                   tempos_de_origem: np.ndarray,
                   mapeamento_tempos_para_solucoes: np.ndarray,
                   labels: np.ndarray,
                   visitado: np.ndarray,
                   vizinhos: np.ndarray,
                   ponto_idx: np.int64,
                   cluster_id: np.int64,
                   eps_s: np.float64,
                   eps_t: np.float64,
                   min_pts: np.int32,
                   sistema_cartesiano: bool = False):
    """
    Expande um cluster a partir de um ponto núcleo, atribuindo rótulos aos vizinhos.

    Esta função executa a etapa de expansão do algoritmo ST-DBSCAN, incluindo novos
    pontos ao cluster atual com base na densidade de vizinhos e nos critérios de
    proximidade espaço-temporal. A expansão é feita iterativamente e inclui novos pontos
    apenas se satisfizerem os requisitos mínimos de densidade.

    Args:
        solucoes (np.ndarray): vetor de soluções
        tempos_de_origem (np.ndarray): vetor de tempos de origem
        mapeamento_tempos_para_solucoes (np.ndarray): mapeamento de tempos para soluções
                devido ao remapeamento do espaço de buscas
        labels (np.ndarray): vetor com os rótulos de cluster atribuídos a cada ponto
        idx (np.int64): índice do ponto núcleo a partir do qual o cluster é expandido
        visitado (np.ndarray): vetor booleano que indica se o ponto já foi visitado
        vizinhos (np.ndarray): vetor com os índices dos pontos vizinhos iniciais
        ponto_idx (np.int64): índice do ponto central
        cluster_id (np.int32): identificador numérico do cluster atual
        eps_s (np.float64): tolerância máxima (janela espacial)
                para definição de vizinhança
        eps_t (np.float64): tolerância máxima (janela temporal)
                para definição de vizinhança
        min_pts (np.int32): número mínimo de pontos
        sistema_cartesiano (bool): indicativo de sistema geo-referenciado a utilizar

    Returns:
        None: a função modifica os vetores `labels` e `visitado` in-place
    """
    labels[ponto_idx] = cluster_id
    i = 0
    while i < len(vizinhos):
        viz_idx = vizinhos[i]
        if not visitado[viz_idx]:
            visitado[viz_idx] = True
            novos_vizinhos = region_query(solucoes,
                                          tempos_de_origem,
                                          mapeamento_tempos_para_solucoes,
                                          viz_idx,
                                          eps_s,
                                          eps_t,
                                          sistema_cartesiano)
            if len(novos_vizinhos) + 1 >= min_pts:
                for nv in novos_vizinhos:
                    if nv not in vizinhos:
                        np.concatenate((vizinhos, nv * np.ones(1)))
        if labels[viz_idx] == -1:
            labels[viz_idx] = cluster_id
        i += 1


@jit(nopython=True, cache=True)
def st_dbscan(solucoes: np.ndarray,
              tempos_de_origem: np.ndarray,
              mapeamento_tempos_para_solucoes: np.ndarray,
              eps_s: np.float64 = EPSILON_D,
              eps_t: np.float64 = EPSILON_T,
              sigma_d: np.float64 = SIGMA_D,
              min_pts: np.int32 = CLUSTER_MIN_PTS,
              sistema_cartesiano: bool = False) -> tuple:
    """
    ST-DBSCAN principal — clusterização com base em distância espaço-temporal.
    Args:
        solucoes (np.ndarray): vetor de soluções
        tempos_de_origem (np.ndarray): vetor de tempos de origem
        mapeamento_tempos_para_solucoes (np.ndarray): mapeamento de tempos para soluções
                devido ao remapeamento do espaço de buscas
        eps_s (np.float64): tolerância máxima (janela espacial)
                para definição de vizinhança
        eps_t (np.float64): tolerância máxima (janela temporal)
                para definição de vizinhança
        min_pts (np.int32): número mínimo de pontos
        sistema_cartesiano (bool): indicativo de sistema geo-referenciado a utilizar

    """
    N = mapeamento_tempos_para_solucoes.shape[0]
    labels = -np.ones(N, dtype=np.int32)
    visitado = np.zeros(N, dtype=np.bool_)
    cluster_id = 0

    for i in range(N):
        if visitado[i]:
            continue

        visitado[i] = True
        vizinhos = region_query(solucoes,
                                tempos_de_origem,
                                mapeamento_tempos_para_solucoes,
                                i,
                                eps_s,
                                eps_t,
                                sistema_cartesiano)

        if len(vizinhos) + 1 < min_pts:
            labels[i] = -1  # Ruído
        else:

            expand_cluster(solucoes,
                           tempos_de_origem,
                           mapeamento_tempos_para_solucoes,
                           labels,
                           visitado,
                           vizinhos,
                           i,
                           cluster_id,
                           eps_s,
                           eps_t,
                           min_pts,
                           sistema_cartesiano)
            cluster_id += 1

    # cálculo de média temporal e espacial

    centroides, detectores = calcular_centroides(solucoes,
                                                 mapeamento_tempos_para_solucoes,
                                                 labels)

    distancias = calcula_distancias_ao_centroide(solucoes,
                                                 mapeamento_tempos_para_solucoes,
                                                 labels,
                                                 centroides)

    verossimilhanca = calcular_entropia_local(tempos_de_origem[labels >= 0]) \
        + funcao_log_verossimilhanca(distancias, sigma_d)

    novas_solucoes = remapeia_solucoes(solucoes, labels, centroides)

    return (labels,
            distancias,
            centroides,
            detectores,
            novas_solucoes,
            verossimilhanca)


if __name__ == "__main__":

    from GeoLightning.Utils.Utils import computa_tempos_de_origem
    num_events = [2, 5, 10, 15, 20, 25, 30, 100, 500, 1000]

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
        solucoes = np.load(file_n_event_positions)
        detection_times = np.load(file_detections_times)
        detection_positions = np.load(file_detections)
        mapeamento_tempos_para_solucoes = np.load(file_spatial_clusters)

        tempos_de_origem = computa_tempos_de_origem(solucoes,
                                                    mapeamento_tempos_para_solucoes,
                                                    detection_times,
                                                    detection_positions)

        (labels,
         distancias,
         centroides,
         detectores,
         novas_solucoes,
         verossimilhanca) = st_dbscan(solucoes,
                                      tempos_de_origem,
                                      mapeamento_tempos_para_solucoes,
                                      EPSILON_D,
                                      EPSILON_T,
                                      SIGMA_D,
                                      CLUSTER_MIN_PTS,
                                      False)

        print(len(np.unique(labels)), verossimilhanca)
